use std::marker::PhantomData;
use std::ptr::{null, null_mut};
use std::{sync::Arc, os::fd::AsRawFd};
use std::path::Path;
use vfio_bindings::bindings::vfio::*;
use vfio_ioctls::{VfioContainer, VfioDevice, VfioError};
use thiserror::Error;
use zerocopy::{AsBytes, FromBytes, FromZeroes};
use std::alloc::{alloc_zeroed, dealloc, Layout};
use vmm_sys_util::eventfd::EventFd;

enum SGLDescriptorType {
    DataBlock = 0x00,
    BitBucket = 0x01,
    Segment = 0x02,
    LastSegment = 0x03,
}


struct VfioBuffer {
    container: Arc<VfioContainer>,
    data: *mut u8,
    len: usize,
    capacity: usize,
}

impl VfioBuffer {
    fn new(container: Arc<VfioContainer>, len: usize) -> Result<Self, VfioError> {
        let capacity = (len + 4095) / 4096 * 4096;
        let prot = libc::PROT_READ | libc::PROT_WRITE;
        let flags = libc::MAP_PRIVATE | libc::MAP_ANONYMOUS;
        let data = unsafe { 
            libc::mmap64(null_mut(), capacity, prot, flags, 0, 0) as *mut u8
        };
        container.vfio_dma_map(data as u64, capacity as u64, data as u64)?;
        Ok(Self { container, data, len, capacity })
    }

    fn as_ptr(&self) -> *const u8 {
        self.data
    }

    fn as_mut_ptr(&self) -> *mut u8 {
        self.data
    }

    fn as_slice(&self) -> &[u8] {
        unsafe { std::slice::from_raw_parts(self.data, self.len) }
    }

    fn as_mut_slice(&mut self) -> &mut [u8] {
        unsafe { std::slice::from_raw_parts_mut(self.data, self.len) }
    }
}

impl Drop for VfioBuffer {
    fn drop(&mut self) {
        self.container.vfio_dma_unmap(self.data as u64, self.capacity as u64).unwrap();
        unsafe { libc::munmap(self.data as *mut libc::c_void, self.capacity) };
    }
}

#[derive(Error, Debug)]
enum AppError {
    #[error("Failed Vfio operation: {0}")]
    VfioError(VfioError),
}

#[derive(FromZeroes, FromBytes, AsBytes, Default, Debug, Clone)]
#[repr(C)]
struct SQE {
    cdw0: u32,
    nsid: u32,
    cdw2: u32,
    cdw3: u32,
    mptr: [u8; 8],
    dptr: [u8; 16],
    cdw10: u32,
    cdw11: u32,
    cdw12: u32,
    cdw13: u32,
    cdw14: u32,
    cdw15: u32,
}

#[derive(FromZeroes, FromBytes, AsBytes, Default, Debug, Clone)]
#[repr(C)]
struct CQE {
    dw0: u32,
    dw1: u32,
    dw2: u32,
    dw3: u32,
}

impl CQE {
    fn cid(&self) -> u16 {
        self.dw3 as u16
    }
    
    fn phase(&self) -> u8 {
        ((self.dw3 >> 16) & 1) as u8
    }

    fn status(&self) -> u16 {
        (self.dw3 >> 17) as u16
    }
}

enum AdminOpcode {
    DeleteIOSubmissionQueue = 0x00,
    CreateIOSubmissionQueue = 0x01,
    DeleteIOCompletionQueue = 0x04,
    CreateIOCompletionQueue = 0x05,
}

enum CreateIOCompletionQueueStatus {
    InvalidQueueIdentifier = 0x01,
    InvalidQueueSize = 0x02,
    InvalidInterruptVector = 0x08,
}

enum NVMOpcode {
    Flush = 0x00,
    Write = 0x01,
    Read = 0x02,
}

fn create_read_sqe(cid: u32, nsid: u32, slba: u64, nlb: u16, dptr: [u8; 16]) -> SQE {
    SQE {
        cdw0: (NVMOpcode::Read as u32) | (cid << 16),
        nsid,
        dptr,
        cdw10: slba as u32,
        cdw11: (slba >> 32) as u32,
        cdw12: (nlb - 1) as u32,
        ..Default::default()
    }
}

const SECTOR_SIZE: usize = 512;
const PAGE_SIZE: usize =4096;

fn create_write_sqe(cid: u32, nsid: u32, slba: u64, nlb: u16, dptr: [u8; 16]) -> SQE {
    SQE {
        cdw0: (NVMOpcode::Write as u32) | (cid << 16),
        nsid: 1,
        dptr,
        cdw10: slba as u32,
        cdw11: (slba >> 32) as u32,
        cdw12: (nlb - 1) as u32,
        ..Default::default()
    }
}

fn create_io_cq(qid: u32, cid: u32, addr: u64, qsize: u32) -> SQE {
    let mut dptr = [0; 16];
    dptr[..8].copy_from_slice(&u64::to_le_bytes(addr as u64));
    dptr[8..16].copy_from_slice(&u64::to_le_bytes((addr as u64) + 4096));
    let iv: u32 = 1;
    let ien: u32 = 1;
    let pc: u32 = 1;
    SQE {
        cdw0: (AdminOpcode::CreateIOCompletionQueue as u32) | (cid << 16),
        dptr: dptr,
        cdw10: qid | ((qsize - 1) << 16),
        cdw11: (pc) | (ien << 1) | (iv << 16),
        ..Default::default()
    }
}

enum CreateIOSubmissionQueueStatus {
    CompletionQueueInvalid = 0x00,
    InvalidQueueIdentifier = 0x01,
    InvalidQueueSize = 0x02,
}

fn create_io_sq(qid: u32, cqid: u32, cid: u32, addr: u64, qsize: u32) -> SQE {
    let mut dptr = [0; 16];
    dptr[..8].copy_from_slice(&u64::to_le_bytes(addr));
    
    let pc: u32 = 1;
    SQE {
        cdw0: (AdminOpcode::CreateIOSubmissionQueue as u32) | (cid << 16),
        dptr: dptr,
        cdw10: qid | ((qsize - 1) << 16),
        cdw11: (pc) | (cqid << 16),
        ..Default::default()
    }
}

struct Controller {
    container: Arc<VfioContainer>,
    dev: VfioDevice,
    cap: Capabilities,
    irqs: Vec<EventFd>,
    bar0: *mut u8,
}

#[derive(Default)]
struct Capabilities {
    mqes: u16,
    to: u8,
    dstrd: u8,
}

impl Controller {
    fn new(container: Arc<VfioContainer>, pci_addr: &str) -> Result<Self, AppError> {
        let path = format!("/sys/bus/pci/devices/{}", pci_addr);
        let dev = VfioDevice::new(&Path::new(&path), container.clone()).map_err(AppError::VfioError)?;

        dev.as_raw_fd();
        let bar0 = unsafe { libc::mmap64(
            null_mut(),
            dev.get_region_size(0) as usize,
            libc::PROT_READ | libc::PROT_WRITE,
            libc::MAP_SHARED,
            dev.as_raw_fd(),
            dev.get_region_offset(0) as i64,
        ) as *mut u8};

        let irq_count = dev.get_irq_info(VFIO_PCI_MSIX_IRQ_INDEX).unwrap().count;
        let irqs: Vec<EventFd> = (0..irq_count).map(|_| EventFd::new(0).unwrap()).collect();
        dev.enable_msix(irqs.iter().collect()).unwrap();

        let mut res = Self {
            container,
            dev,
            cap: Capabilities::default(),
            irqs,
            bar0,
        };
        
        res.read_capabilities();

        Ok(res)
    }

    fn read_capabilities(&mut self) {
        let mut buf: [u8; 8] = [0; 8];
        self.dev.region_read(0, &mut buf, 0x00);
        let cap = u64::from_le_bytes(buf);
        self.cap.mqes =  cap as u16;
        self.cap.to = (cap >> 24) as u8;
        self.cap.dstrd = ((cap >> 32) & 0xf)as u8;
    }

    fn sqtdbl<'a> (&'a self, y: u32) -> &'a mut u32 {
        let offset = (0x1000 + 2 * y * (4 << self.cap.dstrd)) as usize;
        unsafe{ &mut *(self.bar0.add(offset) as *mut u32) }
    }

    fn cqhdbl<'a> (&'a self, y: u32) -> &'a mut u32 {
        let offset = (0x1000 + (2 * y + 1) * (4 << self.cap.dstrd)) as usize;
        unsafe{ &mut *(self.bar0.add(offset) as *mut u32) }
    }
}

impl Drop for Controller {
    fn drop(&mut self) {
        let _ = unsafe{ libc::munmap(
            self.bar0 as *mut libc::c_void,
            self.dev.get_region_size(0) as usize,
        ) };
    }
}

struct AdminQueue {
    inner: Queue
}

impl AdminQueue {
    fn new(ctlr: Arc<Controller>, size: u32) -> Result<Self, AppError> {
        let mut inner = Queue::new(ctlr.clone(), 0, size)?;

        // Configure the admin sq and cq size.
        let asqs: u32 = TryInto::<u32>::try_into(size).unwrap() - 1;
        let acqs: u32 = TryInto::<u32>::try_into(size).unwrap() - 1;
        let aqa = asqs | (acqs << 16);
        ctlr.dev.region_write(0, &u32::to_le_bytes(aqa), 0x24);

        // Configure the admin sq base address.
        let asqb = inner.sq.as_mut_ptr() as u64;
        ctlr.dev.region_write(0, &u64::to_le_bytes(asqb), 0x28);
        // Configure the admin cq base address.
        let acqb = inner.cq.as_mut_ptr() as u64;
        ctlr.dev.region_write(0, &u64::to_le_bytes(acqb), 0x30);

        // Configure the io sq entry size.
        let iosqes: u32 = 6;
        assert!(std::mem::size_of::<SQE>() == (1 << iosqes));
        // Configure the io cq entry size.

        let iocqes: u32 = 4;
        assert!(std::mem::size_of::<CQE>() == (1 << iocqes));
        let cc: u32 = 1 | (iosqes << 16) | (iocqes << 20);
        ctlr.dev.region_write(0, &u32::to_le_bytes(cc), 0x14);

        {
            let mut buf: [u8; 4] = [0; 4];
            ctlr.dev.region_read(0, &mut buf, 0x14);
            let cc = u32::from_le_bytes(buf);
            let en = cc & 0x1;
            let css = (cc >> 4) & 0x7;
            let mps = (cc >> 7) & 0xf;
            let ams = (cc >> 11) & 0x7;
            let shn = (cc >> 14) & 0x3;
            let iosqes = (cc >> 16) & 0xf;
            let iocqes = (cc >> 20) & 0xf;
            let crime = (cc >> 24) & 0x1;
            println!("cc: {{ en: {}, css: {}, mps: {}, ams: {}, shn: {}, iosqes: {}, iocqes: {}, crime: {} }}", en, css, mps, ams, shn, iosqes, iocqes, crime);
        }
        
        loop {
            let mut buf: [u8; 4] = [0x00; 4];
            ctlr.dev.region_read(0, &mut buf, 0x1c);
            let csts = u32::from_le_bytes(buf);
            if csts & 1 == 1 {
                break;
            }
            std::thread::sleep(std::time::Duration::from_millis(100));
        }
    
        Ok(AdminQueue { inner })
    }

    fn push(&mut self, entry: &SQE) -> Result<(), PushError> {
        self.inner.push(entry)
    }

    fn next(&mut self) -> Option<&CQE> {
        self.inner.next()
    }

    fn submit(&mut self) {
        self.inner.submit()
    }

    fn submit_and_wait(&mut self) {
        self.inner.submit_and_wait()
    }
}

struct Queue {
    ctlr: Arc<Controller>,
    id: u32,
    size: u32,
    phase: u8,
    ring_mask: u16,
    sq: VfioBuffer,
    sq_head: u16,
    sq_tail: u16,
    cq: VfioBuffer,
    cq_head: u16,
    cq_tail: u16,
}

/// An error pushing to the submission queue due to it being full.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub struct PushError;

impl std::fmt::Display for PushError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("submission queue is full")
    }
}

impl std::error::Error for PushError {}

impl Queue {
    fn new(ctlr: Arc<Controller>, id: u32, size: u32) -> Result<Self, AppError> {
        let ring_mask = 0xffff >> (0x10 - size.checked_ilog2().unwrap());

        let sq = VfioBuffer::new(ctlr.container.clone(), (size as usize) * std::mem::size_of::<SQE>()).unwrap();
        let cq = VfioBuffer::new(ctlr.container.clone(), (size as usize) * std::mem::size_of::<CQE>()).unwrap();
       
        Ok(Self { ctlr, id, size, phase: 0, ring_mask, sq, sq_head: 0, sq_tail: 0, cq, cq_head: 0, cq_tail: 0 })
    }

    fn sq_len(&self) -> usize {
        self.sq_tail.wrapping_sub(self.sq_head) as usize
    }

    fn sq_capacity(&self) -> usize {
        self.size as usize
    }

    fn sq_is_full(&self) -> bool {
        self.sq_len() == self.sq_capacity()
    }

    fn cq_is_empty(&self) -> bool {
        self.cq_head == self.cq_tail
    }
 
    fn push(&mut self, entry: &SQE) -> Result<(), PushError> {
        if !self.sq_is_full() {
            unsafe { 
                *((self.sq.as_mut_ptr() as *mut SQE).add((self.sq_tail & self.ring_mask) as usize)) = entry.clone();
            };
            self.sq_tail = self.sq_tail.wrapping_add(1);
            Ok(())
        } else {
            Err(PushError)
        }
    }

    fn update_tail(&mut self) {
        loop {
            let cqe = unsafe { 
                &*((self.cq.as_ptr() as *const CQE).add((self.cq_tail & self.ring_mask) as usize))
            };
            let phase = ((cqe.dw3 >> 16) & 1) as u8;
            if phase != self.phase {
                self.cq_tail = self.cq_tail.wrapping_add(1);
                self.phase = if self.cq_tail & self.ring_mask != 0 { self.phase } else { !self.phase };
            } else {
                break
            }
        }
    }

    fn next(&mut self) -> Option<&CQE> {
        self.update_tail();
        if !self.cq_is_empty() {
            let cqe: &CQE = unsafe { 
                &*((self.cq.as_ptr() as *const CQE).add((self.cq_head & self.ring_mask) as usize))
            };
            self.sq_head = cqe.dw2 as u16;
            self.cq_head = self.cq_head.wrapping_add(1);
            *self.ctlr.cqhdbl(self.id) = self.cq_head as u32;
            Some(cqe)
        } else {
            None
        }
    }

    fn submit(&mut self) {
        *self.ctlr.sqtdbl(self.id) = self.sq_tail as u32;
    }

    fn submit_and_wait(&mut self) {
        self.submit();
        let _ = self.ctlr.irqs[self.id as usize].read();
    }
}

struct PRPList<'a> {
    buf: &'a VfioBuffer,
    list: Option<VfioBuffer>,
}

fn div_ceil(a: usize, b: usize) -> usize {
    (a + (b - 1)) / b
}

impl<'a> PRPList<'a> {
    fn new(buf: &'a VfioBuffer) -> Result<Self, VfioError> {
        let entries = div_ceil(buf.as_slice().len(), PAGE_SIZE) - 1;
        let list = if entries > 0 {
            let n = std::mem::size_of::<u64>();
            let len = n * entries;
            let mut list = VfioBuffer::new(buf.container.clone(), len)?;
            for i in 0..entries {
                list.as_mut_slice()[(n * i) .. (n * (i + 1))].copy_from_slice(&u64::to_le_bytes(buf.as_ptr() as u64 + ((i + 1) * PAGE_SIZE) as u64))
            }

            Some(list)
        } else {
            None
        };

        Ok(PRPList { buf, list })
    }

    fn dptr(&self) -> [u8; 16] {
        let mut res = [0x00 as u8; 16];
        res[0..8].copy_from_slice(&u64::to_le_bytes(self.buf.as_ptr() as u64));
        match &self.list {
            Some(list) => {
                res[8..16].copy_from_slice(&u64::to_le_bytes(list.as_ptr() as u64));
            }
            None => {}
        }

        res
    }
}

fn create_vfio_device(pci_addr: &str) -> Result<(), AppError> {
    let container = Arc::new(VfioContainer::new(None).map_err(AppError::VfioError)?);
    let ctlr = Arc::new(Controller::new(container.clone(), pci_addr)?);
    let mut admin_queue = AdminQueue::new(ctlr.clone(), 32).unwrap();
    
    let io_queue_size = 32;
    let io_queue_id = 1;
    let mut io_queue = Queue::new(ctlr.clone(), io_queue_id, io_queue_size).unwrap();

    let iocqb = io_queue.cq.as_ptr() as u64;
    admin_queue.push(&create_io_cq(io_queue_id, 0, iocqb, io_queue_size as u32)).unwrap();
    admin_queue.submit_and_wait();
    let cqe = admin_queue.next().unwrap();
    assert!(cqe.status() == 0);   

    let iosqb = io_queue.sq.as_ptr() as u64;
    admin_queue.push(&create_io_sq(io_queue_id, 1, 0, iosqb, io_queue_size as u32)).unwrap();
    admin_queue.submit_and_wait();
    let cqe = admin_queue.next().unwrap();
    assert!(cqe.status() == 0);

    let mut buf = VfioBuffer::new(container.clone(), 4096 * 3).unwrap();
    buf.as_mut_slice().iter_mut().for_each(|x| *x = 1);
    let prp = PRPList::new(&buf).unwrap();
    io_queue.push(&create_write_sqe(0, 1, 0, 24, prp.dptr())).unwrap();
    io_queue.submit_and_wait();
    let cqe = io_queue.next().unwrap();
    assert!(cqe.status() == 0);
    drop(prp);

    let buf = VfioBuffer::new(container.clone(), 4096 * 3).unwrap();
    let prp = PRPList::new(&buf).unwrap();
    io_queue.push(&create_read_sqe(0, 1, 0, 24, prp.dptr())).unwrap();
    io_queue.submit_and_wait();
    let cqe = io_queue.next().unwrap();
    assert!(cqe.status() == 0);
    println!("{:?}", buf.as_slice());
    drop(prp);

    // {
    //     let cid = cqe.dw3 as u16;
    //     let phase = ((cqe.dw3 >> 16) & 1) as u8;
    //     let status = (cqe.dw3 >> 17) as u16;
    //     let sc = status as u8;
    //     let sct = (status >> 8) & 0x7;
    //     let crd = (status >> 11) & 0x3;
    //     let m = (status >> 13) & 0x1;
    //     let dnr = (status >> 14) & 0x1;
    //     println!("cid: {:x}, phase: {:x}, status: {{ sc: {:x} sct: {:x} crd: {:x} m: {:x} dnr: {:x}}}", cid, phase, sc, sct, crd, m, dnr);
    // }

    Ok(())
}

fn main() -> Result<(), AppError> {
    create_vfio_device("0000:03:00.0")?;

    Ok(())
}
