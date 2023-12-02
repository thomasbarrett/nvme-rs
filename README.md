# nvme-rs
The `nvme-rs` package implements a userspace `nvme` driver using the `vfio` framework.

# Getting Started

## Prerequisites
1. Enable iommu support.

Add the following line to `/etc/default/grub`
```
GRUB_CMDLINE_LINUX="intel_iommu=on"
```

Run the following command to update `/boot/grub/grub.cfg`.
```sh
update-grub
```

A reboot is required for the change to take effect.

2. Bind device to the `vfio-pci` driver.

List all devices in the same iommu group as `0000:01:00.0`.
```sh
ls $(dirname $(find /sys/kernel/iommu_groups/ -name '0000:01:00.0'))
```

Bind `0000:01:00.0` to the `vfio-pci` driver. If there is more than one
device in the iommu group, bind all other devices to either the `vfio-pci`
driver or the `pci-stub` driver.
```sh
echo 'vfio-pci' > /sys/bus/pci/devices/0000:01:00.0/driver_override
echo '0000:01:00.0' > /sys/bus/pci/devices/0000:01:00.0/driver/unbind
echo '0000:01:00.0' > /sys/bus/pci/drivers_probe
```

## Installation
1. Clone the repository
```sh
git clone git@github.com:thomasbarrett/nvme-rs.git
```

2. Build from source
```sh
cargo build --release
```
