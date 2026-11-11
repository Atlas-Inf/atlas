// SPDX-License-Identifier: AGPL-3.0-only

//! The exporter opens `/dev/nvidia0`; only the GB10 PCI identity is admitted.
//! Driver version alone cannot prove a display carveout is unused on a GPU.

use std::path::Path;

use anyhow::{Context, Result, bail};

pub(super) fn check_nvidia0() -> Result<()> {
    check_at(
        Path::new("/proc/driver/nvidia/gpus"),
        Path::new("/sys/bus/pci/devices"),
    )
}

fn check_at(proc_gpus: &Path, pci_devices: &Path) -> Result<()> {
    for entry in std::fs::read_dir(proc_gpus).context("enumerate NVIDIA devices")? {
        let entry = entry?;
        let info = std::fs::read_to_string(entry.path().join("information"))?;
        let is_zero = info.lines().any(|line| {
            line.split_once(':')
                .is_some_and(|(key, value)| key.trim() == "Device Minor" && value.trim() == "0")
        });
        if !is_zero {
            continue;
        }
        let pci = pci_devices.join(entry.file_name());
        let vendor = std::fs::read_to_string(pci.join("vendor"))?;
        let device = std::fs::read_to_string(pci.join("device"))?;
        if vendor.trim() != "0x10de" || device.trim() != "0x2e12" {
            bail!(
                "display carveout requires GB10 (10de:2e12), nvidia0 is {}:{}",
                vendor.trim(),
                device.trim()
            );
        }
        return Ok(());
    }
    bail!("could not identify /dev/nvidia0; display carveout refused")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn only_gb10_at_the_exported_device_minor_is_admitted() {
        let root = tempfile::tempdir().unwrap();
        let proc = root.path().join("gpus");
        let pci = root.path().join("pci");
        let bus = "000f:01:00.0";
        std::fs::create_dir_all(proc.join(bus)).unwrap();
        std::fs::create_dir_all(pci.join(bus)).unwrap();
        let info = proc.join(bus).join("information");
        std::fs::write(&info, "Model: Unknown\nDevice Minor: 0\n").unwrap();
        std::fs::write(pci.join(bus).join("vendor"), "0x10de\n").unwrap();
        let device = pci.join(bus).join("device");
        std::fs::write(&device, "0x2e12\n").unwrap();
        assert!(check_at(&proc, &pci).is_ok());
        std::fs::write(&device, "0x2b85\n").unwrap();
        assert!(check_at(&proc, &pci).is_err());
        std::fs::write(&device, "0x2e12\n").unwrap();
        std::fs::write(&info, "Device Minor: 1\n").unwrap();
        assert!(check_at(&proc, &pci).is_err());
        std::fs::remove_file(&device).unwrap();
        std::fs::write(&info, "Device Minor: 0\n").unwrap();
        assert!(check_at(&proc, &pci).is_err());
    }
}
