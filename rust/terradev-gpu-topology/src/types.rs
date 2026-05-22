use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum PciLocality {
    Pix, // Same PCIe switch
    Pxb, // Same PCIe bridge
    Phb, // Same PCIe host bridge
    Sys, // Different PCIe domain
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GPUDevice {
    pub index: u32,
    pub bus_id: String,
    pub numa_node: Option<u32>,
    pub locality: PciLocality,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NICDevice {
    pub name: String,
    pub pci_address: String,
    pub numa_node: Option<u32>,
    pub rdma_capable: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GPUNICPair {
    pub gpu_index: u32,
    pub nic_name: String,
    pub locality: PciLocality,
    pub score: u32,
}
