#![allow(non_local_definitions)]

mod topology;
mod types;

use pyo3::prelude::*;
use topology::GPUNICOptimizer;
use types::{GPUDevice, GPUNICPair, NICDevice, PciLocality};

#[pymodule]
fn terradev_gpu_topology(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<PyGPUNICOptimizer>()?;
    m.add_class::<PyGPUDevice>()?;
    m.add_class::<PyNICDevice>()?;
    m.add_class::<PyGPUNICPair>()?;
    m.add_class::<PyPciLocality>()?;
    Ok(())
}

#[pyclass]
#[allow(non_local_definitions)]
pub struct PyGPUNICOptimizer;

#[allow(non_local_definitions)]
#[pymethods]
impl PyGPUNICOptimizer {
    #[new]
    fn new() -> Self {
        Self
    }

    fn compute_optimal_pairs(
        &self,
        gpus: Vec<PyGPUDevice>,
        nics: Vec<PyNICDevice>,
    ) -> Vec<PyGPUNICPair> {
        let gpu_devices: Vec<GPUDevice> = gpus.into_iter().map(|g| g.into()).collect();
        let nic_devices: Vec<NICDevice> = nics.into_iter().map(|n| n.into()).collect();

        GPUNICOptimizer::compute_optimal_pairs(gpu_devices, nic_devices)
            .into_iter()
            .map(|p| p.into())
            .collect()
    }
}

#[pyclass]
#[derive(Clone)]
pub struct PyGPUDevice {
    #[pyo3(get, set)]
    pub index: u32,
    #[pyo3(get, set)]
    pub bus_id: String,
    #[pyo3(get, set)]
    pub numa_node: Option<u32>,
    #[pyo3(get, set)]
    pub locality: String,
}

impl From<PyGPUDevice> for GPUDevice {
    fn from(p: PyGPUDevice) -> Self {
        Self {
            index: p.index,
            bus_id: p.bus_id,
            numa_node: p.numa_node,
            locality: match p.locality.as_str() {
                "PIX" => PciLocality::Pix,
                "PXB" => PciLocality::Pxb,
                "PHB" => PciLocality::Phb,
                "SYS" => PciLocality::Sys,
                _ => PciLocality::Sys,
            },
        }
    }
}

#[pyclass]
#[derive(Clone)]
pub struct PyNICDevice {
    #[pyo3(get, set)]
    pub name: String,
    #[pyo3(get, set)]
    pub pci_address: String,
    #[pyo3(get, set)]
    pub numa_node: Option<u32>,
    #[pyo3(get, set)]
    pub rdma_capable: bool,
}

impl From<PyNICDevice> for NICDevice {
    fn from(p: PyNICDevice) -> Self {
        Self {
            name: p.name,
            pci_address: p.pci_address,
            numa_node: p.numa_node,
            rdma_capable: p.rdma_capable,
        }
    }
}

#[pyclass]
#[derive(Clone)]
pub struct PyGPUNICPair {
    #[pyo3(get)]
    pub gpu_index: u32,
    #[pyo3(get)]
    pub nic_name: String,
    #[pyo3(get)]
    pub locality: String,
    #[pyo3(get)]
    pub score: u32,
}

impl From<GPUNICPair> for PyGPUNICPair {
    fn from(p: GPUNICPair) -> Self {
        Self {
            gpu_index: p.gpu_index,
            nic_name: p.nic_name,
            locality: match p.locality {
                PciLocality::Pix => "PIX".to_string(),
                PciLocality::Pxb => "PXB".to_string(),
                PciLocality::Phb => "PHB".to_string(),
                PciLocality::Sys => "SYS".to_string(),
            },
            score: p.score,
        }
    }
}

#[pyclass]
#[derive(Clone)]
pub struct PyPciLocality;
