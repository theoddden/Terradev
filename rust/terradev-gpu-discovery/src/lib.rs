#![allow(non_local_definitions)]

use pyo3::prelude::*;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GPUInfo {
    pub index: u32,
    pub name: String,
    pub memory_total: u64,
    pub memory_free: u64,
    pub compute_capability: String,
    pub pci_bus_id: String,
    pub driver_version: String,
    pub cuda_version: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GPUState {
    pub gpus: Vec<GPUInfo>,
    pub total_count: u32,
    pub total_memory: u64,
    pub free_memory: u64,
}

pub struct GPUDiscovery {
    cache: Option<GPUState>,
    cache_time: Option<std::time::Instant>,
    cache_ttl_secs: u64,
}

impl GPUDiscovery {
    pub fn new(cache_ttl_secs: u64) -> Self {
        Self {
            cache: None,
            cache_time: None,
            cache_ttl_secs,
        }
    }

    pub fn discover_gpus(&mut self) -> Result<GPUState, String> {
        let now = std::time::Instant::now();

        if let Some(cache_time) = self.cache_time {
            if now.duration_since(cache_time).as_secs() < self.cache_ttl_secs {
                return Ok(self.cache.clone().unwrap());
            }
        }

        let gpus = self.query_gpus()?;
        let total_count = gpus.len() as u32;
        let total_memory = gpus.iter().map(|g| g.memory_total).sum();
        let free_memory = gpus.iter().map(|g| g.memory_free).sum();

        let state = GPUState {
            gpus,
            total_count,
            total_memory,
            free_memory,
        };

        self.cache = Some(state.clone());
        self.cache_time = Some(now);

        Ok(state)
    }

    fn query_gpus(&self) -> Result<Vec<GPUInfo>, String> {
        #[cfg(feature = "nvml")]
        {
            self.query_gpus_nvml()
        }

        #[cfg(not(feature = "nvml"))]
        {
            self.query_gpus_fallback()
        }
    }

    #[cfg(feature = "nvml")]
    fn query_gpus_nvml(&self) -> Result<Vec<GPUInfo>, String> {
        use nvml_wrapper::Nvml;

        let nvml = Nvml::init().map_err(|e| e.to_string())?;
        let device_count = nvml.device_count().map_err(|e| e.to_string())?;

        let mut gpus = Vec::new();

        for i in 0..device_count {
            let device = nvml.device_by_index(i).map_err(|e| e.to_string())?;
            let name = device.name().map_err(|e| e.to_string())?;
            let memory_info = device.memory_info().map_err(|e| e.to_string())?;
            let compute_capability = device
                .cuda_compute_capability()
                .map(|cc| format!("{}.{}", cc.major, cc.minor))
                .unwrap_or_else(|_| "unknown".to_string());
            let pci_bus_id = device.pci_info().map_err(|e| e.to_string())?.bus_id;
            let driver_version = nvml
                .sys_driver_version()
                .map_err(|e| e.to_string())?
                .to_string();
            let cuda_version = nvml
                .sys_cuda_driver_version()
                .map_err(|e| e.to_string())?
                .to_string();

            gpus.push(GPUInfo {
                index: i,
                name,
                memory_total: memory_info.total,
                memory_free: memory_info.free,
                compute_capability,
                pci_bus_id,
                driver_version,
                cuda_version,
            });
        }

        Ok(gpus)
    }

    #[cfg(not(feature = "nvml"))]
    fn query_gpus_fallback(&self) -> Result<Vec<GPUInfo>, String> {
        let mut gpus = Vec::new();

        if let Ok(output) = std::process::Command::new("nvidia-smi")
            .args(["--query-gpu=index,name,memory.total,memory.free,compute_cap,pci.bus_id,driver_version,cuda_version"])
            .args(["--format=csv,noheader,nounits"])
            .output()
        {
            let stdout = String::from_utf8_lossy(&output.stdout);
            for line in stdout.lines() {
                let parts: Vec<&str> = line.split(',').map(|s| s.trim()).collect();
                if parts.len() >= 8 {
                    if let (Ok(index), Ok(memory_total), Ok(memory_free)) = (
                        parts[0].parse::<u32>(),
                        parts[2].parse::<u64>(),
                        parts[3].parse::<u64>(),
                    ) {
                        gpus.push(GPUInfo {
                            index,
                            name: parts[1].to_string(),
                            memory_total,
                            memory_free,
                            compute_capability: parts[4].to_string(),
                            pci_bus_id: parts[5].to_string(),
                            driver_version: parts[6].to_string(),
                            cuda_version: parts[7].to_string(),
                        });
                    }
                }
            }
        }

        if gpus.is_empty() {
            return Err("No GPUs detected via nvidia-smi".to_string());
        }

        Ok(gpus)
    }

    pub fn get_gpu_by_index(&mut self, index: u32) -> Result<GPUInfo, String> {
        let state = self.discover_gpus()?;
        state
            .gpus
            .into_iter()
            .find(|g| g.index == index)
            .ok_or_else(|| format!("GPU {} not found", index))
    }

    pub fn invalidate_cache(&mut self) {
        self.cache = None;
        self.cache_time = None;
    }
}

#[pyclass]
pub struct PyGPUDiscovery {
    inner: GPUDiscovery,
}

#[allow(non_local_definitions)]
#[allow(non_local_definitions)]
#[pymethods]
impl PyGPUDiscovery {
    #[new]
    #[pyo3(signature = (cache_ttl_secs = 5))]
    fn new(cache_ttl_secs: u64) -> Self {
        Self {
            inner: GPUDiscovery::new(cache_ttl_secs),
        }
    }

    fn discover_gpus(&mut self) -> PyResult<PyObject> {
        let state = self.inner.discover_gpus().map_err(|e| {
            pyo3::exceptions::PyRuntimeError::new_err(format!("GPU discovery failed: {}", e))
        })?;

        Python::with_gil(|py| {
            let dict = pyo3::types::PyDict::new(py);
            
            let gpu_list = pyo3::types::PyList::new(py, state.gpus.iter().map(|gpu| {
                let gpu_dict = pyo3::types::PyDict::new(py);
                gpu_dict.set_item("index", gpu.index).unwrap();
                gpu_dict.set_item("name", &gpu.name).unwrap();
                gpu_dict.set_item("memory_total", gpu.memory_total).unwrap();
                gpu_dict.set_item("memory_free", gpu.memory_free).unwrap();
                gpu_dict.set_item("compute_capability", &gpu.compute_capability).unwrap();
                gpu_dict.set_item("pci_bus_id", &gpu.pci_bus_id).unwrap();
                gpu_dict.set_item("driver_version", &gpu.driver_version).unwrap();
                gpu_dict.set_item("cuda_version", &gpu.cuda_version).unwrap();
                gpu_dict.to_object(py)
            }));
            
            dict.set_item("gpus", gpu_list)?;
            dict.set_item("total_count", state.total_count)?;
            dict.set_item("total_memory", state.total_memory)?;
            dict.set_item("free_memory", state.free_memory)?;
            Ok(dict.into())
        })
    }

    fn get_gpu_by_index(&mut self, index: u32) -> PyResult<PyObject> {
        let gpu = self.inner.get_gpu_by_index(index).map_err(|e| {
            pyo3::exceptions::PyRuntimeError::new_err(format!("GPU query failed: {}", e))
        })?;

        Python::with_gil(|py| {
            let dict = pyo3::types::PyDict::new(py);
            dict.set_item("index", gpu.index)?;
            dict.set_item("name", gpu.name)?;
            dict.set_item("memory_total", gpu.memory_total)?;
            dict.set_item("memory_free", gpu.memory_free)?;
            dict.set_item("compute_capability", gpu.compute_capability)?;
            dict.set_item("pci_bus_id", gpu.pci_bus_id)?;
            dict.set_item("driver_version", gpu.driver_version)?;
            dict.set_item("cuda_version", gpu.cuda_version)?;
            Ok(dict.into())
        })
    }

    fn invalidate_cache(&mut self) {
        self.inner.invalidate_cache();
    }
}

#[pymodule]
fn terradev_gpu_discovery(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<PyGPUDiscovery>()?;
    Ok(())
}
