#![allow(non_local_definitions)]

mod calculator;
mod types;

use calculator::CostCalculator;
use pyo3::prelude::*;
use types::{CostBreakdown, InstanceType};

#[pymodule]
fn terradev_cost_calculator(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<PyCostCalculator>()?;
    m.add_class::<PyInstanceType>()?;
    m.add_class::<PyCostBreakdown>()?;
    Ok(())
}

#[pyclass]
pub struct PyCostCalculator {
    inner: CostCalculator,
}

#[allow(non_local_definitions)]
#[pymethods]
impl PyCostCalculator {
    #[new]
    fn new() -> Self {
        Self {
            inner: CostCalculator::new(),
        }
    }
    
    fn add_instance_type(&mut self, instance_type: PyInstanceType) {
        self.inner.add_instance_type(instance_type.into());
    }
    
    fn calculate_cost(&self, instance_type_name: String, hours: String, use_spot: bool) -> Option<PyCostBreakdown> {
        let hours_decimal = hours.parse::<rust_decimal::Decimal>().ok()?;
        self.inner.calculate_cost(&instance_type_name, hours_decimal, use_spot).map(|b| b.into())
    }
    
    fn calculate_multi_instance_cost(&self, instances: Vec<(String, String, bool)>) -> PyCostBreakdown {
        let rust_instances: Vec<(String, rust_decimal::Decimal, bool)> = instances
            .into_iter()
            .filter_map(|(name, hours, use_spot)| {
                Some((name, hours.parse().ok()?, use_spot))
            })
            .collect();
        
        self.inner.calculate_multi_instance_cost(&rust_instances).into()
    }
}

#[pyclass]
#[derive(Clone)]
pub struct PyInstanceType {
    #[pyo3(get, set)]
    pub name: String,
    #[pyo3(get, set)]
    pub provider: String,
    #[pyo3(get, set)]
    pub region: String,
    #[pyo3(get, set)]
    pub hourly_cost_usd: String,
    #[pyo3(get, set)]
    pub spot_discount_percent: String,
    #[pyo3(get, set)]
    pub gpu_count: u32,
}

impl From<PyInstanceType> for InstanceType {
    fn from(p: PyInstanceType) -> Self {
        Self {
            name: p.name,
            provider: p.provider,
            region: p.region,
            hourly_cost_usd: p.hourly_cost_usd.parse().unwrap_or_else(|_| rust_decimal::Decimal::ZERO),
            spot_discount_percent: p.spot_discount_percent.parse().unwrap_or_else(|_| rust_decimal::Decimal::ZERO),
            gpu_count: p.gpu_count,
        }
    }
}

#[pyclass]
#[derive(Clone)]
pub struct PyCostBreakdown {
    #[pyo3(get)]
    pub hourly_cost_usd: String,
    #[pyo3(get)]
    pub monthly_cost_usd: String,
    #[pyo3(get)]
    pub spot_hourly_cost_usd: String,
    #[pyo3(get)]
    pub spot_monthly_cost_usd: String,
    #[pyo3(get)]
    pub spot_savings_usd: String,
    #[pyo3(get)]
    pub gpu_count: u32,
}

impl From<CostBreakdown> for PyCostBreakdown {
    fn from(b: CostBreakdown) -> Self {
        Self {
            hourly_cost_usd: b.hourly_cost_usd.to_string(),
            monthly_cost_usd: b.monthly_cost_usd.to_string(),
            spot_hourly_cost_usd: b.spot_hourly_cost_usd.to_string(),
            spot_monthly_cost_usd: b.spot_monthly_cost_usd.to_string(),
            spot_savings_usd: b.spot_savings_usd.to_string(),
            gpu_count: b.gpu_count,
        }
    }
}
