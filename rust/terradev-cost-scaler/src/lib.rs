#![allow(non_local_definitions)]

use chrono::{Duration, Utc};
use pyo3::prelude::*;
use pyo3::types::PyDict;
use rust_decimal::prelude::*;
use rust_decimal::Decimal;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CostMetric {
    pub timestamp: i64,
    pub instance_id: String,
    pub cost_usd: Decimal,
    pub gpu_type: String,
    pub region: String,
    pub provider: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScalingDecision {
    pub action: String, // "scale_up", "scale_down", "no_change"
    pub reason: String,
    pub target_instances: usize,
    pub current_instances: usize,
    pub projected_cost_usd: Decimal,
    pub confidence: f64,
}

#[pyclass]
pub struct CostScaler {
    metrics: Vec<CostMetric>,
    budget_usd: Decimal,
    scaling_window_hours: i64,
}

#[allow(non_local_definitions)]
#[pymethods]
impl CostScaler {
    #[new]
    #[pyo3(signature = (budget_usd=1000.0, scaling_window_hours=24))]
    fn new(budget_usd: f64, scaling_window_hours: i64) -> Self {
        Self {
            metrics: Vec::new(),
            budget_usd: Decimal::from_f64(budget_usd).unwrap_or(Decimal::from(1000)),
            scaling_window_hours,
        }
    }

    fn add_metric(&mut self, metric: &PyDict) -> PyResult<()> {
        let timestamp: i64 = match metric.get_item("timestamp") {
            Ok(Some(t)) => t.extract()?,
            Ok(None) => return Err(pyo3::exceptions::PyKeyError::new_err("timestamp not found")),
            Err(_) => return Err(pyo3::exceptions::PyKeyError::new_err("timestamp not found")),
        };
        let instance_id: String = match metric.get_item("instance_id") {
            Ok(Some(i)) => i.extract()?,
            Ok(None) => {
                return Err(pyo3::exceptions::PyKeyError::new_err(
                    "instance_id not found",
                ))
            }
            Err(_) => {
                return Err(pyo3::exceptions::PyKeyError::new_err(
                    "instance_id not found",
                ))
            }
        };
        let cost_usd: f64 = match metric.get_item("cost_usd") {
            Ok(Some(c)) => c.extract()?,
            Ok(None) => return Err(pyo3::exceptions::PyKeyError::new_err("cost_usd not found")),
            Err(_) => return Err(pyo3::exceptions::PyKeyError::new_err("cost_usd not found")),
        };
        let gpu_type: String = match metric.get_item("gpu_type") {
            Ok(Some(g)) => g.extract()?,
            Ok(None) => return Err(pyo3::exceptions::PyKeyError::new_err("gpu_type not found")),
            Err(_) => return Err(pyo3::exceptions::PyKeyError::new_err("gpu_type not found")),
        };
        let region: String = match metric.get_item("region") {
            Ok(Some(r)) => r.extract()?,
            Ok(None) => return Err(pyo3::exceptions::PyKeyError::new_err("region not found")),
            Err(_) => return Err(pyo3::exceptions::PyKeyError::new_err("region not found")),
        };
        let provider: String = match metric.get_item("provider") {
            Ok(Some(p)) => p.extract()?,
            Ok(None) => return Err(pyo3::exceptions::PyKeyError::new_err("provider not found")),
            Err(_) => return Err(pyo3::exceptions::PyKeyError::new_err("provider not found")),
        };

        self.metrics.push(CostMetric {
            timestamp,
            instance_id,
            cost_usd: Decimal::from_f64(cost_usd).unwrap_or(Decimal::ZERO),
            gpu_type,
            region,
            provider,
        });
        Ok(())
    }

    fn calculate_total_cost(&self, window_hours: Option<i64>) -> PyResult<PyObject> {
        Python::with_gil(|py| {
            let window = window_hours.unwrap_or(self.scaling_window_hours);
            let cutoff = Utc::now() - Duration::hours(window);
            let cutoff_ts = cutoff.timestamp();

            let total: Decimal = self
                .metrics
                .iter()
                .filter(|m| m.timestamp >= cutoff_ts)
                .map(|m| m.cost_usd)
                .sum();

            let dict = PyDict::new(py);
            dict.set_item("total_cost_usd", total.to_string())?;
            dict.set_item("window_hours", window)?;
            dict.set_item("budget_usd", self.budget_usd.to_string())?;
            dict.set_item(
                "budget_remaining_usd",
                (self.budget_usd - total).to_string(),
            )?;
            dict.set_item(
                "budget_utilization_pct",
                if self.budget_usd > Decimal::ZERO {
                    (total / self.budget_usd * Decimal::from(100)).to_string()
                } else {
                    "0".to_string()
                },
            )?;

            Ok(dict.into())
        })
    }

    fn predict_cost(&self, hours_ahead: i64) -> PyResult<PyObject> {
        Python::with_gil(|py| {
            if self.metrics.len() < 2 {
                return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                    "Insufficient metrics for prediction",
                ));
            }

            let window = self.scaling_window_hours;
            let cutoff = Utc::now() - Duration::hours(window);
            let cutoff_ts = cutoff.timestamp();

            let recent: Vec<&CostMetric> = self
                .metrics
                .iter()
                .filter(|m| m.timestamp >= cutoff_ts)
                .collect();

            if recent.is_empty() {
                return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                    "No recent metrics for prediction",
                ));
            }

            let hourly_rate: Decimal =
                recent.iter().map(|m| m.cost_usd).sum::<Decimal>() / Decimal::from(window);

            let predicted_cost = hourly_rate * Decimal::from(hours_ahead);

            let dict = PyDict::new(py);
            dict.set_item("predicted_cost_usd", predicted_cost.to_string())?;
            dict.set_item("hourly_rate_usd", hourly_rate.to_string())?;
            dict.set_item("hours_ahead", hours_ahead)?;

            Ok(dict.into())
        })
    }

    fn make_scaling_decision(
        &self,
        current_instances: usize,
        target_utilization: Option<f64>,
    ) -> PyResult<PyObject> {
        Python::with_gil(|py| {
            let target_util = target_utilization.unwrap_or(0.7);
            let window = self.scaling_window_hours;
            let cutoff = Utc::now() - Duration::hours(window);
            let cutoff_ts = cutoff.timestamp();

            let recent: Vec<&CostMetric> = self
                .metrics
                .iter()
                .filter(|m| m.timestamp >= cutoff_ts)
                .collect();

            if recent.is_empty() {
                let decision = ScalingDecision {
                    action: "no_change".to_string(),
                    reason: "Insufficient data".to_string(),
                    target_instances: current_instances,
                    current_instances,
                    projected_cost_usd: Decimal::ZERO,
                    confidence: 0.0,
                };
                return Ok(self.decision_to_py(py, decision));
            }

            let total_cost: Decimal = recent.iter().map(|m| m.cost_usd).sum();
            let utilization = if self.budget_usd > Decimal::ZERO {
                total_cost / self.budget_usd
            } else {
                Decimal::ZERO
            };

            let (action, target_instances, reason) =
                if utilization > Decimal::from_f64(target_util).unwrap() {
                    let scale_factor = (utilization / Decimal::from_f64(target_util).unwrap())
                        .to_f64()
                        .unwrap();
                    let new_instances = ((current_instances as f64) * scale_factor).ceil() as usize;
                    (
                        "scale_down".to_string(),
                        new_instances,
                        format!(
                            "Budget utilization {:.1}% exceeds target {:.1}%",
                            utilization * Decimal::from(100),
                            target_util * 100.0
                        ),
                    )
                } else if utilization < Decimal::from_f64(target_util * 0.5).unwrap() {
                    let scale_factor = (Decimal::from_f64(target_util).unwrap() / utilization)
                        .to_f64()
                        .unwrap();
                    let new_instances = ((current_instances as f64) * scale_factor).ceil() as usize;
                    (
                        "scale_up".to_string(),
                        new_instances,
                        format!(
                            "Budget utilization {:.1}% below target {:.1}%",
                            utilization * Decimal::from(100),
                            target_util * 100.0
                        ),
                    )
                } else {
                    (
                        "no_change".to_string(),
                        current_instances,
                        format!(
                            "Budget utilization {:.1}% within target range",
                            utilization * Decimal::from(100)
                        ),
                    )
                };

            let projected_cost =
                total_cost / Decimal::from(window) * Decimal::from(self.scaling_window_hours);

            let decision = ScalingDecision {
                action,
                reason,
                target_instances,
                current_instances,
                projected_cost_usd: projected_cost,
                confidence: (recent.len() as f64 / 100.0).min(1.0),
            };

            Ok(self.decision_to_py(py, decision))
        })
    }

    fn clear(&mut self) {
        self.metrics.clear();
    }
}

impl CostScaler {
    fn decision_to_py(&self, py: Python, decision: ScalingDecision) -> PyObject {
        let dict = PyDict::new(py);
        dict.set_item("action", decision.action).unwrap();
        dict.set_item("reason", decision.reason).unwrap();
        dict.set_item("target_instances", decision.target_instances)
            .unwrap();
        dict.set_item("current_instances", decision.current_instances)
            .unwrap();
        dict.set_item(
            "projected_cost_usd",
            decision.projected_cost_usd.to_string(),
        )
        .unwrap();
        dict.set_item("confidence", decision.confidence).unwrap();
        dict.into()
    }
}

#[pymodule]
fn terradev_cost_scaler(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<CostScaler>()?;
    Ok(())
}
