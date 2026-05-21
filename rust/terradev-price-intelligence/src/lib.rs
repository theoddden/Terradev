#![allow(non_local_definitions)]

use pyo3::prelude::*;
use pyo3::types::PyDict;
use serde::{Deserialize, Serialize};
use rust_decimal::Decimal;
use rust_decimal::prelude::*;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PriceTick {
    pub timestamp: i64,
    pub price: Decimal,
    pub provider: String,
    pub region: String,
    pub gpu_type: String,
    pub availability: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PriceStatistics {
    pub mean: Decimal,
    pub std_dev: Decimal,
    pub min: Decimal,
    pub max: Decimal,
    pub median: Decimal,
    pub percentile_25: Decimal,
    pub percentile_75: Decimal,
    pub count: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PriceTrend {
    pub trend: String, // "up", "down", "stable"
    pub change_pct: Decimal,
    pub volatility: Decimal,
    pub confidence: f64,
}

#[pyclass]
pub struct PriceIntelligence {
    ticks: Vec<PriceTick>,
}

#[allow(non_local_definitions)]
#[pymethods]
impl PriceIntelligence {
    #[new]
    fn new() -> Self {
        Self {
            ticks: Vec::new(),
        }
    }

    fn add_tick(&mut self, tick: &PyDict) -> PyResult<()> {
        let timestamp: i64 = tick.get_item("timestamp")?.unwrap().extract()?;
        let price: f64 = tick.get_item("price")?.unwrap().extract()?;
        let provider: String = tick.get_item("provider")?.unwrap().extract()?;
        let region: String = tick.get_item("region")?.unwrap().extract()?;
        let gpu_type: String = tick.get_item("gpu_type")?.unwrap().extract()?;
        let availability: String = match tick.get_item("availability") {
            Ok(Some(a)) => a.extract().unwrap_or("on-demand".to_string()),
            Ok(None) => "on-demand".to_string(),
            Err(_) => "on-demand".to_string(),
        };

        self.ticks.push(PriceTick {
            timestamp,
            price: Decimal::from_f64(price).unwrap_or(Decimal::ZERO),
            provider,
            region,
            gpu_type,
            availability,
        });
        Ok(())
    }

    fn calculate_statistics(&self, gpu_type: &str, region: Option<&str>) -> PyResult<PyObject> {
        Python::with_gil(|py| {
            let filtered: Vec<&PriceTick> = if let Some(r) = region {
                self.ticks.iter().filter(|t| t.gpu_type == gpu_type && t.region == r).collect()
            } else {
                self.ticks.iter().filter(|t| t.gpu_type == gpu_type).collect()
            };

            if filtered.is_empty() {
                return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                    "No price ticks found for the given criteria"
                ));
            }

            let mut prices: Vec<Decimal> = filtered.iter().map(|t| t.price).collect();
            prices.sort();

            let count = prices.len();
            let mean: Decimal = if count > 0 {
                prices.iter().sum::<Decimal>() / Decimal::from_usize(count).unwrap()
            } else {
                Decimal::ZERO
            };

            let variance: Decimal = if count > 0 {
                let sum_sq_diff: Decimal = prices.iter()
                    .map(|p| (p - mean) * (p - mean))
                    .sum();
                sum_sq_diff / Decimal::from_usize(count).unwrap()
            } else {
                Decimal::ZERO
            };

            let std_dev = if variance >= Decimal::ZERO {
                // Use Newton's method for square root approximation
                let mut approx = variance / Decimal::from(2);
                for _ in 0..20 {
                    let new_approx = (approx + variance / approx) / Decimal::from(2);
                    if (new_approx - approx).abs() < Decimal::from_str("0.0000000001").unwrap_or(Decimal::ZERO) {
                        approx = new_approx;
                        break;
                    }
                    approx = new_approx;
                }
                approx
            } else {
                Decimal::ZERO
            };

            let min = prices.first().unwrap_or(&Decimal::ZERO).clone();
            let max = prices.last().unwrap_or(&Decimal::ZERO).clone();

            let median = if count > 0 {
                let mid = count / 2;
                if count % 2 == 0 {
                    (prices[mid - 1] + prices[mid]) / Decimal::from(2)
                } else {
                    prices[mid]
                }
            } else {
                Decimal::ZERO
            };

            let p25_idx = (count as f64 * 0.25) as usize;
            let p75_idx = (count as f64 * 0.75) as usize;
            let percentile_25 = prices.get(p25_idx).unwrap_or(&Decimal::ZERO).clone();
            let percentile_75 = prices.get(p75_idx).unwrap_or(&Decimal::ZERO).clone();

            let stats = PriceStatistics {
                mean,
                std_dev,
                min,
                max,
                median,
                percentile_25,
                percentile_75,
                count,
            };

            let dict = PyDict::new(py);
            dict.set_item("mean", stats.mean.to_string())?;
            dict.set_item("std_dev", stats.std_dev.to_string())?;
            dict.set_item("min", stats.min.to_string())?;
            dict.set_item("max", stats.max.to_string())?;
            dict.set_item("median", stats.median.to_string())?;
            dict.set_item("percentile_25", stats.percentile_25.to_string())?;
            dict.set_item("percentile_75", stats.percentile_75.to_string())?;
            dict.set_item("count", stats.count)?;

            Ok(dict.into())
        })
    }

    fn calculate_trend(&self, gpu_type: &str, region: Option<&str>, window_minutes: Option<u64>) -> PyResult<PyObject> {
        Python::with_gil(|py| {
            let filtered: Vec<&PriceTick> = if let Some(r) = region {
                self.ticks.iter().filter(|t| t.gpu_type == gpu_type && t.region == r).collect()
            } else {
                self.ticks.iter().filter(|t| t.gpu_type == gpu_type).collect()
            };

            if filtered.len() < 2 {
                return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                    "Insufficient price ticks for trend calculation"
                ));
            }

            let window = window_minutes.unwrap_or(60) * 60; // Convert to seconds
            let now = filtered.last().unwrap().timestamp;
            let cutoff = now - window as i64;

            let recent: Vec<&PriceTick> = filtered.iter()
                .filter(|t| t.timestamp >= cutoff)
                .cloned()
                .collect();

            if recent.len() < 2 {
                return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                    "Insufficient recent price ticks for trend calculation"
                ));
            }

            let first_price = recent.first().unwrap().price;
            let last_price = recent.last().unwrap().price;
            let change_pct = if first_price > Decimal::ZERO {
                ((last_price - first_price) / first_price) * Decimal::from(100)
            } else {
                Decimal::ZERO
            };

            let volatility: Decimal = if recent.len() > 1 {
                let prices: Vec<Decimal> = recent.iter().map(|t| t.price).collect();
                let mean: Decimal = prices.iter().sum::<Decimal>() / Decimal::from_usize(prices.len()).unwrap();
                let variance: Decimal = prices.iter()
                    .map(|p| (p - mean) * (p - mean))
                    .sum::<Decimal>() / Decimal::from_usize(prices.len()).unwrap();
                if variance >= Decimal::ZERO {
                    let mut approx = variance / Decimal::from(2);
                    for _ in 0..20 {
                        let new_approx = (approx + variance / approx) / Decimal::from(2);
                        if (new_approx - approx).abs() < Decimal::from_str("0.0000000001").unwrap_or(Decimal::ZERO) {
                            approx = new_approx;
                            break;
                        }
                        approx = new_approx;
                    }
                    approx
                } else {
                    Decimal::ZERO
                }
            } else {
                Decimal::ZERO
            };

            let trend = if change_pct > Decimal::from(5) {
                "up".to_string()
            } else if change_pct < Decimal::from(-5) {
                "down".to_string()
            } else {
                "stable".to_string()
            };

            let confidence = (recent.len() as f64 / 100.0).min(1.0);

            let trend_data = PriceTrend {
                trend,
                change_pct,
                volatility,
                confidence,
            };

            let dict = PyDict::new(py);
            dict.set_item("trend", trend_data.trend)?;
            dict.set_item("change_pct", trend_data.change_pct.to_string())?;
            dict.set_item("volatility", trend_data.volatility.to_string())?;
            dict.set_item("confidence", trend_data.confidence)?;

            Ok(dict.into())
        })
    }

    fn get_best_price(&self, gpu_type: &str, region: Option<&str>) -> PyResult<PyObject> {
        Python::with_gil(|py| {
            let filtered: Vec<&PriceTick> = if let Some(r) = region {
                self.ticks.iter().filter(|t| t.gpu_type == gpu_type && t.region == r).collect()
            } else {
                self.ticks.iter().filter(|t| t.gpu_type == gpu_type).collect()
            };

            if filtered.is_empty() {
                return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                    "No price ticks found"
                ));
            }

            let best = filtered.iter().min_by(|a, b| a.price.cmp(&b.price)).unwrap();

            let dict = PyDict::new(py);
            dict.set_item("price", best.price.to_string())?;
            dict.set_item("provider", &best.provider)?;
            dict.set_item("region", &best.region)?;
            dict.set_item("availability", &best.availability)?;
            dict.set_item("timestamp", best.timestamp)?;

            Ok(dict.into())
        })
    }

    fn clear(&mut self) {
        self.ticks.clear();
    }
}

#[pymodule]
fn terradev_price_intelligence(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<PriceIntelligence>()?;
    Ok(())
}
