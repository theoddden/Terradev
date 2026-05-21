use pyo3::prelude::*;
use pyo3::types::PyDict;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use chrono::Utc;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConsentRecord {
    pub id: String,
    pub data_type: String,
    pub user_id: String,
    pub purpose: String,
    pub granted: bool,
    pub timestamp: i64,
    pub expires_at: Option<i64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PolicyEvaluation {
    pub allowed: bool,
    pub reason: String,
    pub policy_id: String,
}

#[pyclass]
#[allow(non_local_definitions)]
pub struct GovernanceEngine {
    consents: HashMap<String, ConsentRecord>,
    policies: HashMap<String, serde_json::Value>,
}

#[allow(non_local_definitions)]
#[pymethods]
impl GovernanceEngine {
    #[new]
    fn new() -> Self {
        Self {
            consents: HashMap::new(),
            policies: HashMap::new(),
        }
    }

    fn record_consent(&mut self, data_type: String, user_id: String, purpose: String, granted: bool, expires_at: Option<i64>) -> PyResult<String> {
        let id = format!("{}:{}:{}", user_id, data_type, purpose);
        let record = ConsentRecord {
            id: id.clone(),
            data_type,
            user_id,
            purpose,
            granted,
            timestamp: Utc::now().timestamp(),
            expires_at,
        };
        self.consents.insert(id.clone(), record);
        Ok(id)
    }

    fn check_consent(&self, data_type: String, user_id: String, purpose: String) -> PyResult<PyObject> {
        Python::with_gil(|py| {
            let id = format!("{}:{}:{}", user_id, data_type, purpose);
            let now = Utc::now().timestamp();

            if let Some(record) = self.consents.get(&id) {
                let expired = record.expires_at.map_or(false, |exp| exp < now);
                let allowed = record.granted && !expired;

                let dict = PyDict::new(py);
                dict.set_item("allowed", allowed)?;
                dict.set_item("granted", record.granted)?;
                dict.set_item("expired", expired)?;
                dict.set_item("timestamp", record.timestamp)?;
                dict.set_item("expires_at", record.expires_at)?;
                Ok(dict.into())
            } else {
                let dict = PyDict::new(py);
                dict.set_item("allowed", false)?;
                dict.set_item("granted", false)?;
                dict.set_item("expired", false)?;
                dict.set_item("reason", "No consent record found")?;
                Ok(dict.into())
            }
        })
    }

    fn add_policy(&mut self, policy_id: String, policy: &PyDict) -> PyResult<()> {
        let policy_value: serde_json::Value = Python::with_gil(|py| {
            // Convert PyDict to string representation, then parse as JSON
            let policy_str = policy.str().unwrap_or_else(|_| pyo3::types::PyString::new(py, "{}")).to_string();
            serde_json::from_str(&policy_str).unwrap_or_else(|_| serde_json::json!({}))
        });
        self.policies.insert(policy_id, policy_value);
        Ok(())
    }

    fn evaluate_policy(&self, policy_id: String, _context: &PyDict) -> PyResult<PyObject> {
        Python::with_gil(|py| {
            if let Some(policy) = self.policies.get(&policy_id) {
                // Simple policy evaluation - in real implementation would use OPA or similar
                let allowed = policy.get("default_allow")
                    .and_then(|v| v.as_bool())
                    .unwrap_or(false);

                let eval = PolicyEvaluation {
                    allowed,
                    reason: if allowed { "Policy allows action".to_string() } else { "Policy denies action".to_string() },
                    policy_id,
                };

                let dict = PyDict::new(py);
                dict.set_item("allowed", eval.allowed)?;
                dict.set_item("reason", &eval.reason)?;
                dict.set_item("policy_id", &eval.policy_id)?;
                Ok(dict.into())
            } else {
                Err(PyErr::new::<pyo3::exceptions::PyValueError, _>("Policy not found"))
            }
        })
    }

    fn get_consent_history(&self, user_id: Option<String>) -> PyResult<PyObject> {
        Python::with_gil(|py| {
            let list = pyo3::types::PyList::empty(py);
            
            for record in self.consents.values() {
                if user_id.as_ref().map_or(true, |uid| uid == &record.user_id) {
                    let dict = PyDict::new(py);
                    dict.set_item("id", &record.id)?;
                    dict.set_item("data_type", &record.data_type)?;
                    dict.set_item("user_id", &record.user_id)?;
                    dict.set_item("purpose", &record.purpose)?;
                    dict.set_item("granted", record.granted)?;
                    dict.set_item("timestamp", record.timestamp)?;
                    dict.set_item("expires_at", record.expires_at)?;
                    list.append(dict)?;
                }
            }
            Ok(list.into())
        })
    }

    fn revoke_consent(&mut self, data_type: String, user_id: String, purpose: String) -> PyResult<bool> {
        let id = format!("{}:{}:{}", user_id, data_type, purpose);
        if let Some(record) = self.consents.get_mut(&id) {
            record.granted = false;
            record.timestamp = Utc::now().timestamp();
            Ok(true)
        } else {
            Ok(false)
        }
    }

    fn clear(&mut self) {
        self.consents.clear();
        self.policies.clear();
    }
}

#[pymodule]
fn terradev_governance(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<GovernanceEngine>()?;
    Ok(())
}
