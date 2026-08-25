use crate::types::{Quota, QuotaError, QuotaRequest};
use parking_lot::RwLock;
use std::collections::HashMap;
use std::sync::Arc;

pub struct QuotaManager {
    quotas: Arc<RwLock<HashMap<String, Quota>>>,
}

impl QuotaManager {
    pub fn new() -> Self {
        Self {
            quotas: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    pub fn set_quota(&self, resource: String, limit: u64) {
        let mut quotas = self.quotas.write();
        quotas.insert(
            resource.clone(),
            Quota {
                resource: resource.clone(),
                limit,
                used: 0,
                remaining: limit,
            },
        );
    }

    pub fn check_quota(&self, request: &QuotaRequest) -> Result<(), QuotaError> {
        let quotas = self.quotas.read();

        let quota = quotas
            .get(&request.resource)
            .ok_or_else(|| QuotaError::ResourceNotFound(request.resource.clone()))?;

        if quota.remaining >= request.amount {
            Ok(())
        } else {
            Err(QuotaError::QuotaExceeded {
                resource: request.resource.clone(),
                limit: quota.limit,
                requested: request.amount,
            })
        }
    }

    pub fn consume_quota(&self, request: &QuotaRequest) -> Result<(), QuotaError> {
        let mut quotas = self.quotas.write();

        let quota = quotas
            .get_mut(&request.resource)
            .ok_or_else(|| QuotaError::ResourceNotFound(request.resource.clone()))?;

        if quota.remaining >= request.amount {
            quota.used += request.amount;
            quota.remaining -= request.amount;
            Ok(())
        } else {
            Err(QuotaError::QuotaExceeded {
                resource: request.resource.clone(),
                limit: quota.limit,
                requested: request.amount,
            })
        }
    }

    pub fn release_quota(&self, resource: &str, amount: u64) {
        let mut quotas = self.quotas.write();

        if let Some(quota) = quotas.get_mut(resource) {
            quota.used = quota.used.saturating_sub(amount);
            quota.remaining = (quota.remaining + amount).min(quota.limit);
        }
    }

    pub fn get_quota(&self, resource: &str) -> Option<Quota> {
        let quotas = self.quotas.read();
        quotas.get(resource).cloned()
    }

    pub fn list_quotas(&self) -> Vec<Quota> {
        let quotas = self.quotas.read();
        quotas.values().cloned().collect()
    }

    pub fn reset_quota(&self, resource: &str) {
        let mut quotas = self.quotas.write();

        if let Some(quota) = quotas.get_mut(resource) {
            quota.used = 0;
            quota.remaining = quota.limit;
        }
    }
}
