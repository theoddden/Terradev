use crate::types::{CredentialMetadata, Secret};
use parking_lot::RwLock;
use std::collections::HashMap;
use std::sync::Arc;
use zeroize::Zeroize;

pub struct CredentialVault {
    secrets: Arc<RwLock<HashMap<String, Secret>>>,
}

impl CredentialVault {
    pub fn new() -> Self {
        Self {
            secrets: Arc::new(RwLock::new(HashMap::new())),
        }
    }
    
    pub fn store(&self, name: String, value: Vec<u8>, provider: String) {
        let secret = Secret {
            name: name.clone(),
            value,
            created_at: chrono::Utc::now().to_rfc3339(),
        };
        
        let mut secrets = self.secrets.write();
        secrets.insert(name, secret);
    }
    
    pub fn retrieve(&self, name: &str) -> Option<Vec<u8>> {
        let secrets = self.secrets.read();
        secrets.get(name).map(|secret| secret.value.clone())
    }
    
    pub fn get_metadata(&self, name: &str) -> Option<CredentialMetadata> {
        let secrets = self.secrets.read();
        secrets.get(name).map(|secret| CredentialMetadata {
            name: secret.name.clone(),
            provider: "unknown".to_string(),
            created_at: secret.created_at.clone(),
            last_accessed: chrono::Utc::now().to_rfc3339(),
        })
    }
    
    pub fn delete(&self, name: &str) -> bool {
        let mut secrets = self.secrets.write();
        secrets.remove(name).is_some()
    }
    
    pub fn list(&self) -> Vec<String> {
        let secrets = self.secrets.read();
        secrets.keys().cloned().collect()
    }
    
    pub fn clear(&self) {
        let mut secrets = self.secrets.write();
        secrets.clear();
    }
}

impl Drop for CredentialVault {
    fn drop(&mut self) {
        self.clear();
    }
}
