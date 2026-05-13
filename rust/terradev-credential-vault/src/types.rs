use serde::{Deserialize, Serialize};
use zeroize::Zeroize;

#[derive(Debug, Clone, Serialize, Deserialize, Zeroize)]
#[zeroize(drop)]
pub struct Secret {
    pub name: String,
    pub value: Vec<u8>,
    pub created_at: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CredentialMetadata {
    pub name: String,
    pub provider: String,
    pub created_at: String,
    pub last_accessed: String,
}
