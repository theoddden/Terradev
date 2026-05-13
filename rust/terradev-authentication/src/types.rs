use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AlibabaCredentials {
    pub access_key_id: String,
    pub access_key_secret: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OVHCredentials {
    pub application_key: String,
    pub application_secret: String,
    pub consumer_key: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SignatureResult {
    pub signature: String,
    pub timestamp: String,
}
