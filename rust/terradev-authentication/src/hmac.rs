use crate::encoding::{encode_parameters, percent_encode_rfc3986};
use crate::types::{AlibabaCredentials, OVHCredentials, SignatureResult};
use hmac::{Hmac, Mac};
use sha1::Sha1;
use std::time::{SystemTime, UNIX_EPOCH};

type HmacSha1 = Hmac<Sha1>;

pub struct AlibabaSigner;

impl AlibabaSigner {
    pub fn sign_request(
        credentials: &AlibabaCredentials,
        http_method: &str,
        url: &str,
        params: &[(String, String)],
    ) -> SignatureResult {
        let timestamp = Self::get_timestamp();
        
        // Build canonical query string
        let canonical_query = encode_parameters(params);
        
        // Build string to sign
        let string_to_sign = format!(
            "{}&{}&{}",
            http_method,
            percent_encode_rfc3986(url),
            percent_encode_rfc3986(&canonical_query)
        );
        
        // Create HMAC-SHA1 signature
        let mut mac = HmacSha1::new_from_slice(credentials.access_key_secret.as_bytes()).unwrap();
        mac.update(string_to_sign.as_bytes());
        let signature = hex::encode(mac.finalize().into_bytes());
        
        SignatureResult {
            signature,
            timestamp,
        }
    }
    
    fn get_timestamp() -> String {
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs();
        format!("{}", now)
    }
}

pub struct OVHSigner;

impl OVHSigner {
    pub fn sign_request(
        credentials: &OVHCredentials,
        http_method: &str,
        url: &str,
        body: &str,
        timestamp: &str,
    ) -> SignatureResult {
        // Build signature string
        let signature_string = format!(
            "{}+{}+{}+{}+{}+{}",
            credentials.application_secret,
            http_method,
            url,
            body,
            timestamp,
            credentials.consumer_key
        );
        
        // Create SHA1 signature
        let mut mac = HmacSha1::new_from_slice(credentials.application_secret.as_bytes()).unwrap();
        mac.update(signature_string.as_bytes());
        let signature = hex::encode(mac.finalize().into_bytes());
        
        SignatureResult {
            signature,
            timestamp: timestamp.to_string(),
        }
    }
}
