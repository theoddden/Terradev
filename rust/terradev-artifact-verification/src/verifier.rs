use crate::types::{Artifact, VerificationError, VerificationResult};
use sha2::{Digest, Sha256};
use std::io::Read;

pub struct ArtifactVerifier;

impl ArtifactVerifier {
    pub fn compute_sha256(data: &[u8]) -> String {
        let mut hasher = Sha256::new();
        hasher.update(data);
        hex::encode(hasher.finalize())
    }
    
    pub fn verify_artifact(
        data: &[u8],
        expected_checksum: &str,
        algorithm: &str,
    ) -> Result<VerificationResult, VerificationError> {
        let computed = match algorithm {
            "sha256" => Self::compute_sha256(data),
            _ => return Err(VerificationError::HashMismatch {
                expected: expected_checksum.to_string(),
                actual: "unsupported algorithm".to_string(),
            }),
        };
        
        if computed == expected_checksum {
            Ok(VerificationResult {
                is_valid: true,
                computed_checksum: computed,
                expected_checksum: expected_checksum.to_string(),
                algorithm: algorithm.to_string(),
            })
        } else {
            Ok(VerificationResult {
                is_valid: false,
                computed_checksum: computed,
                expected_checksum: expected_checksum.to_string(),
                algorithm: algorithm.to_string(),
            })
        }
    }
    
    pub fn verify_file(
        path: &str,
        expected_checksum: &str,
        algorithm: &str,
    ) -> Result<VerificationResult, VerificationError> {
        let mut file = std::fs::File::open(path)
            .map_err(|_| VerificationError::NotFound(path.to_string()))?;
        
        let mut data = Vec::new();
        file.read_to_end(&mut data)
            .map_err(|_| VerificationError::NotFound(path.to_string()))?;
        
        Self::verify_artifact(&data, expected_checksum, algorithm)
    }
}
