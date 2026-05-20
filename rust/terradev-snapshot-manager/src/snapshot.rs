use crate::types::{ModelState, SnapshotError};
use zstd::stream::{decode_all as read_all, encode_all};
use std::io::Cursor;

pub struct SnapshotManager {
    compression_level: i32,
}

impl SnapshotManager {
    pub fn new(compression_level: i32) -> Self {
        Self {
            compression_level: compression_level.clamp(1, 22),
        }
    }
    
    pub fn save_snapshot(&self, state: &ModelState) -> Result<Vec<u8>, SnapshotError> {
        // Serialize using bincode
        let serialized = bincode::serialize(state)
            .map_err(|e| SnapshotError::SerializationFailed(e.to_string()))?;
        
        // Compress using zstd
        let compressed = encode_all(Cursor::new(&serialized), self.compression_level)
            .map_err(|e| SnapshotError::CompressionFailed(e.to_string()))?;
        
        Ok(compressed)
    }
    
    pub fn load_snapshot(&self, data: &[u8]) -> Result<ModelState, SnapshotError> {
        // Decompress using zstd
        let decompressed = read_all(data)
            .map_err(|e| SnapshotError::CompressionFailed(e.to_string()))?;
        
        // Deserialize using bincode
        let state = bincode::deserialize(&decompressed)
            .map_err(|e| SnapshotError::SerializationFailed(e.to_string()))?;
        
        Ok(state)
    }
    
    pub fn save_snapshot_to_file(&self, state: &ModelState, path: &str) -> Result<(), SnapshotError> {
        let data = self.save_snapshot(state)?;
        std::fs::write(path, data)
            .map_err(|e| SnapshotError::IoError(e.to_string()))?;
        Ok(())
    }
    
    pub fn load_snapshot_from_file(&self, path: &str) -> Result<ModelState, SnapshotError> {
        let data = std::fs::read(path)
            .map_err(|e| SnapshotError::IoError(e.to_string()))?;
        self.load_snapshot(&data)
    }
    
    pub fn get_compression_ratio(&self, state: &ModelState) -> Result<f64, SnapshotError> {
        let serialized = bincode::serialize(state)
            .map_err(|e| SnapshotError::SerializationFailed(e.to_string()))?;
        let compressed = self.save_snapshot(state)?;
        
        Ok(compressed.len() as f64 / serialized.len() as f64)
    }
}
