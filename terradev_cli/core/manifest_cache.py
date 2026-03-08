#!/usr/bin/env python3
"""
Manifest Cache + Idempotent Re-provision
CLI-native drift detection and rollback system (20 LOC implementation)
"""

import json
import hashlib
import asyncio
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict

@dataclass
class ManifestNode:
    """Single node in manifest"""
    provider: str
    pod_id: str
    instance_id: str
    gpus: int
    gpu_type: str
    region: str
    status: str
    created_at: str
    ttl: str

@dataclass
class Manifest:
    """Manifest file structure"""
    job: str
    version: str
    nodes: List[ManifestNode]
    dataset_hash: str
    ttl: str
    created_at: str
    metadata: Dict[str, Any]

class ManifestCache:
    """CLI-native manifest cache for drift detection and rollback"""
    
    def __init__(self, cache_dir: str = "./manifests"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
    
    def store_manifest(self, manifest: Manifest) -> str:
        """Store manifest to cache"""
        manifest_path = self.cache_dir / f"{manifest.job}.{manifest.version}.json"
        
        with open(manifest_path, 'w') as f:
            json.dump(asdict(manifest), f, indent=2)
        
        return str(manifest_path)
    
    def load_manifest(self, job: str, version: Optional[str] = None) -> Optional[Manifest]:
        """Load manifest from cache"""
        if version:
            manifest_path = self.cache_dir / f"{job}.{version}.json"
        else:
            # Load latest version
            manifests = list(self.cache_dir.glob(f"{job}.*.json"))
            if not manifests:
                return None
            manifest_path = max(manifests, key=lambda p: p.stat().st_mtime)
        
        if not manifest_path.exists():
            return None
        
        with open(manifest_path, 'r') as f:
            data = json.load(f)
        
        # Convert nodes back to ManifestNode objects
        nodes = [ManifestNode(**node) for node in data['nodes']]
        data['nodes'] = nodes
        
        return Manifest(**data)
    
    def list_versions(self, job: str) -> List[str]:
        """List all versions for a job"""
        manifests = list(self.cache_dir.glob(f"{job}.*.json"))
        versions = []
        
        for manifest_path in manifests:
            parts = manifest_path.stem.split('.')
            if len(parts) >= 2:
                versions.append(parts[1])
        
        return sorted(versions, reverse=True)
    
    def delete_manifest(self, job: str, version: str) -> bool:
        """Delete specific manifest version"""
        manifest_path = self.cache_dir / f"{job}.{version}.json"
        if manifest_path.exists():
            manifest_path.unlink()
            return True
        return False
    
    def compute_dataset_hash(self, dataset_path: str) -> str:
        """Compute SHA256 hash of dataset for drift detection"""
        hash_sha256 = hashlib.sha256()
        
        # For directories, hash file list + sizes
        if Path(dataset_path).is_dir():
            for file_path in sorted(Path(dataset_path).rglob('*')):
                if file_path.is_file():
                    hash_sha256.update(str(file_path.relative_to(dataset_path)).encode())
                    hash_sha256.update(str(file_path.stat().st_size).encode())
        else:
            # For single files, hash the file content
            with open(dataset_path, 'rb') as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    hash_sha256.update(chunk)
        
        return f"sha256:{hash_sha256.hexdigest()}"
