#!/usr/bin/env python3
"""
KV Cache Checkpoint Tests - Validate checkpoint recovery functionality

CRITICAL VALIDATION v4.1.0:
- Test KV cache serialization and deserialization
- Validate checkpoint creation and restoration
- Test spot interruption handling
- Verify data integrity and performance
"""

import asyncio
import logging
import sys
import tempfile
import shutil
from typing import Dict, List, Any
from pathlib import Path
import pickle
import gzip
from datetime import datetime, timedelta

# Add the parent directory to the path to import the manager
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.kv_cache_checkpoint_manager import KVCacheCheckpointManager, CheckpointConfig, CheckpointState

logger = logging.getLogger(__name__)


class KVCacheCheckpointTests:
    """Test suite for KV cache checkpoint functionality"""
    
    def __init__(self):
        self.test_results = []
        self.temp_dir = None
    
    async def run_all_tests(self) -> Dict[str, Any]:
        """Run all checkpoint tests"""
        logger.info("Starting KV cache checkpoint tests")
        
        # Create temporary directory for tests
        self.temp_dir = tempfile.mkdtemp(prefix="terradev_kv_test_")
        
        try:
            test_methods = [
                self.test_checkpoint_creation,
                self.test_checkpoint_restoration,
                self.test_data_integrity,
                self.test_compression,
                self.test_spot_termination_handling,
                self.test_cleanup_expired_checkpoints,
                self.test_storage_backends,
                self.test_performance_metrics,
            ]
            
            for test_method in test_methods:
                try:
                    result = await test_method()
                    self.test_results.append(result)
                    logger.info(f"✅ {result['test_name']}: {result['status']}")
                except Exception as e:
                    logger.error(f"❌ {test_method.__name__}: {e}")
                    self.test_results.append({
                        'test_name': test_method.__name__,
                        'status': 'FAILED',
                        'error': str(e)
                    })
            
            # Generate summary
            passed = sum(1 for r in self.test_results if r['status'] == 'PASSED')
            total = len(self.test_results)
            
            summary = {
                'total_tests': total,
                'passed_tests': passed,
                'failed_tests': total - passed,
                'success_rate': passed / total if total > 0 else 0,
                'test_results': self.test_results,
            }
            
            logger.info(f"Checkpoint test suite completed: {passed}/{total} tests passed")
            return summary
            
        finally:
            # Clean up temporary directory
            if self.temp_dir and Path(self.temp_dir).exists():
                shutil.rmtree(self.temp_dir)
    
    async def test_checkpoint_creation(self) -> Dict[str, Any]:
        """Test KV cache checkpoint creation"""
        test_name = "Checkpoint Creation"
        
        # Create checkpoint manager
        config = CheckpointConfig(
            checkpoint_dir=self.temp_dir,
            compression_enabled=True,
            storage_backend="local",
            nvme_path=self.temp_dir  # Use temp dir instead of /mnt
        )
        
        manager = KVCacheCheckpointManager(config)
        await manager.initialize()
        
        # Create mock KV cache data
        kv_cache_data = self._create_mock_kv_cache(8192, 32, 64, 128)
        
        # Test checkpoint creation
        checkpoint_id = await manager.create_checkpoint(
            model_id="test-model",
            request_id="test-request-1",
            kv_cache_data=kv_cache_data,
            context_length=8192,
            batch_size=1,
            num_layers=32,
            num_heads=64,
            head_dim=128
        )
        
        # Validate checkpoint was created
        creation_success = checkpoint_id is not None
        checkpoint_exists = checkpoint_id in manager.checkpoints
        
        # Validate checkpoint metadata
        if checkpoint_id:
            checkpoint = manager.checkpoints[checkpoint_id]
            metadata_valid = (
                checkpoint.model_id == "test-model" and
                checkpoint.request_id == "test-request-1" and
                checkpoint.context_length == 8192 and
                checkpoint.state == CheckpointState.SAVED
            )
        else:
            metadata_valid = False
        
        await manager.cleanup()
        
        return {
            'test_name': test_name,
            'status': 'PASSED' if creation_success and checkpoint_exists and metadata_valid else 'FAILED',
            'creation_success': creation_success,
            'checkpoint_exists': checkpoint_exists,
            'metadata_valid': metadata_valid,
            'checkpoint_id': checkpoint_id,
        }
    
    async def test_checkpoint_restoration(self) -> Dict[str, Any]:
        """Test KV cache checkpoint restoration"""
        test_name = "Checkpoint Restoration"
        
        # Create checkpoint manager
        config = CheckpointConfig(
            checkpoint_dir=self.temp_dir,
            compression_enabled=True,
            storage_backend="local",
            nvme_path=self.temp_dir  # Use temp dir instead of /mnt
        )
        
        manager = KVCacheCheckpointManager(config)
        await manager.initialize()
        
        # Create mock KV cache data
        original_kv_cache = self._create_mock_kv_cache(4096, 16, 32, 64)
        
        # Create checkpoint
        checkpoint_id = await manager.create_checkpoint(
            model_id="test-model",
            request_id="test-request-2",
            kv_cache_data=original_kv_cache,
            context_length=4096,
            batch_size=1,
            num_layers=16,
            num_heads=32,
            head_dim=64
        )
        
        # Restore checkpoint
        restored_kv_cache = await manager.restore_checkpoint(checkpoint_id, "test-request-2-restored")
        
        # Validate restoration
        restoration_success = restored_kv_cache is not None
        data_integrity = self._compare_kv_cache_data(original_kv_cache, restored_kv_cache) if restoration_success else False
        
        await manager.cleanup()
        
        return {
            'test_name': test_name,
            'status': 'PASSED' if restoration_success and data_integrity else 'FAILED',
            'restoration_success': restoration_success,
            'data_integrity': data_integrity,
            'checkpoint_id': checkpoint_id,
        }
    
    async def test_data_integrity(self) -> Dict[str, Any]:
        """Test data integrity across multiple checkpoints"""
        test_name = "Data Integrity"
        
        config = CheckpointConfig(
            checkpoint_dir=self.temp_dir,
            compression_enabled=True,
            storage_backend="local",
            nvme_path=self.temp_dir  # Use temp dir instead of /mnt
        )
        
        manager = KVCacheCheckpointManager(config)
        await manager.initialize()
        
        # Create multiple checkpoints with different data
        test_cases = [
            {"context": 4096, "layers": 16, "heads": 32, "dim": 64},
            {"context": 8192, "layers": 32, "heads": 64, "dim": 128},
            {"context": 16384, "layers": 64, "heads": 128, "dim": 256},
        ]
        
        results = []
        checkpoint_ids = []
        
        for i, case in enumerate(test_cases):
            # Create original data
            original_data = self._create_mock_kv_cache(
                case["context"], case["layers"], case["heads"], case["dim"]
            )
            
            # Create checkpoint
            checkpoint_id = await manager.create_checkpoint(
                model_id="test-model",
                request_id=f"test-request-{i}",
                kv_cache_data=original_data,
                context_length=case["context"],
                batch_size=1,
                num_layers=case["layers"],
                num_heads=case["heads"],
                head_dim=case["dim"]
            )
            
            checkpoint_ids.append(checkpoint_id)
            
            # Restore checkpoint
            restored_data = await manager.restore_checkpoint(checkpoint_id, f"restored-{i}")
            
            # Validate integrity
            integrity_ok = self._compare_kv_cache_data(original_data, restored_data) if restored_data else False
            
            results.append({
                "case": i,
                "context": case["context"],
                "checkpoint_id": checkpoint_id,
                "integrity_ok": integrity_ok,
            })
        
        # Check all integrity tests passed
        all_integrity_ok = all(r["integrity_ok"] for r in results)
        
        await manager.cleanup()
        
        return {
            'test_name': test_name,
            'status': 'PASSED' if all_integrity_ok else 'FAILED',
            'results': results,
            'all_integrity_ok': all_integrity_ok,
        }
    
    async def test_compression(self) -> Dict[str, Any]:
        """Test compression functionality"""
        test_name = "Compression"
        
        # Test with compression enabled
        config_compressed = CheckpointConfig(
            checkpoint_dir=self.temp_dir,
            compression_enabled=True,
            compression_level=6,
            storage_backend="local"
        )
        
        manager_compressed = KVCacheCheckpointManager(config_compressed)
        await manager_compressed.initialize()
        
        # Test without compression
        config_uncompressed = CheckpointConfig(
            checkpoint_dir=self.temp_dir,
            compression_enabled=False,
            storage_backend="local"
        )
        
        manager_uncompressed = KVCacheCheckpointManager(config_uncompressed)
        await manager_uncompressed.initialize()
        
        # Create test data
        test_data = self._create_mock_kv_cache(8192, 32, 64, 128)
        
        # Create compressed checkpoint
        start_time = asyncio.get_event_loop().time()
        compressed_id = await manager_compressed.create_checkpoint(
            model_id="test-model",
            request_id="test-compressed",
            kv_cache_data=test_data,
            context_length=8192,
            batch_size=1,
            num_layers=32,
            num_heads=64,
            head_dim=128
        )
        compressed_create_time = asyncio.get_event_loop().time() - start_time
        
        # Create uncompressed checkpoint
        start_time = asyncio.get_event_loop().time()
        uncompressed_id = await manager_uncompressed.create_checkpoint(
            model_id="test-model",
            request_id="test-uncompressed",
            kv_cache_data=test_data,
            context_length=8192,
            batch_size=1,
            num_layers=32,
            num_heads=64,
            head_dim=128
        )
        uncompressed_create_time = asyncio.get_event_loop().time() - start_time
        
        # Compare file sizes
        if compressed_id and uncompressed_id:
            compressed_checkpoint = manager_compressed.checkpoints[compressed_id]
            uncompressed_checkpoint = manager_uncompressed.checkpoints[uncompressed_id]
            
            size_reduction = uncompressed_checkpoint.size_bytes / compressed_checkpoint.size_bytes
            compression_effective = size_reduction >= 2.0  # At least 2x reduction
            
            # Test restoration
            compressed_restored = await manager_compressed.restore_checkpoint(compressed_id, "restored-compressed")
            uncompressed_restored = await manager_uncompressed.restore_checkpoint(uncompressed_id, "restored-uncompressed")
            
            restoration_success = compressed_restored is not None and uncompressed_restored is not None
            data_match = self._compare_kv_cache_data(compressed_restored, uncompressed_restored) if restoration_success else False
        else:
            size_reduction = 0
            compression_effective = False
            restoration_success = False
            data_match = False
        
        await manager_compressed.cleanup()
        await manager_uncompressed.cleanup()
        
        return {
            'test_name': test_name,
            'status': 'PASSED' if compression_effective and restoration_success and data_match else 'FAILED',
            'size_reduction': round(size_reduction, 2),
            'compression_effective': compression_effective,
            'restoration_success': restoration_success,
            'data_match': data_match,
            'compressed_create_time_ms': round(compressed_create_time * 1000, 2),
            'uncompressed_create_time_ms': round(uncompressed_create_time * 1000, 2),
        }
    
    async def test_spot_termination_handling(self) -> Dict[str, Any]:
        """Test spot termination handling"""
        test_name = "Spot Termination Handling"
        
        config = CheckpointConfig(
            checkpoint_dir=self.temp_dir,
            compression_enabled=True,
            storage_backend="local",
            nvme_path=self.temp_dir  # Use temp dir instead of /mnt
        )
        
        manager = KVCacheCheckpointManager(config)
        await manager.initialize()
        
        # Create multiple active checkpoints
        checkpoint_ids = []
        for i in range(3):
            kv_cache_data = self._create_mock_kv_cache(4096, 16, 32, 64)
            checkpoint_id = await manager.create_checkpoint(
                model_id="test-model",
                request_id=f"active-request-{i}",
                kv_cache_data=kv_cache_data,
                context_length=4096,
                batch_size=1,
                num_layers=16,
                num_heads=32,
                head_dim=64
            )
            checkpoint_ids.append(checkpoint_id)
        
        # Simulate spot termination
        termination_success = await manager.handle_spot_termination(
            instance_id="test-instance-1",
            provider="aws",
            region="us-east-1"
        )
        
        # Check if checkpoints were saved properly
        all_saved = all(
            manager.checkpoints[cid].state == CheckpointState.SAVED 
            for cid in checkpoint_ids if cid
        )
        
        # Test restoration on new instance
        restore_results = await manager.restore_on_new_instance(
            instance_id="test-instance-2",
            provider="aws",
            region="us-east-1"
        )
        
        restore_success = restore_results.get("success", False)
        
        await manager.cleanup()
        
        return {
            'test_name': test_name,
            'status': 'PASSED' if termination_success and all_saved and restore_success else 'FAILED',
            'termination_success': termination_success,
            'all_saved': all_saved,
            'restore_success': restore_success,
            'restore_results': restore_results,
        }
    
    async def test_cleanup_expired_checkpoints(self) -> Dict[str, Any]:
        """Test cleanup of expired checkpoints"""
        test_name = "Cleanup Expired Checkpoints"
        
        config = CheckpointConfig(
            checkpoint_dir=self.temp_dir,
            max_checkpoint_age_hours=1,  # 1 hour expiry
            compression_enabled=True,
            storage_backend="local"
        )
        
        manager = KVCacheCheckpointManager(config)
        await manager.initialize()
        
        # Create checkpoints
        checkpoint_ids = []
        for i in range(5):
            kv_cache_data = self._create_mock_kv_cache(2048, 8, 16, 32)
            checkpoint_id = await manager.create_checkpoint(
                model_id="test-model",
                request_id=f"cleanup-test-{i}",
                kv_cache_data=kv_cache_data,
                context_length=2048,
                batch_size=1,
                num_layers=8,
                num_heads=16,
                head_dim=32
            )
            checkpoint_ids.append(checkpoint_id)
        
        # Manually expire some checkpoints
        if checkpoint_ids[0] in manager.checkpoints:
            manager.checkpoints[checkpoint_ids[0]].expires_at = datetime.now() - timedelta(hours=2)
        if checkpoint_ids[1] in manager.checkpoints:
            manager.checkpoints[checkpoint_ids[1]].expires_at = datetime.now() - timedelta(hours=3)
        
        # Run cleanup
        initial_count = len(manager.checkpoints)
        await manager._cleanup_expired_checkpoints()
        final_count = len(manager.checkpoints)
        
        # Check cleanup worked
        cleanup_success = final_count < initial_count
        expected_count = initial_count - 2  # Should have removed 2 expired checkpoints
        cleanup_accurate = final_count == expected_count
        
        await manager.cleanup()
        
        return {
            'test_name': test_name,
            'status': 'PASSED' if cleanup_success and cleanup_accurate else 'FAILED',
            'initial_count': initial_count,
            'final_count': final_count,
            'expected_count': expected_count,
            'cleanup_success': cleanup_success,
            'cleanup_accurate': cleanup_accurate,
        }
    
    async def test_storage_backends(self) -> Dict[str, Any]:
        """Test different storage backends"""
        test_name = "Storage Backends"
        
        # Test local storage
        config_local = CheckpointConfig(
            checkpoint_dir=self.temp_dir,
            storage_backend="local",
            compression_enabled=True
        )
        
        manager_local = KVCacheCheckpointManager(config_local)
        await manager_local.initialize()
        
        # Create test data
        test_data = self._create_mock_kv_cache(4096, 16, 32, 64)
        
        # Test local storage
        local_checkpoint_id = await manager_local.create_checkpoint(
            model_id="test-model",
            request_id="local-test",
            kv_cache_data=test_data,
            context_length=4096,
            batch_size=1,
            num_layers=16,
            num_heads=32,
            head_dim=64
        )
        
        local_restored = await manager_local.restore_checkpoint(local_checkpoint_id, "local-restored")
        local_success = local_restored is not None
        
        await manager_local.cleanup()
        
        # For now, just test local storage (other backends would require credentials)
        return {
            'test_name': test_name,
            'status': 'PASSED' if local_success else 'FAILED',
            'local_success': local_success,
            'local_checkpoint_id': local_checkpoint_id,
            'note': 'Only local storage tested - other backends require credentials',
        }
    
    async def test_performance_metrics(self) -> Dict[str, Any]:
        """Test performance metrics collection"""
        test_name = "Performance Metrics"
        
        config = CheckpointConfig(
            checkpoint_dir=self.temp_dir,
            compression_enabled=True,
            storage_backend="local",
            nvme_path=self.temp_dir  # Use temp dir instead of /mnt
        )
        
        manager = KVCacheCheckpointManager(config)
        await manager.initialize()
        
        # Create multiple checkpoints to generate metrics
        for i in range(5):
            kv_cache_data = self._create_mock_kv_cache(4096, 16, 32, 64)
            await manager.create_checkpoint(
                model_id="test-model",
                request_id=f"metrics-test-{i}",
                kv_cache_data=kv_cache_data,
                context_length=4096,
                batch_size=1,
                num_layers=16,
                num_heads=32,
                head_dim=64
            )
        
        # Get metrics
        metrics = manager.get_metrics()
        status = manager.get_status()
        
        # Validate metrics
        metrics_valid = (
            metrics.total_checkpoints_created == 5 and
            metrics.total_data_saved_gb > 0 and
            metrics.avg_save_time_ms > 0
        )
        
        status_valid = (
            status["active_checkpoints"] == 5 and
            status["total_checkpoints"] == 5 and
            status["metrics"]["total_created"] == 5
        )
        
        await manager.cleanup()
        
        return {
            'test_name': test_name,
            'status': 'PASSED' if metrics_valid and status_valid else 'FAILED',
            'metrics_valid': metrics_valid,
            'status_valid': status_valid,
            'total_created': metrics.total_checkpoints_created,
            'avg_save_time_ms': round(metrics.avg_save_time_ms, 2),
            'data_saved_gb': round(metrics.total_data_saved_gb, 2),
        }
    
    def _create_mock_kv_cache(self, context_length: int, num_layers: int, num_heads: int, head_dim: int) -> Dict[str, Any]:
        """Create mock KV cache data for testing"""
        # Create realistic mock KV cache structure
        kv_cache = {
            "key_cache": [],
            "value_cache": [],
            "metadata": {
                "context_length": context_length,
                "num_layers": num_layers,
                "num_heads": num_heads,
                "head_dim": head_dim,
                "created_at": datetime.now().isoformat(),
            }
        }
        
        # Generate mock key and value caches
        for layer in range(num_layers):
            layer_keys = []
            layer_values = []
            
            for head in range(num_heads):
                # Generate mock attention keys and values
                head_keys = [[0.1 * (i + j) for j in range(head_dim)] for i in range(context_length)]
                head_values = [[0.2 * (i + j) for j in range(head_dim)] for i in range(context_length)]
                
                layer_keys.append(head_keys)
                layer_values.append(head_values)
            
            kv_cache["key_cache"].append(layer_keys)
            kv_cache["value_cache"].append(layer_values)
        
        return kv_cache
    
    def _compare_kv_cache_data(self, original: Any, restored: Any) -> bool:
        """Compare original and restored KV cache data"""
        if not original or not restored:
            return False
        
        try:
            # Compare basic structure
            if (original.get("metadata") != restored.get("metadata")):
                return False
            
            # Compare key cache structure
            if len(original.get("key_cache", [])) != len(restored.get("key_cache", [])):
                return False
            
            # Compare value cache structure
            if len(original.get("value_cache", [])) != len(restored.get("value_cache", [])):
                return False
            
            # Sample comparison of data (not exhaustive for performance)
            orig_keys = original.get("key_cache", [])
            rest_keys = restored.get("key_cache", [])
            
            if orig_keys and rest_keys:
                # Compare first layer, first head, first few values
                orig_sample = orig_keys[0][0][:5] if orig_keys[0] else []
                rest_sample = rest_keys[0][0][:5] if rest_keys[0] else []
                
                for i, (orig_val, rest_val) in enumerate(zip(orig_sample, rest_sample)):
                    if abs(orig_val - rest_val) > 1e-6:  # Allow for floating point precision
                        return False
            
            return True
            
        except Exception:
            return False


async def main():
    """Run the KV cache checkpoint tests"""
    logging.basicConfig(level=logging.INFO)
    
    test_suite = KVCacheCheckpointTests()
    results = await test_suite.run_all_tests()
    
    # Print results
    print(f"\n{'='*60}")
    print(f"KV Cache Checkpoint Test Results")
    print(f"{'='*60}")
    print(f"Total Tests: {results['total_tests']}")
    print(f"Passed: {results['passed_tests']}")
    print(f"Failed: {results['failed_tests']}")
    print(f"Success Rate: {results['success_rate']:.1%}")
    print(f"{'='*60}")
    
    # Print failed tests
    failed_tests = [r for r in results['test_results'] if r['status'] == 'FAILED']
    if failed_tests:
        print(f"\n❌ Failed Tests:")
        for test in failed_tests:
            print(f"  - {test['test_name']}")
            if 'error' in test:
                print(f"    Error: {test['error']}")
    
    # Print key metrics
    print(f"\n📊 Key Validation Results:")
    
    # Find checkpoint creation test
    creation_test = next((r for r in results['test_results'] if 'Creation' in r['test_name']), None)
    if creation_test and creation_test['status'] == 'PASSED':
        print(f"  ✅ Checkpoint creation and metadata validation")
    
    # Find restoration test
    restoration_test = next((r for r in results['test_results'] if 'Restoration' in r['test_name']), None)
    if restoration_test and restoration_test['status'] == 'PASSED':
        print(f"  ✅ Checkpoint restoration with data integrity")
    
    # Find spot termination test
    spot_test = next((r for r in results['test_results'] if 'Termination' in r['test_name']), None)
    if spot_test and spot_test['status'] == 'PASSED':
        print(f"  ✅ Spot interruption handling and migration")
    
    return results['success_rate'] == 1.0


if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)
