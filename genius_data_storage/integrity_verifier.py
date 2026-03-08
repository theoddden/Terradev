#!/usr/bin/env python3
"""
Data Integrity Verifier - Lambda Function
Revolutionary data integrity verification across distributed storage
"""

import json
import hashlib
import boto3
import logging
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed
import time

# Configure logging
logger = logging.getLogger()
logger.setLevel(logging.INFO)

class DataIntegrityVerifier:
    """Revolutionary data integrity verification system"""
    
    def __init__(self):
        self.verification_algorithms = {
            'checksum': 'SHA256',
            'hash': 'MD5',
            'crc': 'CRC32'
        }
        
        self.verification_frequency = {
            'hot_data': 'daily',
            'warm_data': 'weekly',
            'cold_data': 'monthly'
        }
        
        self.repair_policy = {
            'auto_repair': True,
            'max_retries': 3,
            'quarantine_corrupted': True
        }
        
        self.monitoring = {
            'alerts_enabled': True,
            'metrics_retention': '30days',
            'dashboard_enabled': True
        }
        
        # Initialize S3 clients
        self.s3_clients = {}
        regions = os.environ.get('AWS_REGIONS', '').split(',')
        for region in regions:
            self.s3_clients[region] = boto3.client('s3', region_name=region)
        
        logger.info("Data Integrity Verifier initialized")
    
    def handler(self, event, context):
        """Main Lambda handler for integrity verification"""
        logger.info("Starting data integrity verification")
        
        try:
            # Parse verification request
            verification_request = self._parse_verification_request(event)
            
            # Determine verification strategy
            strategy = self._determine_verification_strategy(verification_request)
            
            # Execute verification
            verification_result = self._execute_verification(verification_request, strategy)
            
            # Generate verification report
            report = self._generate_verification_report(verification_request, strategy, verification_result)
            
            # Return results
            return {
                'statusCode': 200,
                'body': json.dumps({
                    'verification_id': verification_request['verification_id'],
                    'verification_result': verification_result,
                    'verification_report': report,
                    'timestamp': datetime.now().isoformat()
                })
            }
            
        except Exception as e:
            logger.error(f"Integrity verification failed: {str(e)}")
            return {
                'statusCode': 500,
                'body': json.dumps({
                    'error': str(e),
                    'timestamp': datetime.now().isoformat()
                })
            }
    
    def _parse_verification_request(self, event: Dict[str, Any]) -> Dict[str, Any]:
        """Parse verification request from event"""
        return {
            'verification_id': event.get('verification_id', f"verify_{datetime.now().timestamp()}"),
            'data_path': event['data_path'],
            'verification_type': event.get('verification_type', 'full'),
            'algorithm': event.get('algorithm', os.environ.get('CHECKSUM_ALGORITHM', 'SHA256')),
            'parallel_verification': event.get('parallel_verification', 'true').lower() == 'true',
            'max_retries': int(event.get('max_retries', os.environ.get('MAX_RETRIES', '3'))),
            'integrity_threshold': float(event.get('integrity_threshold', os.environ.get('INTEGRITY_THRESHOLD', '0.99'))),
            'auto_repair': event.get('auto_repair', 'true').lower() == 'true',
            'quarantine_corrupted': event.get('quarantine_corrupted', 'true').lower() == 'true'
        }
    
    def _determine_verification_strategy(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Determine optimal verification strategy"""
        logger.info(f"Determining verification strategy for {request['data_path']}")
        
        # Parse data path to determine location
        location = self._parse_data_path(request['data_path'])
        
        # Determine verification scope
        verification_scope = self._determine_verification_scope(request['verification_type'])
        
        # Determine verification frequency
        verification_frequency = self._determine_verification_frequency(request['data_path'])
        
        # Determine repair strategy
        repair_strategy = self._determine_repair_strategy(request)
        
        return {
            'location': location,
            'verification_scope': verification_scope,
            'verification_frequency': verification_frequency,
            'repair_strategy': repair_strategy,
            'algorithm': request['algorithm'],
            'parallel_verification': request['parallel_verification'],
            'max_retries': request['max_retries'],
            'integrity_threshold': request['integrity_threshold']
        }
    
    def _parse_data_path(self, data_path: str) -> Dict[str, Any]:
        """Parse data path to extract location information"""
        if data_path.startswith('s3://'):
            parts = data_path[5:].split('/', 1)
            return {
                'provider': 'aws',
                'bucket': parts[0],
                'key': parts[1] if len(parts) > 1 else '',
                'region': self._get_bucket_region(parts[0])
            }
        else:
            # Handle other providers
            return {
                'provider': 'unknown',
                'bucket': '',
                'key': data_path,
                'region': 'us-east-1'
            }
    
    def _determine_verification_scope(self, verification_type: str) -> str:
        """Determine verification scope"""
        if verification_type == 'full':
            return 'full_verification'
        elif verification_type == 'sample':
            return 'sample_verification'
        elif verification_type == 'metadata':
            return 'metadata_verification'
        else:
            return 'full_verification'
    
    def _determine_verification_frequency(self, data_path: str) -> str:
        """Determine verification frequency based on data tier"""
        # Simplified tier detection
        if 'hot' in data_path.lower():
            return self.verification_frequency['hot_data']
        elif 'warm' in data_path.lower():
            return self.verification_frequency['warm_data']
        elif 'cold' in data_path.lower():
            return self.verification_frequency['cold_data']
        else:
            return self.verification_frequency['warm_data']
    
    def _determine_repair_strategy(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Determine repair strategy"""
        return {
            'auto_repair': request['auto_repair'],
            'max_retries': request['max_retries'],
            'quarantine_corrupted': request['quarantine_corrupted'],
            'repair_sources': self._find_repair_sources(request['data_path'])
        }
    
    def _find_repair_sources(self, data_path: str) -> List[Dict[str, Any]]:
        """Find potential repair sources for corrupted data"""
        repair_sources = []
        
        # Find other regions with the same data
        regions = list(self.s3_clients.keys())
        for region in regions:
            try:
                client = self.s3_clients[region]
                # Check if data exists in this region
                location = self._parse_data_path(data_path)
                if location['provider'] == 'aws':
                    try:
                        client.head_object(Bucket=location['bucket'], Key=location['key'])
                        repair_sources.append({
                            'provider': 'aws',
                            'region': region,
                            'bucket': location['bucket'],
                            'key': location['key'],
                            'priority': 'high' if region == location['region'] else 'medium'
                        })
                    except:
                        pass
            except Exception as e:
                logger.debug(f"Failed to check repair source in {region}: {e}")
        
        return repair_sources
    
    def _get_bucket_region(self, bucket: str) -> str:
        """Get bucket region"""
        try:
            s3 = boto3.client('s3')
            response = s3.get_bucket_location(Bucket=bucket)
            region = response['LocationConstraint']
            return region if region else 'us-east-1'
        except Exception:
            return 'us-east-1'
    
    def _execute_verification(self, request: Dict[str, Any], strategy: Dict[str, Any]) -> Dict[str, Any]:
        """Execute integrity verification"""
        logger.info(f"Executing verification with strategy: {strategy}")
        
        # Get all chunks to verify
        chunks = self._get_chunks_to_verify(request['data_path'], strategy)
        
        # Verify chunks
        if strategy['parallel_verification']:
            verification_results = self._verify_chunks_parallel(chunks, strategy)
        else:
            verification_results = self._verify_chunks_sequential(chunks, strategy)
        
        # Calculate overall integrity
        overall_integrity = self._calculate_overall_integrity(verification_results)
        
        # Determine if repair is needed
        repair_needed = overall_integrity < strategy['integrity_threshold']
        
        # Execute repair if needed
        repair_result = None
        if repair_needed and strategy['repair_strategy']['auto_repair']:
            repair_result = self._execute_repair(request, strategy, verification_results)
        
        return {
            'verification_id': request['verification_id'],
            'chunks_verified': len(chunks),
            'verification_results': verification_results,
            'overall_integrity': overall_integrity,
            'repair_needed': repair_needed,
            'repair_result': repair_result,
            'verification_time': time.time()
        }
    
    def _get_chunks_to_verify(self, data_path: str, strategy: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Get all chunks to verify"""
        chunks = []
        
        # Parse data path
        location = self._parse_data_path(data_path)
        
        # List all objects in the bucket
        if location['provider'] == 'aws':
            client = self.s3_clients.get(location['region'])
            if client:
                try:
                    # List objects with prefix
                    prefix = location['key'].split('/')[0] if '/' in location['key'] else ''
                    response = client.list_objects_v2(
                        Bucket=location['bucket'],
                        Prefix=prefix
                    )
                    
                    for obj in response.get('Contents', []):
                        chunks.append({
                            'provider': 'aws',
                            'region': location['region'],
                            'bucket': location['bucket'],
                            'key': obj['Key'],
                            'size': obj['Size'],
                            'etag': obj['ETag'],
                            'last_modified': obj['LastModified']
                        })
                except Exception as e:
                    logger.error(f"Failed to list chunks: {e}")
        
        return chunks
    
    def _verify_chunks_parallel(self, chunks: List[Dict[str, Any]], strategy: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Verify chunks in parallel"""
        logger.info(f"Verifying {len(chunks)} chunks in parallel")
        
        verification_results = []
        
        with ThreadPoolExecutor(max_workers=10) as executor:
            # Submit verification tasks
            future_to_chunk = {
                executor.submit(self._verify_chunk, chunk, strategy): chunk
                for chunk in chunks
            }
            
            # Collect results
            for future in as_completed(future_to_chunk):
                chunk = future_to_chunk[future]
                try:
                    result = future.result()
                    verification_results.append(result)
                except Exception as e:
                    logger.error(f"Failed to verify chunk {chunk['key']}: {e}")
                    verification_results.append({
                        'chunk': chunk,
                        'success': False,
                        'error': str(e),
                        'checksum': None,
                        'verified_at': datetime.now().isoformat()
                    })
        
        return verification_results
    
    def _verify_chunks_sequential(self, chunks: List[Dict[str, Any]], strategy: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Verify chunks sequentially"""
        logger.info(f"Verifying {len(chunks)} chunks sequentially")
        
        verification_results = []
        
        for chunk in chunks:
            result = self._verify_chunk(chunk, strategy)
            verification_results.append(result)
        
        return verification_results
    
    def _verify_chunk(self, chunk: Dict[str, Any], strategy: Dict[str, Any]) -> Dict[str, Any]:
        """Verify individual chunk"""
        try:
            # Get chunk data
            data = self._get_chunk_data(chunk)
            
            # Calculate checksum
            checksum = self._calculate_checksum(data, strategy['algorithm'])
            
            # Compare with stored checksum
            stored_checksum = self._get_stored_checksum(chunk)
            
            # Determine if chunk is valid
            is_valid = checksum == stored_checksum
            
            return {
                'chunk': chunk,
                'success': True,
                'error': None,
                'checksum': checksum,
                'stored_checksum': stored_checksum,
                'is_valid': is_valid,
                'verified_at': datetime.now().isoformat()
            }
            
        except Exception as e:
            return {
                'chunk': chunk,
                'success': False,
                'error': str(e),
                'checksum': None,
                'stored_checksum': None,
                'is_valid': False,
                'verified_at': datetime.now().isoformat()
            }
    
    def _get_chunk_data(self, chunk: Dict[str, Any]) -> bytes:
        """Get chunk data from storage"""
        if chunk['provider'] == 'aws':
            client = self.s3_clients.get(chunk['region'])
            if client:
                response = client.get_object(
                    Bucket=chunk['bucket'],
                    Key=chunk['key']
                )
                return response['Body'].read()
        
        raise ValueError(f"Unsupported provider: {chunk['provider']}")
    
    def _calculate_checksum(self, data: bytes, algorithm: str) -> str:
        """Calculate checksum for data"""
        if algorithm == 'SHA256':
            return hashlib.sha256(data).hexdigest()
        elif algorithm == 'MD5':
            return hashlib.md5(data).hexdigest()
        elif algorithm == 'CRC32':
            return str(hashlib.crc32(data))
        else:
            return hashlib.sha256(data).hexdigest()
    
    def _get_stored_checksum(self, chunk: Dict[str, Any]) -> str:
        """Get stored checksum from metadata"""
        # For compressed chunks, checksum is stored in metadata
        if chunk['key'].endswith('.bin'):
            # Get object metadata
            if chunk['provider'] == 'aws':
                client = self.s3_clients.get(chunk['region'])
                if client:
                    response = client.head_object(
                        Bucket=chunk['bucket'],
                        Key=chunk['key']
                    )
                    metadata = response.get('Metadata', {})
                    return metadata.get('checksum', '')
        
        # For regular chunks, use ETag
        return chunk.get('etag', '').strip('"')
    
    def _calculate_overall_integrity(self, verification_results: List[Dict[str, Any]]) -> float:
        """Calculate overall integrity score"""
        if not verification_results:
            return 0.0
        
        valid_chunks = sum(1 for result in verification_results if result.get('is_valid', False))
        total_chunks = len(verification_results)
        
        return valid_chunks / total_chunks if total_chunks > 0 else 0.0
    
    def _execute_repair(self, request: Dict[str, Any], strategy: Dict[str, Any], 
                        verification_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Execute repair for corrupted chunks"""
        logger.info("Executing repair for corrupted chunks")
        
        # Find corrupted chunks
        corrupted_chunks = [
            result for result in verification_results
            if not result.get('is_valid', False)
        ]
        
        repair_results = []
        
        for corrupted_chunk in corrupted_chunks:
            chunk = corrupted_chunk['chunk']
            
            # Try to repair from other sources
            repair_result = self._repair_chunk(chunk, strategy)
            repair_results.append(repair_result)
            
            # Quarantine if repair fails
            if not repair_result['success'] and strategy['repair_strategy']['quarantine_corrupted']:
                self._quarantine_chunk(chunk, strategy)
        
        return {
            'corrupted_chunks': len(corrupted_chunks),
            'repair_results': repair_results,
            'successful_repairs': sum(1 for result in repair_results if result['success']),
            'quarantined_chunks': sum(1 for result in repair_results if not result['success'] and strategy['repair_strategy']['quarantine_corrupted'])
        }
    
    def _repair_chunk(self, chunk: Dict[str, Any], strategy: Dict[str, Any]) -> Dict[str, Any]:
        """Repair individual chunk"""
        repair_sources = strategy['repair_strategy']['repair_sources']
        
        for source in repair_sources:
            try:
                # Try to copy from repair source
                if self._copy_chunk_from_source(chunk, source):
                    return {
                        'chunk': chunk,
                        'success': True,
                        'source_used': source,
                        'repaired_at': datetime.now().isoformat()
                    }
            except Exception as e:
                logger.debug(f"Failed to repair from {source}: {e}")
                continue
        
        return {
            'chunk': chunk,
            'success': False,
            'error': 'No valid repair sources available',
            'repaired_at': datetime.now().isoformat()
        }
    
    def _copy_chunk_from_source(self, chunk: Dict[str, Any], source: Dict[str, Any]) -> bool:
        """Copy chunk from repair source"""
        if source['provider'] == 'aws':
            source_client = self.s3_clients.get(source['region'])
            target_client = self.s3_clients.get(chunk['region'])
            
            if source_client and target_client:
                # Copy object
                copy_source = {
                    'Bucket': source['bucket'],
                    'Key': source['key']
                }
                
                target_client.copy_object(
                    CopySource=copy_source,
                    Bucket=chunk['bucket'],
                    Key=chunk['key']
                )
                
                return True
        
        return False
    
    def _quarantine_chunk(self, chunk: Dict[str, Any], strategy: Dict[str, Any]):
        """Quarantine corrupted chunk"""
        try:
            if chunk['provider'] == 'aws':
                client = self.s3_clients.get(chunk['region'])
                if client:
                    # Move to quarantine bucket
                    quarantine_key = f"quarantine/{chunk['key']}"
                    
                    client.copy_object(
                        CopySource={
                            'Bucket': chunk['bucket'],
                            'Key': chunk['key']
                        },
                        Bucket=chunk['bucket'],
                        Key=quarantine_key,
                        Metadata={
                            'quarantine_reason': 'integrity_verification_failed',
                            'quarantine_date': datetime.now().isoformat(),
                            'original_key': chunk['key']
                        }
                    )
                    
                    # Delete original
                    client.delete_object(
                        Bucket=chunk['bucket'],
                        Key=chunk['key']
                    )
                    
                    logger.info(f"Quarantined chunk: {chunk['key']}")
        except Exception as e:
            logger.error(f"Failed to quarantine chunk {chunk['key']}: {e}")
    
    def _generate_verification_report(self, request: Dict[str, Any], strategy: Dict[str, Any], 
                                    verification_result: Dict[str, Any]) -> Dict[str, Any]:
        """Generate comprehensive verification report"""
        return {
            'verification_id': request['verification_id'],
            'data_path': request['data_path'],
            'verification_strategy': strategy,
            'verification_result': verification_result,
            'summary': {
                'total_chunks': verification_result['chunks_verified'],
                'valid_chunks': sum(1 for result in verification_result['verification_results'] if result.get('is_valid', False)),
                'corrupted_chunks': sum(1 for result in verification_result['verification_results'] if not result.get('is_valid', False)),
                'overall_integrity': verification_result['overall_integrity'],
                'repair_needed': verification_result['repair_needed']
            },
            'performance_stats': {
                'verification_time': verification_result['verification_time'],
                'parallel_verification': strategy['parallel_verification'],
                'algorithm_used': strategy['algorithm']
            },
            'repair_stats': verification_result.get('repair_result', {}),
            'recommendations': self._generate_recommendations(verification_result),
            'created_at': datetime.now().isoformat()
        }
    
    def _generate_recommendations(self, verification_result: Dict[str, Any]) -> List[str]:
        """Generate recommendations based on verification results"""
        recommendations = []
        
        if verification_result['overall_integrity'] < 0.95:
            recommendations.append("Consider increasing verification frequency")
        
        if verification_result['repair_needed']:
            recommendations.append("Review repair sources and data replication strategy")
        
        if verification_result['overall_integrity'] < 0.90:
            recommendations.append("Investigate root cause of data corruption")
        
        if not recommendations:
            recommendations.append("Data integrity is healthy")
        
        return recommendations

# Lambda handler
def handler(event, context):
    """Lambda handler function"""
    verifier = DataIntegrityVerifier()
    return verifier.handler(event, context)
