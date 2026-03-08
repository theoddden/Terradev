#!/usr/bin/env python3
"""
Genius Data Compressor - Lambda Function
Revolutionary data compression and chunking for massive storage with zero egress
"""

import json
import gzip
import zipfile
import os
import hashlib
import boto3
import concurrent.futures
from io import BytesIO
from typing import Dict, List, Any, Tuple, Optional
import logging
from datetime import datetime

# Configure logging
logger = logging.getLogger()
logger.setLevel(logging.INFO)

class GeniusDataCompressor:
    """Revolutionary data compression and chunking system"""
    
    def __init__(self):
        self.compression_algorithms = {
            'text': 'gzip',
            'images': 'webp',
            'videos': 'h265',
            'zip': 'deflate',
            'iac': 'gzip',
            'models': 'gzip'
        }
        
        self.compression_levels = {
            'hot_data': 6,
            'warm_data': 9,
            'cold_data': 9
        }
        
        self.chunk_sizes = {
            'small_files': '1MB',
            'medium_files': '10MB',
            'large_files': '100MB',
            'huge_files': '1GB'
        }
        
        self.parallel_processing = {
            'max_workers': 10,
            'batch_size': 100,
            'memory_limit': '1GB'
        }
        
        # Initialize S3 clients for all regions
        self.s3_clients = {}
        regions = os.environ.get('AWS_REGIONS', '').split(',')
        for region in regions:
            self.s3_clients[region] = boto3.client('s3', region_name=region)
        
        logger.info("Genius Data Compressor initialized")
    
    def handler(self, event, context):
        """Main Lambda handler for data compression"""
        logger.info("Starting data compression process")
        
        try:
            # Parse event
            compression_request = self._parse_event(event)
            
            # Determine compression strategy
            strategy = self._determine_compression_strategy(compression_request)
            
            # Compress and chunk data
            compressed_chunks = self._compress_and_chunk(compression_request, strategy)
            
            # Distribute chunks across regions
            distribution_result = self._distribute_chunks(compressed_chunks, strategy)
            
            # Generate metadata
            metadata = self._generate_metadata(compression_request, strategy, distribution_result)
            
            # Return results
            return {
                'statusCode': 200,
                'body': json.dumps({
                    'compression_id': compression_request['compression_id'],
                    'original_size': compression_request['data_size'],
                    'compressed_size': sum(chunk['size'] for chunk in compressed_chunks),
                    'compression_ratio': self._calculate_compression_ratio(compression_request, compressed_chunks),
                    'chunks_count': len(compressed_chunks),
                    'distribution_result': distribution_result,
                    'metadata': metadata,
                    'timestamp': datetime.now().isoformat()
                })
            }
            
        except Exception as e:
            logger.error(f"Compression failed: {str(e)}")
            return {
                'statusCode': 500,
                'body': json.dumps({
                    'error': str(e),
                    'timestamp': datetime.now().isoformat()
                })
            }
    
    def _parse_event(self, event: Dict[str, Any]) -> Dict[str, Any]:
        """Parse compression request from event"""
        return {
            'compression_id': event.get('compression_id', f"comp_{datetime.now().timestamp()}"),
            'data_source': event.get('data_source', 's3'),
            'data_path': event['data_path'],
            'data_type': event.get('data_type', 'auto'),
            'data_size': event.get('data_size', 0),
            'data_tier': event.get('data_tier', 'warm_data'),
            'compression_level': int(event.get('compression_level', os.environ.get('COMPRESSION_LEVEL', '6'))),
            'chunk_size_mb': int(event.get('chunk_size_mb', os.environ.get('CHUNK_SIZE_MB', '100'))),
            'parallel_uploads': int(event.get('parallel_uploads', os.environ.get('PARALLEL_UPLOADS', '10')))
        }
    
    def _determine_compression_strategy(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Determine optimal compression strategy"""
        data_type = request['data_type']
        data_tier = request['data_tier']
        
        # Auto-detect data type if not specified
        if data_type == 'auto':
            data_type = self._detect_data_type(request['data_path'])
        
        # Get compression algorithm
        algorithm = self.compression_algorithms.get(data_type, 'gzip')
        
        # Get compression level
        compression_level = self.compression_levels.get(data_tier, 6)
        
        # Get chunk size
        chunk_size = self._get_chunk_size(request['data_size'])
        
        return {
            'algorithm': algorithm,
            'level': compression_level,
            'chunk_size': chunk_size,
            'data_type': data_type,
            'parallel_uploads': min(request['parallel_uploads'], self.parallel_processing['max_workers'])
        }
    
    def _detect_data_type(self, file_path: str) -> str:
        """Auto-detect data type from file path"""
        file_path = file_path.lower()
        
        if file_path.endswith(('.txt', '.csv', '.json', '.xml', '.yaml', '.yml', '.py', '.js', '.html', '.css')):
            return 'text'
        elif file_path.endswith(('.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.webp')):
            return 'images'
        elif file_path.endswith(('.mp4', '.avi', '.mov', '.mkv', '.webm', '.flv')):
            return 'videos'
        elif file_path.endswith(('.zip', '.rar', '.7z', '.tar', '.gz')):
            return 'zip'
        elif file_path.endswith(('.tf', '.tfstate', '.hcl', '.json')):
            return 'iac'
        elif file_path.endswith(('.pkl', '.model', '.h5', '.pt', '.pth')):
            return 'models'
        else:
            return 'text'  # Default to text compression
    
    def _get_chunk_size(self, data_size: int) -> int:
        """Get optimal chunk size based on data size"""
        if data_size < 10 * 1024 * 1024:  # < 10MB
            return 1 * 1024 * 1024  # 1MB
        elif data_size < 100 * 1024 * 1024:  # < 100MB
            return 10 * 1024 * 1024  # 10MB
        elif data_size < 1024 * 1024 * 1024:  # < 1GB
            return 100 * 1024 * 1024  # 100MB
        else:
            return 1024 * 1024 * 1024  # 1GB
    
    def _compress_and_chunk(self, request: Dict[str, Any], strategy: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Compress and chunk data"""
        logger.info(f"Compressing data with strategy: {strategy}")
        
        # Download data
        data = self._download_data(request)
        
        # Compress data
        compressed_data = self._compress_data(data, strategy)
        
        # Chunk compressed data
        chunks = self._chunk_data(compressed_data, strategy['chunk_size'])
        
        # Add metadata to chunks
        chunks_with_metadata = []
        for i, chunk in enumerate(chunks):
            chunk_metadata = {
                'chunk_id': f"{request['compression_id']}_chunk_{i:04d}",
                'chunk_index': i,
                'total_chunks': len(chunks),
                'size': len(chunk),
                'checksum': hashlib.sha256(chunk).hexdigest(),
                'compression_algorithm': strategy['algorithm'],
                'compression_level': strategy['level'],
                'data_type': strategy['data_type'],
                'original_size': request['data_size'],
                'compression_ratio': len(chunk) / request['data_size'] if request['data_size'] > 0 else 0,
                'created_at': datetime.now().isoformat()
            }
            chunks_with_metadata.append({
                'data': chunk,
                'metadata': chunk_metadata
            })
        
        return chunks_with_metadata
    
    def _download_data(self, request: Dict[str, Any]) -> bytes:
        """Download data from source"""
        if request['data_source'] == 's3':
            # Parse S3 path
            s3_path = request['data_path']
            bucket_name = s3_path.split('/')[0]
            key = '/'.join(s3_path.split('/')[1:])
            
            # Get region for bucket
            region = self._get_bucket_region(bucket_name)
            
            # Download data
            response = self.s3_clients[region].get_object(Bucket=bucket_name, Key=key)
            return response['Body'].read()
        
        else:
            raise ValueError(f"Unsupported data source: {request['data_source']}")
    
    def _get_bucket_region(self, bucket_name: str) -> str:
        """Get bucket region"""
        try:
            # Try to get bucket location
            s3 = boto3.client('s3')
            response = s3.get_bucket_location(Bucket=bucket_name)
            region = response['LocationConstraint']
            return region if region else 'us-east-1'
        except Exception:
            return 'us-east-1'  # Default region
    
    def _compress_data(self, data: bytes, strategy: Dict[str, Any]) -> bytes:
        """Compress data using specified algorithm"""
        algorithm = strategy['algorithm']
        level = strategy['level']
        
        if algorithm == 'gzip':
            return gzip.compress(data, compresslevel=level)
        elif algorithm == 'deflate':
            return zlib.compress(data, level=level)
        else:
            # Default to gzip
            return gzip.compress(data, compresslevel=level)
    
    def _chunk_data(self, data: bytes, chunk_size: int) -> List[bytes]:
        """Chunk data into specified size"""
        chunks = []
        for i in range(0, len(data), chunk_size):
            chunks.append(data[i:i + chunk_size])
        return chunks
    
    def _distribute_chunks(self, chunks: List[Dict[str, Any]], strategy: Dict[str, Any]) -> Dict[str, Any]:
        """Distribute chunks across regions"""
        logger.info(f"Distributing {len(chunks)} chunks across regions")
        
        # Get available regions
        regions = list(self.s3_clients.keys())
        
        # Distribute chunks round-robin across regions
        distribution = {}
        parallel_uploads = strategy['parallel_uploads']
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=parallel_uploads) as executor:
            futures = []
            
            for i, chunk in enumerate(chunks):
                region = regions[i % len(regions)]
                chunk_id = chunk['metadata']['chunk_id']
                
                future = executor.submit(self._upload_chunk, chunk, region)
                futures.append((future, region, chunk_id))
            
            # Wait for all uploads to complete
            results = []
            for future, region, chunk_id in futures:
                try:
                    result = future.result()
                    results.append({
                        'chunk_id': chunk_id,
                        'region': region,
                        'bucket': result['bucket'],
                        'key': result['key'],
                        'etag': result['etag'],
                        'size': result['size']
                    })
                except Exception as e:
                    logger.error(f"Failed to upload chunk {chunk_id} to {region}: {e}")
                    results.append({
                        'chunk_id': chunk_id,
                        'region': region,
                        'error': str(e)
                    })
        
        return {
            'uploaded_chunks': results,
            'total_regions': len(regions),
            'parallel_uploads': parallel_uploads,
            'success_rate': len([r for r in results if 'error' not in r]) / len(results) * 100
        }
    
    def _upload_chunk(self, chunk: Dict[str, Any], region: str) -> Dict[str, Any]:
        """Upload chunk to specified region"""
        chunk_data = chunk['data']
        chunk_metadata = chunk['metadata']
        
        # Generate bucket name
        bucket_name = f"genius-data-storage-{region}-{chunk_metadata['compression_id'][:8]}"
        
        # Generate key
        key = f"compressed/{chunk_metadata['data_type']}/{chunk_metadata['chunk_id']}.bin"
        
        # Upload chunk
        response = self.s3_clients[region].put_object(
            Bucket=bucket_name,
            Key=key,
            Body=chunk_data,
            Metadata=chunk_metadata,
            ServerSideEncryption='AES256',
            StorageClass='STANDARD'
        )
        
        return {
            'bucket': bucket_name,
            'key': key,
            'etag': response['ETag'],
            'size': len(chunk_data)
        }
    
    def _generate_metadata(self, request: Dict[str, Any], strategy: Dict[str, Any], 
                          distribution_result: Dict[str, Any]) -> Dict[str, Any]:
        """Generate comprehensive metadata"""
        return {
            'compression_id': request['compression_id'],
            'original_source': request['data_path'],
            'original_size': request['data_size'],
            'compression_strategy': strategy,
            'distribution_result': distribution_result,
            'created_at': datetime.now().isoformat(),
            'compression_stats': {
                'algorithm': strategy['algorithm'],
                'level': strategy['level'],
                'chunk_size': strategy['chunk_size'],
                'parallel_uploads': strategy['parallel_uploads']
            },
            'storage_stats': {
                'total_chunks': len(distribution_result['uploaded_chunks']),
                'regions_used': distribution_result['total_regions'],
                'success_rate': distribution_result['success_rate']
            }
        }
    
    def _calculate_compression_ratio(self, request: Dict[str, Any], chunks: List[Dict[str, Any]]) -> float:
        """Calculate compression ratio"""
        original_size = request['data_size']
        compressed_size = sum(chunk['metadata']['size'] for chunk in chunks)
        
        if original_size == 0:
            return 0.0
        
        return compressed_size / original_size

# Lambda handler
def handler(event, context):
    """Lambda handler function"""
    compressor = GeniusDataCompressor()
    return compressor.handler(event, context)
