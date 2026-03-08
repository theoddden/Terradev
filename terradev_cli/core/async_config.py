#!/usr/bin/env python3
"""
Async Configuration Manager - Non-blocking configuration I/O
Replaces synchronous file operations with async alternatives
"""

import asyncio
import aiofiles
import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime
import fcntl
import os

logger = logging.getLogger(__name__)


class AsyncConfigManager:
    """
    Asynchronous configuration manager with file locking and caching.
    
    Features:
    - Non-blocking file I/O
    - File locking for concurrent access
    - In-memory caching with TTL
    - Atomic writes with temporary files
    - Backup and recovery
    """
    
    def __init__(self, config_dir: Optional[Path] = None):
        self.config_dir = config_dir or Path.home() / '.terradev'
        self.config_dir.mkdir(exist_ok=True)
        
        # In-memory cache with TTL
        self._cache: Dict[str, Dict[str, Any]] = {}
        self._cache_timestamps: Dict[str, datetime] = {}
        self._cache_ttl = 300  # 5 minutes
        
        # File locking
        self._locks: Dict[str, asyncio.Lock] = {}
    
    def _get_lock(self, filename: str) -> asyncio.Lock:
        """Get or create a lock for a specific file"""
        if filename not in self._locks:
            self._locks[filename] = asyncio.Lock()
        return self._locks[filename]
    
    def _is_cache_valid(self, filename: str) -> bool:
        """Check if cached data is still valid"""
        if filename not in self._cache:
            return False
        
        if filename not in self._cache_timestamps:
            return False
        
        age = datetime.now() - self._cache_timestamps[filename]
        return age.total_seconds() < self._cache_ttl
    
    async def load_json(self, filename: str, use_cache: bool = True) -> Dict[str, Any]:
        """
        Load JSON configuration file asynchronously
        
        Args:
            filename: Name of the config file
            use_cache: Whether to use in-memory cache
            
        Returns:
            Parsed JSON data or empty dict if file doesn't exist
        """
        file_path = self.config_dir / filename
        lock = self._get_lock(filename)
        
        # Check cache first
        if use_cache and self._is_cache_valid(filename):
            logger.debug(f"Cache hit for config file: {filename}")
            return self._cache[filename].copy()
        
        async with lock:
            try:
                if not file_path.exists():
                    logger.debug(f"Config file not found, returning empty dict: {filename}")
                    return {}
                
                async with aiofiles.open(file_path, 'r', encoding='utf-8') as f:
                    content = await f.read()
                    data = json.loads(content) if content.strip() else {}
                
                # Update cache
                self._cache[filename] = data.copy()
                self._cache_timestamps[filename] = datetime.now()
                
                logger.debug(f"Loaded config file: {filename} ({len(data)} keys)")
                return data
                
            except json.JSONDecodeError as e:
                logger.error(f"Invalid JSON in config file {filename}: {e}")
                return {}
            except Exception as e:
                logger.error(f"Error loading config file {filename}: {e}")
                return {}
    
    async def save_json(self, filename: str, data: Dict[str, Any], 
                       create_backup: bool = True) -> bool:
        """
        Save JSON configuration file asynchronously with atomic writes
        
        Args:
            filename: Name of the config file
            data: Data to save
            create_backup: Whether to create backup before overwriting
            
        Returns:
            True if successful, False otherwise
        """
        file_path = self.config_dir / filename
        lock = self._get_lock(filename)
        
        async with lock:
            try:
                # Create backup if requested and file exists
                if create_backup and file_path.exists():
                    backup_path = file_path.with_suffix(f'.backup.{int(datetime.now().timestamp())}')
                    async with aiofiles.open(file_path, 'r') as src:
                        async with aiofiles.open(backup_path, 'w') as dst:
                            await dst.write(await src.read())
                    logger.debug(f"Created backup: {backup_path}")
                
                # Write to temporary file first (atomic write)
                temp_path = file_path.with_suffix('.tmp')
                json_content = json.dumps(data, indent=2, sort_keys=True)
                
                async with aiofiles.open(temp_path, 'w', encoding='utf-8') as f:
                    await f.write(json_content)
                    await f.flush()  # Ensure data is written to disk
                
                # Atomic move
                temp_path.replace(file_path)
                
                # Update cache
                self._cache[filename] = data.copy()
                self._cache_timestamps[filename] = datetime.now()
                
                logger.debug(f"Saved config file: {filename} ({len(data)} keys)")
                return True
                
            except Exception as e:
                logger.error(f"Error saving config file {filename}: {e}")
                # Clean up temp file if it exists
                temp_path = file_path.with_suffix('.tmp')
                if temp_path.exists():
                    temp_path.unlink()
                return False
    
    async def update_json(self, filename: str, updates: Dict[str, Any], 
                         merge: bool = True) -> bool:
        """
        Update JSON configuration file with partial updates
        
        Args:
            filename: Name of the config file
            updates: Data to update/merge
            merge: Whether to merge with existing data or replace
            
        Returns:
            True if successful, False otherwise
        """
        if merge:
            current_data = await self.load_json(filename, use_cache=False)
            current_data.update(updates)
            return await self.save_json(filename, current_data)
        else:
            return await self.save_json(filename, updates)
    
    async def delete_key(self, filename: str, key: str) -> bool:
        """
        Delete a specific key from JSON configuration
        
        Args:
            filename: Name of the config file
            key: Key to delete
            
        Returns:
            True if successful, False otherwise
        """
        current_data = await self.load_json(filename, use_cache=False)
        if key in current_data:
            del current_data[key]
            return await self.save_json(filename, current_data)
        return True  # Key doesn't exist, consider it successful
    
    async def get_key(self, filename: str, key: str, default: Any = None) -> Any:
        """
        Get a specific key from JSON configuration
        
        Args:
            filename: Name of the config file
            key: Key to retrieve
            default: Default value if key doesn't exist
            
        Returns:
            Value or default
        """
        data = await self.load_json(filename)
        return data.get(key, default)
    
    async def set_key(self, filename: str, key: str, value: Any) -> bool:
        """
        Set a specific key in JSON configuration
        
        Args:
            filename: Name of the config file
            key: Key to set
            value: Value to set
            
        Returns:
            True if successful, False otherwise
        """
        return await self.update_json(filename, {key: value})
    
    def invalidate_cache(self, filename: Optional[str] = None):
        """
        Invalidate cache for specific file or all files
        
        Args:
            filename: Specific file to invalidate, or None for all
        """
        if filename:
            self._cache.pop(filename, None)
            self._cache_timestamps.pop(filename, None)
        else:
            self._cache.clear()
            self._cache_timestamps.clear()
    
    async def cleanup_cache(self):
        """Clean up expired cache entries"""
        now = datetime.now()
        expired_files = [
            filename for filename, timestamp in self._cache_timestamps.items()
            if (now - timestamp).total_seconds() > self._cache_ttl
        ]
        
        for filename in expired_files:
            self.invalidate_cache(filename)
        
        if expired_files:
            logger.debug(f"Cleaned up cache for {len(expired_files)} files")
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        return {
            'cached_files': len(self._cache),
            'cache_ttl_seconds': self._cache_ttl,
            'files': list(self._cache.keys()),
        }


# Global async config manager instance
_global_config_manager: Optional[AsyncConfigManager] = None


def get_async_config_manager() -> AsyncConfigManager:
    """Get the global async config manager instance"""
    global _global_config_manager
    if _global_config_manager is None:
        _global_config_manager = AsyncConfigManager()
    return _global_config_manager
