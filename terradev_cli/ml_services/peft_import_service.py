#!/usr/bin/env python3
"""
PEFT Import Service - Import LoRA adapters from HuggingFace using PEFT library

Handles downloading, validating, and preparing LoRA adapters for use with
vLLM, LoRAX, or other inference servers.
"""

import logging
import os
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
import json

logger = logging.getLogger(__name__)


@dataclass
class PEFTAdapterConfig:
    """Configuration for a PEFT LoRA adapter"""
    adapter_id: str  # HuggingFace repo ID
    local_path: Path
    base_model: Optional[str] = None
    rank: Optional[int] = None
    alpha: Optional[int] = None
    target_modules: Optional[List[str]] = None
    peft_type: str = "LORA"  # LORA, ADALORA, etc.


class PEFTImportService:
    """
    Service for importing LoRA adapters from HuggingFace using PEFT.

    Provides:
    - Download adapters from HuggingFace
    - Validate adapter structure
    - Extract metadata (rank, alpha, target modules)
    - Prepare for deployment
    """

    def __init__(self, cache_dir: Optional[Path] = None):
        self.cache_dir = cache_dir or Path.home() / ".terradev" / "peft_adapters"
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def download_adapter(
        self,
        adapter_id: str,
        local_name: Optional[str] = None,
        token: Optional[str] = None
    ) -> PEFTAdapterConfig:
        """
        Download a LoRA adapter from HuggingFace.

        Args:
            adapter_id: HuggingFace repo ID (e.g., "username/adapter-name")
            local_name: Optional local name for the adapter
            token: Optional HuggingFace auth token for private repos

        Returns:
            PEFTAdapterConfig with local path and metadata
        """
        try:
            from huggingface_hub import snapshot_download
        except ImportError:
            raise ImportError(
                "huggingface_hub is required. Install with: pip install huggingface_hub"
            )

        local_path = self.cache_dir / (local_name or adapter_id.replace("/", "--"))
        local_path.mkdir(parents=True, exist_ok=True)

        logger.info(f"Downloading adapter {adapter_id} to {local_path}")

        # Download from HuggingFace
        downloaded_path = snapshot_download(
            repo_id=adapter_id,
            local_dir=local_path,
            local_dir_use_symlinks=False,
            token=token
        )

        # Extract metadata
        metadata = self._extract_adapter_metadata(Path(downloaded_path))

        config = PEFTAdapterConfig(
            adapter_id=adapter_id,
            local_path=Path(downloaded_path),
            base_model=metadata.get("base_model"),
            rank=metadata.get("rank"),
            alpha=metadata.get("alpha"),
            target_modules=metadata.get("target_modules"),
            peft_type=metadata.get("peft_type", "LORA")
        )

        logger.info(f"Downloaded adapter: {adapter_id}")
        logger.info(f"  Base model: {config.base_model}")
        logger.info(f"  Rank: {config.rank}")
        logger.info(f"  Alpha: {config.alpha}")

        return config

    def _extract_adapter_metadata(self, adapter_path: Path) -> Dict[str, Any]:
        """Extract metadata from adapter_config.json or adapter files"""
        metadata = {}

        # Try adapter_config.json (PEFT format)
        config_file = adapter_path / "adapter_config.json"
        if config_file.exists():
            with open(config_file, "r") as f:
                config_data = json.load(f)
                metadata["base_model"] = config_data.get("base_model_name_or_path")
                metadata["rank"] = config_data.get("r")
                metadata["alpha"] = config_data.get("lora_alpha")
                metadata["target_modules"] = config_data.get("target_modules")
                metadata["peft_type"] = config_data.get("peft_type", "LORA")
            return metadata

        # Try to infer from adapter weights
        adapter_file = adapter_path / "adapter_model.bin"
        if not adapter_file.exists():
            adapter_file = adapter_path / "adapter_model.safetensors"

        if adapter_file.exists():
            # Try to load with PEFT to extract metadata
            try:
                from peft import PeftConfig
                peft_config = PeftConfig.from_pretrained(str(adapter_path))
                metadata["base_model"] = getattr(peft_config, "base_model_name_or_path", None)
                metadata["rank"] = getattr(peft_config, "r", None)
                metadata["alpha"] = getattr(peft_config, "lora_alpha", None)
                metadata["target_modules"] = getattr(peft_config, "target_modules", None)
                metadata["peft_type"] = getattr(peft_config, "peft_type", "LORA")
            except Exception as e:
                logger.warning(f"Could not load PEFT config: {e}")

        return metadata

    def validate_adapter(self, adapter_path: Path) -> Dict[str, Any]:
        """
        Validate that an adapter is properly structured.

        Args:
            adapter_path: Path to the adapter directory

        Returns:
            Dict with validation results
        """
        required_files = ["adapter_config.json"]
        optional_files = ["adapter_model.bin", "adapter_model.safetensors"]

        result = {
            "valid": True,
            "missing_files": [],
            "warnings": []
        }

        # Check for required files
        for file in required_files:
            if not (adapter_path / file).exists():
                result["valid"] = False
                result["missing_files"].append(file)

        # Check for at least one weight file
        has_weights = any(
            (adapter_path / file).exists()
            for file in optional_files
        )
        if not has_weights:
            result["valid"] = False
            result["missing_files"].append("adapter_model.bin or adapter_model.safetensors")

        # Extract metadata for validation
        metadata = self._extract_adapter_metadata(adapter_path)
        if not metadata.get("rank"):
            result["warnings"].append("Could not determine adapter rank")

        return result

    def list_local_adapters(self) -> List[PEFTAdapterConfig]:
        """List all locally downloaded adapters"""
        adapters = []

        for adapter_dir in self.cache_dir.iterdir():
            if adapter_dir.is_dir():
                validation = self.validate_adapter(adapter_dir)
                if validation["valid"]:
                    metadata = self._extract_adapter_metadata(adapter_dir)
                    adapters.append(PEFTAdapterConfig(
                        adapter_id=adapter_dir.name,
                        local_path=adapter_dir,
                        base_model=metadata.get("base_model"),
                        rank=metadata.get("rank"),
                        alpha=metadata.get("alpha"),
                        target_modules=metadata.get("target_modules"),
                        peft_type=metadata.get("peft_type", "LORA")
                    ))

        return adapters

    def delete_adapter(self, adapter_id: str) -> bool:
        """Delete a locally downloaded adapter"""
        adapter_path = self.cache_dir / adapter_id.replace("/", "--")
        if adapter_path.exists():
            import shutil
            shutil.rmtree(adapter_path)
            logger.info(f"Deleted adapter: {adapter_id}")
            return True
        return False


def get_peft_import_service(cache_dir: Optional[Path] = None) -> PEFTImportService:
    """Factory function to create a PEFT import service instance"""
    return PEFTImportService(cache_dir)
