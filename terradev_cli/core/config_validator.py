#!/usr/bin/env python3
"""
Config Validator - Compile-time configuration schema validation

Rust implementation provides:
- JSON schema validation
- Type checking
- Required field enforcement
- Prevents deployment failures from invalid configurations
"""

import json
import logging
from typing import Dict

logger = logging.getLogger(__name__)

# Rust config validator integration
try:
    from terradev_config_validator import PyConfigValidator
    USE_RUST_VALIDATOR = True
    logger.info("Using Rust config validator for compile-time validation")
except ImportError:
    USE_RUST_VALIDATOR = False
    logger.info("Rust config validator not available, using Python fallback")


class ConfigValidator:
    """Config validator with Rust backend or Python fallback"""
    
    def __init__(self, schema_json: str):
        if USE_RUST_VALIDATOR:
            self._rust_validator = PyConfigValidator(schema_json=schema_json)
        else:
            self._schema = json.loads(schema_json)
    
    def validate(self, config_json: str) -> Dict:
        """Validate a configuration against the schema"""
        if USE_RUST_VALIDATOR:
            report = self._rust_validator.validate(config_json=config_json)
            return {
                "is_valid": report.is_valid,
                "errors": report.errors,
            }
        else:
            # Python fallback - basic validation
            config = json.loads(config_json)
            errors = []
            
            if "required" in self._schema:
                for field in self._schema["required"]:
                    if field not in config:
                        errors.append(f"Missing required field: {field}")
            
            if "properties" in self._schema:
                for field, schema in self._schema["properties"].items():
                    if field in config:
                        expected_type = schema.get("type")
                        if expected_type == "string" and not isinstance(config[field], str):
                            errors.append(f"Field {field} should be string")
                        elif expected_type == "number" and not isinstance(config[field], (int, float)):
                            errors.append(f"Field {field} should be number")
            
            return {
                "is_valid": len(errors) == 0,
                "errors": errors,
            }
