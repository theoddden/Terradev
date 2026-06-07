#!/usr/bin/env python3
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Copyright 2026 Terradev
#
"""
Terradev Telemetry Stub (Open Source - Apache 2.0)
This module provides a no-op telemetry client for open source compatibility.
All telemetry functionality has been removed for open source compliance.
"""

import logging
from typing import Dict, Any

logger = logging.getLogger(__name__)


class TelemetryClient:
    """No-op telemetry client for open source compatibility"""

    def __init__(self):
        logger.debug("TelemetryClient initialized (no-op for open source)")

    def log_action(self, action: str, details: Dict[str, Any] = None):
        """No-op log action for open source compatibility"""
        pass

    def check_license(self, action: str = "provision") -> Dict[str, Any]:
        """No-op license check - always allowed for open source"""
        return {
            "allowed": True,
            "tier": "open-source",
            "limit": float("inf"),
            "usage": 0,
            "reason": "Open source - no restrictions",
        }


# Global telemetry instance
_telemetry = None


def get_mandatory_telemetry() -> TelemetryClient:
    """Get global telemetry instance (no-op for open source)"""
    global _telemetry
    if _telemetry is None:
        _telemetry = TelemetryClient()
    return _telemetry


# Alias for backward compatibility
MandatoryTelemetryClient = TelemetryClient
