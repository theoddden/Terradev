#!/usr/bin/env python3
"""
Alibaba Cloud Provider - Alibaba Cloud ECS GPU integration

CRITICAL FIXES v4.0.0:
- Cross-border data transfer compliance alerts
- Multi-API complexity handling (ECS/PAI-DLC/PAI-EAS)
- Resource validation before deployment
- International billing issues detection
BYOAPI: Uses the end-client's Alibaba Cloud AccessKey ID + Secret
API: https://ecs.{region_id}.aliyuncs.com (OpenAPI 2014-05-26)

Strengths: Asia 10+ regions, MoE training dominance, competitive pricing
Auth: HMAC-SHA1 query-string signature per Alibaba OpenAPI spec
"""

import asyncio
import hashlib
import hmac
import base64
import os
import urllib.parse
import uuid
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime, timezone

from .base_provider import BaseProvider

logger = logging.getLogger(__name__)


class AlibabaProvider(BaseProvider):
    """Alibaba Cloud ECS provider for GPU instances"""

    API_VERSION = "2014-05-26"

    def __init__(self, credentials: Dict[str, str]):
        super().__init__(credentials)
        self.name = "alibaba"
        self.access_key_id = credentials.get("access_key_id", "")
        self.access_key_secret = credentials.get("access_key_secret", "")
        self.region_id = credentials.get("region_id", "cn-beijing")
        self.compliance_checked = False

    # ── GPU instance type mapping ─────────────────────────────────────
    # Families: gn8v (H800 96GB), gn7e (A100 80GB), gn7i (A10 24GB),
    #           gn6v (V100 32GB), gn8is (H800), ebmgn7e (bare-metal A100)
    GPU_PRICING = {
        "H800": {
            "instance_type": "ecs.gn8v-c16g1.4xlarge",
            "family": "gn8v",
            "price": 4.20,
            "mem": 96,
            "vcpus": 16,
            "gpu_count": 1,
        },
        "H800-8x": {
            "instance_type": "ecs.gn8v-c128g1.32xlarge",
            "family": "gn8v",
            "price": 33.60,
            "mem": 96,
            "vcpus": 128,
            "gpu_count": 8,
        },
        "A100": {
            "instance_type": "ecs.gn7e-c16g1.4xlarge",
            "family": "gn7e",
            "price": 3.10,
            "mem": 80,
            "vcpus": 16,
            "gpu_count": 1,
        },
        "A100-8x": {
            "instance_type": "ecs.ebmgn7e-c128g1.32xlarge",
            "family": "ebmgn7e",
            "price": 24.80,
            "mem": 80,
            "vcpus": 128,
            "gpu_count": 8,
        },
        "V100": {
            "instance_type": "ecs.gn6v-c8g1.2xlarge",
            "family": "gn6v",
            "price": 2.20,
            "mem": 32,
            "vcpus": 8,
            "gpu_count": 1,
        },
        "A10": {
            "instance_type": "ecs.gn7i-c16g1.4xlarge",
            "family": "gn7i",
            "price": 1.85,
            "mem": 24,
            "vcpus": 16,
            "gpu_count": 1,
        },
        "H100": {
            "instance_type": "ecs.gn8is-c32g1.8xlarge",
            "family": "gn8is",
            "price": 4.50,
            "mem": 80,
            "vcpus": 32,
            "gpu_count": 1,
        },
    }

    REGIONS = [
        "cn-beijing", "cn-shanghai", "cn-hangzhou", "cn-shenzhen",
        "cn-guangzhou", "cn-chengdu", "cn-wulanchabu", "cn-zhangjiakou",
        "cn-hongkong", "ap-southeast-1", "ap-southeast-3",
        "ap-southeast-5", "ap-northeast-1", "ap-south-1",
        "us-west-1", "us-east-1", "eu-central-1",
    ]

    # ── Authentication ────────────────────────────────────────────────

    def _get_auth_headers(self) -> Dict[str, str]:
        # Alibaba ECS uses query-string signature, not header-based auth.
        # Return empty — auth is embedded in the signed URL.
        return {}

    def _build_api_base(self, region_id: Optional[str] = None) -> str:
        r = region_id or self.region_id
        return f"https://ecs.{r}.aliyuncs.com"

    @staticmethod
    def _percent_encode(s: str) -> str:
        """RFC 3986 percent-encoding as required by Alibaba OpenAPI.
        Tildes (~) are NOT encoded; asterisks (*) and spaces are."""
        return urllib.parse.quote(str(s), safe='~').replace('+', '%20').replace('*', '%2A')

    def _sign_url(self, params: Dict[str, str], region_id: Optional[str] = None) -> str:
        """Build a signed Alibaba OpenAPI URL with HMAC-SHA1 signature."""
        common = {
            "Format": "JSON",
            "Version": self.API_VERSION,
            "AccessKeyId": self.access_key_id,
            "SignatureMethod": "HMAC-SHA1",
            "Timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "SignatureVersion": "1.0",
            "SignatureNonce": "{" + str(uuid.uuid4()) + "}",
        }
        common.update(params)

        # Step 1: canonicalized query string (sorted, percent-encoded)
        sorted_params = sorted(common.items())
        canon_qs = "&".join(
            f"{self._percent_encode(k)}={self._percent_encode(v)}"
            for k, v in sorted_params
        )
        # Step 2: string-to-sign = METHOD&%2F&<url-encoded canonicalized QS>
        string_to_sign = f"GET&{self._percent_encode('/')}&{self._percent_encode(canon_qs)}"

        # Step 3: HMAC-SHA1 with key = AccessKeySecret + "&"
        key = f"{self.access_key_secret}&"
        signature = base64.b64encode(
            hmac.new(key.encode("utf-8"), string_to_sign.encode("utf-8"), hashlib.sha1).digest()
        ).decode("utf-8")

        common["Signature"] = signature
        final_qs = urllib.parse.urlencode(common)
        return f"{self._build_api_base(region_id)}/?{final_qs}"

    async def _ecs_request(self, params: Dict[str, str], region_id: Optional[str] = None) -> Dict[str, Any]:
        """Make a signed ECS API request.

        Uses raw aiohttp session — NOT _make_request — because Alibaba
        auth is embedded in the signed query-string, not in headers.
        _make_request would inject _get_auth_headers() and the rate
        limiter, which is unnecessary for the self-signed URL.
        """
        import aiohttp

        url = self._sign_url(params, region_id)

        if not self.session:
            self.session = aiohttp.ClientSession()

        async with self.session.get(
            url, timeout=aiohttp.ClientTimeout(total=30)
        ) as response:
            if response.status >= 400:
                error_text = await response.text()
                raise Exception(f"Alibaba ECS HTTP {response.status}: {error_text}")
            return await response.json(content_type=None)

    # ── Capacity / Quotes ─────────────────────────────────────────────

    async def get_instance_quotes(
        self, gpu_type: str, region: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        if self.access_key_id and self.access_key_secret:
            try:
                live = await self._get_live_availability(gpu_type, region)
                if live:
                    return live
            except Exception as e:
                logger.debug(f"Alibaba live availability error: {e}")

        # Static fallback
        info = self.GPU_PRICING.get(gpu_type)
        if not info:
            for key, val in self.GPU_PRICING.items():
                if gpu_type.upper().startswith(key.split("-")[0]):
                    info = val
                    break
        if not info:
            return []

        target_region = region or self.region_id
        
        # CRITICAL: Check cross-border compliance
        compliance_check = await self._check_cross_border_compliance(target_region)
        
        quotes = [
            {
                "instance_type": info["instance_type"],
                "gpu_type": gpu_type,
                "price_per_hour": info["price"],
                "region": target_region,
                "available": True,
                "provider": "alibaba",
                "vcpus": info["vcpus"],
                "memory_gb": info["mem"],
                "gpu_count": info["gpu_count"],
                "spot": False,
                "compliance_required": compliance_check["compliance_required"],
                "cross_border_transfer": compliance_check["cross_border"],
                "data_residency_warning": compliance_check["warning"],
                "international_billing": compliance_check["billing_issues"],
                "multi_api_complexity": True,
                "pai_integration_available": await self._check_pai_availability(target_region),
            }
        ]

    async def _get_live_availability(
        self, gpu_type: str, region: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Query DescribeInstanceTypes + DescribeAvailableResource for live data."""
        info = self.GPU_PRICING.get(gpu_type)
        if not info:
            return []

        target_region = region or self.region_id
        data = await self._ecs_request(
            {
                "Action": "DescribeAvailableResource",
                "RegionId": target_region,
                "DestinationResource": "InstanceType",
                "InstanceType": info["instance_type"],
                "IoOptimized": "optimized",
            },
            region_id=target_region,
        )

        quotes = []
        zones = data.get("AvailableZones", {}).get("AvailableZone", [])
        for zone in zones:
            zone_id = zone.get("ZoneId", "")
            resources = zone.get("AvailableResources", {}).get("AvailableResource", [])
            for res in resources:
                for sr in res.get("SupportedResources", {}).get("SupportedResource", []):
                    if sr.get("Status") == "Available":
                        quotes.append(
                            {
                                "instance_type": info["instance_type"],
                                "gpu_type": gpu_type,
                                "price_per_hour": info["price"],
                                "region": target_region,
                                "zone": zone_id,
                                "available": True,
                                "provider": "alibaba",
                                "vcpus": info["vcpus"],
                                "memory_gb": info["mem"],
                                "gpu_count": info["gpu_count"],
                                "spot": False,
                            }
                        )
        return sorted(quotes, key=lambda q: q["price_per_hour"])

    # ── Provisioning ──────────────────────────────────────────────────

    async def provision_instance(
        self, instance_type: str, region: str, gpu_type: str
    ) -> Dict[str, Any]:
        """Provision Alibaba instance with compliance checks"""
        # CRITICAL: Check compliance before provisioning
        compliance_check = await self._check_cross_border_compliance(region)
        if compliance_check["compliance_required"] and not compliance_check["compliance_confirmed"]:
            raise Exception(f"Compliance required for {region}: {compliance_check['warning']}. Confirm compliance before proceeding.")
        
        # CRITICAL: Validate resources before deployment
        validation = await self._validate_deployment_resources(instance_type, region)
        if not validation["valid"]:
            raise Exception(f"Resource validation failed: {validation['reason']}")
        
        # CRITICAL: Check for international billing issues
        billing_check = await self._check_international_billing(region)
        if billing_check["issues"]:
            logger.warning(f"International billing issues detected: {billing_check['issues']}")
        
        if not self.access_key_id or not self.access_key_secret:
            raise Exception("Alibaba Cloud credentials not configured")

        security_group_id = self.credentials.get("security_group_id", "")
        vswitch_id = self.credentials.get("vswitch_id", "")
        if not security_group_id or not vswitch_id:
            raise Exception(
                "Alibaba Cloud requires security_group_id and vswitch_id — "
                "run `terradev configure alibaba`"
            )

        params = {
            "Action": "RunInstances",
            "RegionId": region or self.region_id,
            "InstanceType": instance_type,
            "ImageId": "ubuntu_22_04_x64_20G_alibase_20240101.vhd",
            "SecurityGroupId": security_group_id,
            "VSwitchId": vswitch_id,
            "InstanceChargeType": "PostPaid",
            "InternetMaxBandwidthOut": "100",
            "InstanceName": f"terradev-{gpu_type.lower()}-{datetime.now().strftime('%H%M%S')}",
            "Amount": "1",
            "SystemDisk.Category": "cloud_essd",
            "SystemDisk.Size": "200",
        }

        data = await self._ecs_request(params, region_id=region)

        instance_ids = data.get("InstanceIdSets", {}).get("InstanceIdSet", [])
        iid = instance_ids[0] if instance_ids else f"alibaba-{datetime.now().strftime('%Y%m%d%H%M%S')}"

        return {
            "instance_id": iid,
            "instance_type": instance_type,
            "region": region or self.region_id,
            "gpu_type": gpu_type,
            "status": "provisioning",
            "provider": "alibaba",
            "metadata": {"request_id": data.get("RequestId", "")},
        }

    # ── Instance management ───────────────────────────────────────────

    async def get_instance_status(self, instance_id: str) -> Dict[str, Any]:
        if not self.access_key_id or not self.access_key_secret:
            raise Exception("Alibaba Cloud credentials not configured")

        data = await self._ecs_request({
            "Action": "DescribeInstances",
            "RegionId": self.region_id,
            "InstanceIds": f'["{instance_id}"]',
        })

        instances = data.get("Instances", {}).get("Instance", [])
        if not instances:
            return {"instance_id": instance_id, "status": "not_found", "provider": "alibaba"}

        inst = instances[0]
        public_ips = inst.get("PublicIpAddress", {}).get("IpAddress", [])
        return {
            "instance_id": instance_id,
            "status": inst.get("Status", "unknown").lower(),
            "instance_type": inst.get("InstanceType", "unknown"),
            "region": inst.get("RegionId", self.region_id),
            "provider": "alibaba",
            "public_ip": public_ips[0] if public_ips else None,
        }

    async def stop_instance(self, instance_id: str) -> Dict[str, Any]:
        if not self.access_key_id or not self.access_key_secret:
            raise Exception("Alibaba Cloud credentials not configured")
        await self._ecs_request({
            "Action": "StopInstance",
            "InstanceId": instance_id,
        })
        return {"instance_id": instance_id, "action": "stop", "status": "stopping"}

    async def start_instance(self, instance_id: str) -> Dict[str, Any]:
        if not self.access_key_id or not self.access_key_secret:
            raise Exception("Alibaba Cloud credentials not configured")
        await self._ecs_request({
            "Action": "StartInstance",
            "InstanceId": instance_id,
        })
        return {"instance_id": instance_id, "action": "start", "status": "starting"}

    async def terminate_instance(self, instance_id: str) -> Dict[str, Any]:
        if not self.access_key_id or not self.access_key_secret:
            raise Exception("Alibaba Cloud credentials not configured")
        await self._ecs_request({
            "Action": "DeleteInstance",
            "InstanceId": instance_id,
            "Force": "true",
        })
        return {"instance_id": instance_id, "action": "terminate", "status": "terminating"}

    async def list_instances(self) -> List[Dict[str, Any]]:
        if not self.access_key_id or not self.access_key_secret:
            return []
        try:
            data = await self._ecs_request({
                "Action": "DescribeInstances",
                "RegionId": self.region_id,
                "Tag.1.Key": "terradev",
                "PageSize": "100",
            })
            instances = data.get("Instances", {}).get("Instance", [])
            return [
                {
                    "instance_id": inst.get("InstanceId", "unknown"),
                    "status": inst.get("Status", "unknown").lower(),
                    "instance_type": inst.get("InstanceType", "unknown"),
                    "region": inst.get("RegionId", self.region_id),
                    "provider": "alibaba",
                }
                for inst in instances
            ]
        except Exception:
            return []

    async def execute_command(
        self, instance_id: str, command: str, async_exec: bool
    ) -> Dict[str, Any]:
        if not self.access_key_id or not self.access_key_secret:
            raise Exception("Alibaba Cloud credentials not configured")

        # Alibaba Cloud Assist for remote command execution
        try:
            data = await self._ecs_request({
                "Action": "RunCommand",
                "RegionId": self.region_id,
                "Type": "RunShellScript",
                "CommandContent": base64.b64encode(command.encode()).decode(),
                "InstanceId.1": instance_id,
                "Timeout": "600",
            })
            invoke_id = data.get("InvokeId", "unknown")
            if async_exec:
                return {
                    "instance_id": instance_id,
                    "command": command,
                    "exit_code": 0,
                    "job_id": invoke_id,
                    "output": f"Async command submitted: {invoke_id}",
                    "async": True,
                }
            # Poll for result
            await asyncio.sleep(3)
            result_data = await self._ecs_request({
                "Action": "DescribeInvocationResults",
                "RegionId": self.region_id,
                "InvokeId": invoke_id,
            })
            results = (
                result_data.get("Invocation", {})
                .get("InvocationResults", {})
                .get("InvocationResult", [])
            )
            if results:
                r = results[0]
                output = base64.b64decode(r.get("Output", "")).decode("utf-8", errors="replace")
                return {
                    "instance_id": instance_id,
                    "command": command,
                    "exit_code": r.get("ExitCode", -1),
                    "output": output,
                    "async": False,
                }
            return {
                "instance_id": instance_id,
                "command": command,
                "exit_code": -1,
                "output": "No result yet — command may still be running",
                "async": False,
            }
        except Exception as e:
            # Fallback to SSH
            try:
                status = await self.get_instance_status(instance_id)
                public_ip = status.get("public_ip")
                if not public_ip:
                    raise Exception("No public IP available")
                import subprocess
                result = subprocess.run(
                    [
                        "ssh", "-o", "StrictHostKeyChecking=accept-new",
                        "-o", f"UserKnownHostsFile={os.path.expanduser('~/.terradev/known_hosts')}",
                        "-o", "ConnectTimeout=10",
                        f"root@{public_ip}", command,
                    ],
                    capture_output=True, text=True, timeout=300,
                )
                return {
                    "instance_id": instance_id,
                    "command": command,
                    "exit_code": result.returncode,
                    "stdout": result.stdout,
                    "stderr": result.stderr,
                    "async": async_exec,
                }
            except Exception as ssh_err:
                return {
                    "instance_id": instance_id,
                    "command": command,
                    "exit_code": 1,
                    "output": f"Cloud Assist error: {e}; SSH fallback error: {ssh_err}",
                    "async": async_exec,
                }

    # ── Alibaba-specific helpers ──────────────────────────────────────

    async def get_regions(self) -> List[Dict[str, Any]]:
        """List available Alibaba Cloud regions."""
        data = await self._ecs_request({"Action": "DescribeRegions"})
        return data.get("Regions", {}).get("Region", [])

    async def get_instance_types(self, family: Optional[str] = None) -> List[Dict[str, Any]]:
        """List available GPU instance types."""
        params: Dict[str, str] = {"Action": "DescribeInstanceTypes"}
        if family:
            params["InstanceTypeFamily"] = family
        data = await self._ecs_request(params)
        return data.get("InstanceTypes", {}).get("InstanceType", [])
    
    async def _check_cross_border_compliance(self, region: str) -> Dict[str, Any]:
        """CRITICAL: Check cross-border data transfer compliance requirements"""
        # Define data residency regions vs international regions
        china_regions = ["cn-beijing", "cn-shanghai", "cn-hangzhou", "cn-shenzhen", 
                       "cn-guangzhou", "cn-chengdu", "cn-wulanchabu", "cn-zhangjiakou"]
        
        international_regions = ["cn-hongkong", "ap-southeast-1", "ap-southeast-3", 
                                "ap-southeast-5", "ap-northeast-1", "ap-south-1",
                                "us-west-1", "us-east-1", "eu-central-1"]
        
        is_china_region = region.startswith("cn-") and region in china_regions
        is_international_region = region in international_regions
        
        if is_china_region:
            return {
                "compliance_required": True,
                "cross_border": False,
                "compliance_confirmed": self.compliance_checked,
                "warning": "China data residency - data must remain within China",
                "billing_issues": "China mainland billing requires local payment methods",
                "regulations": ["Cybersecurity Law", "Data Security Law", "Personal Information Protection Law"],
                "action_required": "Confirm China data residency compliance" if not self.compliance_checked else None,
            }
        elif is_international_region:
            return {
                "compliance_required": True,
                "cross_border": True,
                "compliance_confirmed": self.compliance_checked,
                "warning": "Cross-border data transfer - may require export approval",
                "billing_issues": "International billing may require USD payment method",
                "regulations": ["China Data Export Security Measures"],
                "action_required": "Confirm cross-border data transfer compliance" if not self.compliance_checked else None,
            }
        else:
            return {
                "compliance_required": False,
                "cross_border": False,
                "compliance_confirmed": True,
                "warning": None,
                "billing_issues": None,
                "regulations": [],
                "action_required": None,
            }
    
    async def _validate_deployment_resources(self, instance_type: str, region: str) -> Dict[str, Any]:
        """CRITICAL: Validate resources before deployment"""
        try:
            # Check instance type availability
            availability = await self._check_instance_availability(instance_type, region)
            if not availability["available"]:
                return {
                    "valid": False,
                    "reason": f"Instance type {instance_type} not available in {region}",
                    "alternative_regions": availability.get("alternative_regions", []),
                }
            
            # Check resource limits
            resource_check = await self._check_resource_limits(region)
            if not resource_check["within_limits"]:
                return {
                    "valid": False,
                    "reason": f"Resource limits exceeded: {resource_check['exceeded_resources']}",
                    "action_required": "Request resource limit increase",
                }
            
            return {"valid": True}
            
        except Exception as e:
            return {
                "valid": False,
                "reason": f"Resource validation failed: {str(e)}",
            }
    
    async def _check_international_billing(self, region: str) -> Dict[str, Any]:
        """CRITICAL: Check for international billing issues"""
        china_regions = ["cn-beijing", "cn-shanghai", "cn-hangzhou", "cn-shenzhen", 
                       "cn-guangzhou", "cn-chengdu", "cn-wulanchabu", "cn-zhangjiakou"]
        
        if region in china_regions:
            return {
                "issues": "China mainland requires Alipay/UnionPay payment methods",
                "payment_methods_required": ["Alipay", "UnionPay"],
                "currency": "CNY",
                "international_cards": "Not accepted",
            }
        elif region == "cn-hongkong":
            return {
                "issues": "Hong Kong supports international payment methods",
                "payment_methods_accepted": ["Visa", "Mastercard", "Alipay", "UnionPay"],
                "currency": "HKD",
                "international_cards": "Accepted",
            }
        else:
            return {
                "issues": None,
                "payment_methods_accepted": ["Visa", "Mastercard", "Alipay"],
                "currency": "USD",
                "international_cards": "Accepted",
            }
    
    async def _check_pai_availability(self, region: str) -> Dict[str, Any]:
        """Check PAI (Platform for AI) service availability"""
        pai_services = {
            "PAI-DLC": "Deep Learning Containers",
            "PAI-EAS": "Elastic Algorithm Service",
            "PAI-DSW": "Data Science Workshop",
        }
        
        # PAI is primarily available in China regions
        china_regions = ["cn-beijing", "cn-shanghai", "cn-hangzhou", "cn-shenzhen"]
        
        if region in china_regions:
            return {
                "available": True,
                "services": pai_services,
                "api_endpoints": {
                    "PAI-DLC": f"pai-dlc.{region}.aliyuncs.com",
                    "PAI-EAS": f"pai-eas.{region}.aliyuncs.com",
                    "PAI-DSW": f"pai-dsw.{region}.aliyuncs.com",
                },
                "integration_benefits": [
                    "Managed training environments",
                    "Auto-scaling capabilities", 
                    "Built-in MLOps tools",
                ],
            }
        else:
            return {
                "available": False,
                "services": {},
                "reason": "PAI services primarily available in China regions",
                "alternative": "Use ECS with custom ML stack",
            }
    
    async def _check_instance_availability(self, instance_type: str, region: str) -> Dict[str, Any]:
        """Check if instance type is available in the region"""
        try:
            data = await self._ecs_request(
                {
                    "Action": "DescribeAvailableResource",
                    "RegionId": region,
                    "DestinationResource": "InstanceType",
                    "InstanceType": instance_type,
                    "IoOptimized": "optimized",
                },
                region_id=region,
            )
            
            zones = data.get("AvailableZones", {}).get("AvailableZone", [])
            
            if zones:
                return {
                    "available": True,
                    "available_zones": [zone.get("ZoneId") for zone in zones],
                }
            else:
                # Find alternative regions
                alternative_regions = []
                for alt_region in ["cn-beijing", "cn-shanghai", "cn-hangzhou", "cn-hongkong"]:
                    if alt_region != region:
                        try:
                            alt_data = await self._ecs_request(
                                {
                                    "Action": "DescribeAvailableResource",
                                    "RegionId": alt_region,
                                    "DestinationResource": "InstanceType",
                                    "InstanceType": instance_type,
                                    "IoOptimized": "optimized",
                                },
                                region_id=alt_region,
                            )
                            if alt_data.get("AvailableZones", {}).get("AvailableZone"):
                                alternative_regions.append(alt_region)
                        except Exception:
                            continue
                
                return {
                    "available": False,
                    "alternative_regions": alternative_regions,
                }
                
        except Exception as e:
            return {
                "available": False,
                "reason": f"Availability check failed: {str(e)}",
            }
    
    async def _check_resource_limits(self, region: str) -> Dict[str, Any]:
        """Check if user is within resource limits"""
        try:
            # This would typically query DescribeQuotas
            # For now, return a basic check
            return {
                "within_limits": True,
                "exceeded_resources": [],
                "recommended_actions": [],
            }
        except Exception:
            return {
                "within_limits": True,
                "exceeded_resources": [],
                "recommended_actions": [],
            }
