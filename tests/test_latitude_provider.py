#!/usr/bin/env python3
"""
Test suite for Latitude.sh Provider
"""

import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch
from datetime import datetime, timedelta

from terradev_cli.providers.latitude_provider import LatitudeProvider


class TestLatitudeProvider:
    """Test cases for Latitude.sh provider"""

    @pytest.fixture
    def provider(self):
        """Create provider instance for testing"""
        credentials = {"api_key": "test_api_key_12345"}
        return LatitudeProvider(credentials)

    @pytest.fixture
    def mock_session(self):
        """Mock aiohttp session"""
        session = AsyncMock()
        return session

    @pytest.mark.asyncio
    async def test_get_instance_quotes_bare_metal(self, provider, mock_session):
        """Test getting bare metal quotes"""
        # Mock API response
        mock_response = {
            "data": [
                {
                    "id": "plan_test123",
                    "attributes": {
                        "slug": "g3-h100-medium-43",
                        "name": "g3.h100.medium-43",
                        "specs": {
                            "cpu": {"cores": 6, "type": "E-2276G"},
                            "memory": {"total": 32},
                            "gpu": {"count": 4, "type": "NVIDIA H100"},
                            "drives": [{"count": 1, "size": "3.8TB", "type": "SSD"}],
                            "nics": [{"count": 1, "type": "10 Gbps"}]
                        },
                        "regions": [
                            {
                                "name": "Brazil",
                                "pricing": {
                                    "USD": {"hour": 10, "month": 50, "year": 100}
                                }
                            }
                        ]
                    }
                }
            ]
        }
        
        with patch.object(provider, '_make_request', return_value=mock_response):
            with patch.object(provider, '_get_virtual_machine_quotes', return_value=[]):
                quotes = await provider.get_instance_quotes("H100")
                
                assert len(quotes) > 0
                quote = quotes[0]
                assert quote["provider"] == "latitude"
                assert quote["gpu_type"] == "H100"
                assert quote["instance_category"] == "bare_metal"
                assert quote["isolation"] == "bare_metal"
                assert quote["ssh_access"] is True
                assert quote["ipmi_access"] is True
                assert quote["price_per_hour"] == 10

    @pytest.mark.asyncio
    async def test_provision_bare_metal_instance(self, provider, mock_session):
        """Test provisioning bare metal instance"""
        # Mock required API responses
        projects_response = {"data": [{"id": "proj_test123"}]}
        regions_response = {"data": [{"id": "ASH"}]}
        server_response = {
            "data": {
                "id": "sv_test123",
                "attributes": {
                    "hostname": "terradev-h100-20231201120000",
                    "status": "provisioning",
                    "primary_ipv4": "192.168.1.100",
                    "primary_ipv6": "2001:db8::1",
                    "specs": {
                        "cpu": "Xeon E-2276G CPU @ 3.80GHz (6 cores)",
                        "gpu": {"count": 4, "type": "NVIDIA H100"}
                    },
                    "plan": {"id": "plan_test", "slug": "g3-h100-medium-43"},
                    "interfaces": [
                        {"role": "ipmi", "name": "IPMI"},
                        {"role": "internal", "name": "PXE"}
                    ]
                }
            }
        }
        
        with patch.object(provider, '_make_request') as mock_request:
            mock_request.side_effect = [
                projects_response,
                regions_response, 
                server_response
            ]
            
            result = await provider.provision_instance(
                "latitude-bare-metal-g3-h100-medium-43",
                "Brazil", 
                "H100"
            )
            
            assert result["instance_id"] == "sv_test123"
            assert result["provider"] == "latitude"
            assert result["instance_category"] == "bare_metal"
            assert result["isolation"] == "bare_metal"
            assert result["ssh_access"] is True
            assert result["ipmi_access"] is True
            assert result["gpu_type"] == "H100"

    @pytest.mark.asyncio
    async def test_get_instance_status_bare_metal(self, provider):
        """Test getting bare metal instance status"""
        mock_response = {
            "data": {
                "id": "sv_test123",
                "attributes": {
                    "status": "on",
                    "hostname": "test-server",
                    "primary_ipv4": "192.168.1.100",
                    "ipmi_status": "Normal",
                    "specs": {
                        "cpu": "Xeon E-2276G CPU @ 3.80GHz (6 cores)",
                        "gpu": {"count": 4, "type": "NVIDIA H100"}
                    },
                    "locked": False
                }
            }
        }
        
        with patch.object(provider, '_make_request', return_value=mock_response):
            status = await provider.get_instance_status("sv_test123")
            
            assert status["instance_id"] == "sv_test123"
            assert status["status"] == "on"
            assert status["instance_category"] == "bare_metal"
            assert status["primary_ipv4"] == "192.168.1.100"
            assert status["ipmi_status"] == "Normal"

    @pytest.mark.asyncio
    async def test_list_instances(self, provider):
        """Test listing all instances"""
        mock_servers_response = {
            "data": [
                {
                    "id": "sv_test123",
                    "attributes": {
                        "status": "on",
                        "hostname": "bare-metal-server",
                        "primary_ipv4": "192.168.1.100",
                        "plan": {"slug": "g3-h100-medium-43"},
                        "specs": {"gpu": {"type": "NVIDIA H100"}},
                        "role": "Bare Metal"
                    }
                }
            ]
        }
        
        def mock_make_request(method, url, **kwargs):
            if "virtual-machines" in url:
                return {"data": []}  # No VMs
            else:
                return mock_servers_response  # Servers
        
        with patch.object(provider, '_make_request', side_effect=mock_make_request):
            instances = await provider.list_instances()
            
            assert len(instances) == 1
            instance = instances[0]
            assert instance["instance_id"] == "sv_test123"
            assert instance["instance_category"] == "bare_metal"
            assert instance["gpu_type"] == "H100"
            assert instance["role"] == "Bare Metal"

    @pytest.mark.asyncio
    async def test_execute_command_ssh(self, provider):
        """Test SSH command execution"""
        # Mock instance status to get IP
        mock_status = {
            "instance_id": "sv_test123",
            "status": "on",
            "primary_ipv4": "192.168.1.100",
            "instance_category": "bare_metal"
        }
        
        with patch.object(provider, 'get_instance_status', return_value=mock_status):
            with patch('subprocess.run') as mock_subprocess:
                mock_result = MagicMock()
                mock_result.returncode = 0
                mock_result.stdout = "Command executed successfully"
                mock_result.stderr = ""
                mock_subprocess.return_value = mock_result
                
                result = await provider.execute_command(
                    "sv_test123", 
                    "nvidia-smi", 
                    async_exec=False
                )
                
                assert result["instance_id"] == "sv_test123"
                assert result["exit_code"] == 0
                assert result["stdout"] == "Command executed successfully"
                assert result["execution_method"] == "ssh"
                assert result["async"] is False

    @pytest.mark.asyncio
    async def test_rate_limiting(self, provider):
        """Test rate limiting behavior"""
        # Set rate limit
        provider.rate_limit_until = datetime.now() + timedelta(seconds=60)
        
        quotes = await provider.get_instance_quotes("H100")
        
        assert len(quotes) == 1
        assert quotes[0]["rate_limited"] is True
        assert "retry_after" in quotes[0]

    @pytest.mark.asyncio
    async def test_api_error_handling(self, provider):
        """Test API error handling"""
        with patch.object(provider, '_make_request', side_effect=Exception("API Error")):
            quotes = await provider.get_instance_quotes("H100")
            assert quotes == []

    def test_extract_gpu_type(self, provider):
        """Test GPU type extraction from specs"""
        # Test NVIDIA GPU
        specs = {"gpu": {"type": "NVIDIA H100"}}
        gpu_type = provider._extract_gpu_type(specs)
        assert gpu_type == "H100"
        
        # Test unknown GPU
        specs = {"gpu": {}}
        gpu_type = provider._extract_gpu_type(specs)
        assert gpu_type == "unknown"
        
        # Test no GPU
        specs = {}
        gpu_type = provider._extract_gpu_type(specs)
        assert gpu_type == "unknown"

    def test_auth_headers(self, provider):
        """Test authentication header generation"""
        headers = provider._get_auth_headers()
        assert "Authorization" in headers
        assert headers["Authorization"] == "Bearer test_api_key_12345"

    @pytest.mark.asyncio
    async def test_stop_start_terminate_instance(self, provider):
        """Test instance lifecycle operations"""
        operations = ["stop", "start", "terminate"]
        
        for operation in operations:
            with patch.object(provider, '_make_request') as mock_request:
                mock_request.return_value = {}
                
                if operation == "stop":
                    result = await provider.stop_instance("sv_test123")
                    assert result["action"] == "stop"
                    assert result["status"] == "stopping"
                elif operation == "start":
                    result = await provider.start_instance("sv_test123")
                    assert result["action"] == "start"
                    assert result["status"] == "starting"
                elif operation == "terminate":
                    result = await provider.terminate_instance("sv_test123")
                    assert result["action"] == "terminate"
                    assert result["status"] == "terminating"

    @pytest.mark.asyncio
    async def test_virtual_machine_quotes(self, provider):
        """Test virtual machine quote retrieval"""
        # Mock VM API response
        mock_response = {
            "data": [
                {
                    "id": "vm_plan_test",
                    "attributes": {
                        "slug": "gpu-vm-h100-80",
                        "specs": {
                            "gpu": {"count": 1, "type": "NVIDIA H100", "vram_per_gpu": 80},
                            "cpu": {"cores": 8},
                            "memory": {"total": 64}
                        },
                        "pricing": {"hour": 5, "month": 200},
                        "region": "us-east"
                    }
                }
            ]
        }
        
        with patch.object(provider, '_make_request', return_value=mock_response):
            quotes = await provider._get_virtual_machine_quotes("H100")
            
            assert len(quotes) == 1
            quote = quotes[0]
            assert quote["instance_category"] == "virtual_machine"
            assert quote["isolation"] == "virtual_machine"
            assert quote["dedicated_gpu"] is True
            assert quote["virtualization"] == "kvm"

    def test_gpu_plans_mapping(self, provider):
        """Test GPU plans configuration"""
        assert "H100" in provider.GPU_PLANS
        assert "A100" in provider.GPU_PLANS
        assert "RTX4090" in provider.GPU_PLANS
        assert "RTX6000PRO" in provider.GPU_PLANS
        
        h100_plan = provider.GPU_PLANS["H100"]
        assert h100_plan["gpu_count"] == 4
        assert h100_plan["memory_gb"] == 32
        assert h100_plan["cpu_type"] == "E-2276G"

    @pytest.mark.asyncio
    async def test_missing_api_key_error(self, provider):
        """Test behavior with missing API key"""
        provider.api_key = ""
        
        with pytest.raises(Exception, match="API key not configured"):
            await provider.provision_instance(
                "latitude-bare-metal-test",
                "us-east",
                "H100"
            )
        
        quotes = await provider.get_instance_quotes("H100")
        assert quotes == []


if __name__ == "__main__":
    pytest.main([__file__])
