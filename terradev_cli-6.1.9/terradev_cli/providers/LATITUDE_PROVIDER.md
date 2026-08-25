# Latitude.sh Provider Integration

## Overview

The Latitude.sh provider provides integration with Latitude.sh's bare metal and virtual machine GPU cloud services. This provider supports both dedicated bare metal servers and virtual machines with GPU capabilities.

## Features

### **Dual Instance Support**
- **Bare Metal Servers**: Full dedicated hardware with IPMI access
- **Virtual Machines**: GPU-enabled VMs with dedicated GPU resources

### **GPU Support**
- **NVIDIA H100** (4x GPU configurations)
- **NVIDIA A100** (2x GPU configurations) 
- **NVIDIA RTX 4090** (2x GPU configurations)
- **NVIDIA RTX PRO 6000 Blackwell** (2x GPU configurations)

### **Key Capabilities**
- JSON:API compliant implementation
- Built-in rate limiting with exponential backoff
- SSH access for both bare metal and VM instances
- IPMI out-of-band management for bare metal
- Real-time pricing and availability
- Multi-region support

## Authentication

### API Key Setup
1. Log in to [Latitude.sh Dashboard](https://www.latitude.sh/dashboard)
2. Navigate to API Keys section
3. Create new API key with appropriate permissions
4. Configure IP restrictions for enhanced security (optional)

### Environment Configuration
```bash
export LATITUDE_API_KEY="your_api_key_here"
```

### Credentials Format
```python
credentials = {
    "api_key": "your_latitude_api_key"
}
```

## Usage Examples

### Basic Provider Initialization
```python
from terradev_cli.providers.provider_factory import ProviderFactory

factory = ProviderFactory()
provider = factory.create_provider("latitude", {
    "api_key": "your_api_key"
})
```

### Getting GPU Instance Quotes
```python
# Get quotes for H100 GPU instances
quotes = await provider.get_instance_quotes("H100")

for quote in quotes:
    print(f"Type: {quote['instance_type']}")
    print(f"Price: ${quote['price_per_hour']}/hour")
    print(f"Category: {quote['instance_category']}")
    print(f"Isolation: {quote['isolation']}")
    print(f"SSH Access: {quote['ssh_access']}")
    print(f"GPU Count: {quote['gpu_count']}")
```

### Provisioning Bare Metal Instance
```python
result = await provider.provision_instance(
    instance_type="latitude-bare-metal-g3-h100-medium-43",
    region="Brazil", 
    gpu_type="H100",
    ssh_key_ids=["ssh_key_123"],
    user_data="#cloud-init\npackages: [nvidia-driver-470]"
)

print(f"Instance ID: {result['instance_id']}")
print(f"SSH Access: {result['ssh_access']}")
print(f"IPMI Access: {result['ipmi_access']}")
print(f"Primary IP: {result['primary_ipv4']}")
```

### Provisioning Virtual Machine
```python
result = await provider.provision_instance(
    instance_type="latitude-vm-gpu-h100-80",
    region="us-east",
    gpu_type="H100"
)

print(f"VM ID: {result['instance_id']}")
print(f"Dedicated GPU: {result['dedicated_gpu']}")
print(f"Virtualization: {result['virtualization']}")
```

### Instance Management
```python
# Get instance status
status = await provider.get_instance_status("sv_12345")
print(f"Status: {status['status']}")
print(f"IP: {status['primary_ipv4']}")

# Execute commands via SSH
result = await provider.execute_command(
    "sv_12345", 
    "nvidia-smi",
    async_exec=False
)
print(f"GPU Info: {result['stdout']}")

# Power management
await provider.stop_instance("sv_12345")
await provider.start_instance("sv_12345")
await provider.terminate_instance("sv_12345")
```

## Instance Categories

### Bare Metal Servers
- **Isolation**: Full dedicated hardware
- **Access**: Direct SSH + IPMI management
- **Performance**: Bare metal performance with no virtualization overhead
- **Use Cases**: High-performance computing, dedicated workloads
- **Return Format**:
```json
{
  "instance_category": "bare_metal",
  "isolation": "bare_metal", 
  "ssh_access": true,
  "ipmi_access": true,
  "role": "Bare Metal"
}
```

### Virtual Machines
- **Isolation**: Virtualized environment (KVM)
- **Access**: SSH access (container/VM level)
- **Performance**: Near-bare metal with dedicated GPU
- **Use Cases**: GPU workloads, ML inference, flexible scaling
- **Return Format**:
```json
{
  "instance_category": "virtual_machine",
  "isolation": "virtual_machine",
  "ssh_access": true,
  "ipmi_access": false,
  "dedicated_gpu": true,
  "virtualization": "kvm"
}
```

## Pricing Structure

### Bare Metal Pricing
- **Billing**: Hourly, monthly, yearly options
- **GPU Examples**:
  - H100 (4x): ~$10/hour
  - A100 (2x): ~$8/hour
  - RTX 4090 (2x): ~$4/hour

### Virtual Machine Pricing
- **Billing**: Hourly, monthly options
- **GPU Examples**:
  - H100 (1x): ~$5/hour
  - A100 (1x): ~$4/hour

## Rate Limiting

The provider includes built-in rate limiting:
- **Automatic Detection**: Parses 429 responses
- **Retry Logic**: Exponential backoff with `retry_after` header
- **Graceful Degradation**: Returns rate-limited status instead of failures

## Error Handling

### Common Error Scenarios
1. **Missing API Key**: Returns empty results with clear error
2. **Rate Limiting**: Automatic retry with user feedback
3. **Instance Not Found**: Attempts both bare metal and VM endpoints
4. **Network Issues**: Graceful fallback with debug logging

### Error Response Format
```python
{
    "provider": "latitude",
    "available": false,
    "reason": "Rate limited",
    "retry_after": "60 seconds",
    "rate_limited": true
}
```

## SSH Access

### Bare Metal SSH
- **User**: `root`
- **Port**: `22`
- **Key Management**: Upload SSH keys via dashboard or API
- **IPMI**: Out-of-band management on separate interface

### Virtual Machine SSH
- **User**: `root`
- **Port**: `22` 
- **Key Management**: Same as bare metal
- **Container Access**: VM-level isolation

## Regional Availability

### Supported Regions
- **Brazil** (SAO)
- **United States** (ASH, multiple locations)
- **Europe** (various facilities)
- **Asia-Pacific** (expanding coverage)

### Stock Levels
- **Real-time**: `stock_level` field in quotes
- **Instant Deployment**: `deploys_instantly` array
- **Capacity Reservations**: Available for high-demand GPUs

## Integration Notes

### Provider Registration
The Latitude provider is automatically registered in the provider factory:
```python
from terradev_cli.providers.provider_factory import ProviderFactory

factory = ProviderFactory()
provider = factory.create_provider("latitude", credentials)
```

### API Compliance
- **JSON:API**: Full compliance with JSON:API specification
- **Content Types**: `application/vnd.api+json` for requests
- **Error Handling**: Structured error responses
- **Pagination**: Supported for list operations

### Dependencies
- **aiohttp**: HTTP client for async requests
- **asyncio**: Async/await support
- **Standard Library**: No additional dependencies required

## Troubleshooting

### Common Issues
1. **API Key Not Working**: Verify key permissions and IP restrictions
2. **SSH Connection Failed**: Check instance status and IP availability
3. **Rate Limiting**: Wait for retry period or implement backoff
4. **Instance Not Found**: Verify instance ID and category (bare metal vs VM)

### Debug Logging
Enable debug logging for troubleshooting:
```python
import logging
logging.getLogger("terradev_cli.providers.latitude_provider").setLevel(logging.DEBUG)
```

### Testing
Run the comprehensive test suite:
```bash
python -m pytest tests/test_latitude_provider.py -v
```

## API Reference

### Core Methods
- `get_instance_quotes(gpu_type, region)` - Get pricing/availability
- `provision_instance(instance_type, region, gpu_type)` - Create instance
- `get_instance_status(instance_id)` - Check instance status
- `list_instances()` - List all instances
- `execute_command(instance_id, command, async_exec)` - SSH execution
- `stop_instance(instance_id)` - Power off instance
- `start_instance(instance_id)` - Power on instance
- `terminate_instance(instance_id)` - Destroy instance

### Response Fields
All responses include standardized fields:
- `provider`: "latitude"
- `instance_category`: "bare_metal" | "virtual_machine"
- `isolation`: Hardware isolation level
- `ssh_access`: SSH availability
- `gpu_type`: GPU model (H100, A100, etc.)
- `region`: Geographic region
- `status`: Current instance status

## Future Enhancements

### Planned Features
- **Auto-scaling**: Automatic instance scaling based on demand
- **Spot Instances**: Pre-emptible pricing options
- **Load Balancing**: Integrated load balancer support
- **Monitoring**: Enhanced metrics and alerting
- **Storage**: Persistent storage options for VMs

### API Evolution
- **Virtual Machine Endpoints**: Full VM API discovery and integration
- **GPU Sharing**: Multi-tenant GPU instance support
- **Custom Images**: User-defined machine images
- **Network APIs**: Advanced networking configuration

## Support

### Documentation
- [Latitude.sh API Docs](https://www.latitude.sh/docs/api-reference)
- [GPU Instances Guide](https://www.latitude.sh/docs/vms/gpu-instances)
- [Provider Source Code](terradev_cli/providers/latitude_provider.py)

### Contact
- **Latitude.sh Support**: https://www.latitude.sh/support
- **Terradev Issues**: GitHub repository issue tracker
- **Community**: Discord/Slack channels for developer support
