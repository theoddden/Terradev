# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [4.0.12] - 2026-03-12

### 🚀 **New Latitude.sh Provider Integration**
- **Dual Instance Support**: Full support for both bare metal servers AND virtual machines with GPU
- **Premium GPU Access**: NVIDIA H100, A100, RTX 4090, RTX PRO 6000 Blackwell support
- **Bare Metal Performance**: Dedicated hardware with IPMI out-of-band management
- **VM Flexibility**: GPU-enabled virtual machines with dedicated GPU resources
- **JSON:API Compliant**: Complete API specification compliance with built-in rate limiting
- **SSH Access Patterns**: Direct SSH for bare metal, container SSH for VMs
- **Real-time Pricing**: Live quotes with stock levels and instant deployment information

### 🔧 **Provider Enhancements**
- **Provider Count**: Updated to 20 supported cloud providers
- **Instance Differentiation**: Clear categorization between bare metal and virtual machine instances
- **Rate Limiting**: Automatic exponential backoff with 429 response handling
- **Error Handling**: Graceful degradation with detailed error responses
- **Test Coverage**: 13 comprehensive tests with 100% pass rate

### 📊 **Configuration & Usage**
- **Easy Setup**: Simple `LATITUDE_API_KEY` environment variable configuration
- **CLI Integration**: Full integration with existing terradev CLI commands
- **Instance Types**: Support for both `latitude-bare-metal-*` and `latitude-vm-*` instance types
- **Regional Support**: Multi-region availability with real-time stock information

### 🎯 **Technical Improvements**
- **Async Support**: Full async/await implementation throughout provider
- **Type Safety**: Complete type annotations for better IDE support
- **Documentation**: Comprehensive integration guide and API reference
- **Factory Registration**: Automatic provider registration in provider factory

---

## [2.9.8] - 2026-02-20

### 🎯 **Documentation Corrections**
- **Fixed Tier Pricing**: Corrected GPU limits to 1/8/32 for Research/Research+/Enterprise tiers
- **Removed Emojis**: Cleaned up all emoji characters from README for professional appearance
- **Removed Project Structure**: Eliminated detailed project structure section from documentation
- **Updated Version**: Bumped to v2.9.8 with corrected information

### 📚 **Documentation Accuracy**
- **Tier Limits**: Fixed concurrent instance limits (1/8/32 GPUs)
- **Professional Formatting**: Removed all emoji characters throughout README
- **Streamlined Content**: Removed unnecessary project structure details
- **Corrected Information**: Ensured all pricing and tier information is accurate

### 🔧 **Technical Updates**
- **Version Sync**: Updated README version to match package version
- **Clean Documentation**: Professional-grade README without emojis
- **Accurate Pricing**: Corrected tier pricing table with proper GPU limits

---

## [2.9.7] - 2026-02-20

### 🎯 **Major Documentation Update**
- **Complete README Overhaul**: Comprehensive documentation with GitOps automation
- **HuggingFace Spaces Integration**: One-click model deployment documentation
- **BYOAPI Security Model**: Detailed authentication and security explanations
- **Enhanced Quick Start**: 17-step comprehensive getting started guide
- **Project Structure**: Complete architecture documentation
- **Integration Guides**: Jupyter, GitHub Actions, Docker workflows

### 📚 **Documentation Features**
- **GitOps Workflows**: Production-ready ArgoCD/Flux integration
- **HF Spaces Templates**: LLM, embedding, and image model deployment
- **CLI Command Reference**: Complete command documentation with examples
- **Pricing Tiers**: Clear feature comparison across tiers
- **Security Architecture**: Zero-trust credential management
- **Integration Matrix**: W&B, Prometheus, Grafana setup guides

### 🚀 **Enhanced User Experience**
- **One-Command Deployments**: Simplified HF Spaces deployment
- **Template-Based Workflows**: Pre-configured model templates
- **Multi-Environment Support**: Dev, staging, production workflows
- **Policy as Code**: Gatekeeper/Kyverno integration
- **Manifest Cache**: Versioned deployment management

### 📖 **Documentation Structure**
- **Why Terradev**: Clear value proposition and use cases
- **Installation Guide**: Multiple installation options
- **Quick Start**: Comprehensive 17-step tutorial
- **CLI Commands**: Complete command reference
- **Integrations**: Detailed third-party service setup
- **Project Architecture**: Complete codebase structure

### 🔗 **External Links**
- **GitHub Repository**: Updated repository links
- **License Details**: Comprehensive license information
- **Integration Examples**: Real-world usage patterns
- **Security Documentation**: BYOAPI security model

---

## [2.9.6] - 2026-02-20

### 🚀 **Major Features**
- **InferX Serverless Integration**: Complete serverless AI inference platform
  - <2s cold starts with snapshot technology
  - 90% GPU utilization optimization
  - 30+ models per GPU capacity
  - OpenAI-compatible API support

### 🎯 **InferX Provider Features**
- Serverless deployment with pay-per-request pricing
- GPU slicing and multi-tenant isolation
- Snapshot technology for instant model loading
- AI-powered cost optimization with 70% savings potential
- Comprehensive Kubernetes platform deployment

### ⚡ **Performance Optimizations**
- SPDK blobstore for high-performance snapshot storage
- GPU-aware scheduling and resource pooling
- KEDA-based auto-scaling for model functions
- Custom resource definitions for model management

### 💰 **Cost Optimization**
- Spot-first GPU instance strategy (70% cost reduction)
- Resource pooling and sharing capabilities
- Tiered storage classes for different use cases
- AI-powered cost analysis and recommendations

### 🐳 **Kubernetes Platform**
- Complete InferX platform deployment automation
- GPU node pools with Karpenter integration
- Multi-tier storage classes (blobstore, cache, database)
- Network policies and security isolation
- Monitoring dashboards and metrics

### 🔧 **CLI Commands**
- `terradev inferx configure` - Provider setup
- `terradev inferx deploy` - Model deployment
- `terradev inferx status` - Deployment monitoring
- `terradev inferx list` - Model inventory
- `terradev inferx usage` - Usage statistics
- `terradev inferx quote` - Pricing information
- `terradev inferx optimize` - Cost optimization

### 📊 **Monitoring & Observability**
- Custom Prometheus metrics for InferX
- Grafana dashboards for performance monitoring
- Health checks and readiness probes
- Resource utilization tracking

### 🛡️ **Security & Isolation**
- Multi-tenant namespace isolation
- RBAC permissions for model controller
- Network policies for traffic control
- Pod security contexts and capabilities

### 📦 **Package Updates**
- Added InferX provider dependencies
- Updated Kubernetes client libraries
- Enhanced async/await support throughout
- Improved error handling and logging

### 🐛 **Bug Fixes**
- Fixed GPU resource allocation issues
- Resolved snapshot storage permissions
- Improved error messages for deployment failures
- Enhanced timeout handling for long-running operations

### 📚 **Documentation**
- Complete InferX integration guide
- Kubernetes deployment instructions
- Cost optimization best practices
- API reference documentation

---

## [2.9.5] - Previous Release

### 🔄 **Previous Features**
- Multi-cloud GPU provisioning
- GitOps automation
- HuggingFace Spaces deployment
- Cost optimization analytics
- Provider integrations (AWS, GCP, Azure, RunPod, VastAI, etc.)

---

## [Unreleased]

### 🚀 **Upcoming Features**
- Additional provider integrations
- Enhanced monitoring capabilities
- Advanced cost optimization algorithms
- Multi-region deployment support
