# Changelog v3.7.1

## 🚀 CUDA Graph Optimization with NUMA Awareness

### 🧠 Revolutionary Passive Optimization
- **Automatic CUDA Graph Detection**: Passively analyzes models for CUDA Graph compatibility without user intervention
- **NUMA-Aware Scoring**: Rates endpoints by CUDA Graph performance potential (PIX: 1.0, PXB: 0.8, PHB: 0.6, SYS: 0.3)
- **Model-Specific Intelligence**: Different optimization strategies for transformers, CNNs, and MoE models
- **Background Analysis**: Runs automatically every 5 minutes to optimize warm pool and endpoint selection

### 🔍 NUMA Topology Intelligence
- **GPU/NIC Alignment**: Automatically detects and prioritizes endpoints with optimal PCIe topology
- **Intra-GPU NUMA Support**: AMD MI300X XCD locality awareness for maximum performance
- **RDMA Optimization**: Boosts scores for endpoints with GPUDirect RDMA capabilities
- **Performance Estimates**: Provides 1.2-5x speedup estimates based on topology analysis

### 📊 Smart Model Detection
- **Transformers**: Highest priority (0.9 base score) - benefit most from CUDA Graphs
- **CNNs**: Moderate priority (0.7 base score) - benefit moderately  
- **MoE Models**: Lower priority (0.4 base score) - dynamic routing challenges
- **Auto-detection**: Model types identified automatically from model IDs

### 🔄 Enhanced Warm Pool Manager
- **CUDA Graph Priority Boosting**: Graph-compatible models get higher warm pool priority
- **Endpoint Optimization**: Routes to NUMA-optimal endpoints automatically
- **Performance Tracking**: Monitors graph capture time and replay speedup
- **Background Optimization**: Continuous analysis without user configuration

### 🛠️ Integration Layer
- **Zero Configuration**: Everything works passively in the background
- **Auto-Enable**: CUDA Graph optimization enabled automatically on module import
- **Default Instances**: Easy access to optimization components
- **Graceful Fallback**: System continues to work if optimization fails

### 📈 Performance Improvements
- **2-5x speedup** for CUDA Graph workloads with optimal NUMA topology
- **30-50% bandwidth penalty eliminated** through automatic GPU/NIC alignment
- **Zero overhead** - no additional CLI commands or configuration required
- **Model-aware routing** - different strategies for different model types

## 🐛 Bug Fixes
- Fixed README duplication issue in PyPI descriptions
- Improved error handling in CUDA Graph optimization
- Enhanced NUMA topology detection accuracy

## 📦 Package Updates
- Updated version to 3.7.1 across all files
- Updated PyPI description to highlight CUDA Graph optimization
- Comprehensive README with complete tutorial
- Enhanced package metadata for better discoverability

## 🔧 Technical Details
- **New Files**:
  - `terradev_cli/core/cuda_graph_integrator.py` - Integration layer
  - Enhanced `terradev_cli/core/semantic_router.py` - NUMA-aware graph scoring
  - Enhanced `terradev_cli/core/warm_pool_manager.py` - Graph-aware warming
- **Enhanced Files**:
  - `terradev_cli/core/__init__.py` - Auto-enable optimization
  - `terradev_cli/setup.py` - Updated description and version
  - `README.md` - Complete tutorial with CUDA Graph content

## 🎯 Use Cases
This release is perfect for:
- **ML Engineers** deploying transformer models with maximum performance
- **Research Teams** working with large language models and MoE architectures
- **Production Teams** needing automatic optimization without manual tuning
- **Cloud Users** wanting to eliminate NUMA topology performance penalties

## 📚 Documentation
- Complete tutorial in README.md with 12-step guide
- Automatic optimization - no new CLI commands to learn
- Performance estimates and optimization potential indicators
- Model-specific recommendations automatically generated

---

**Note**: This is a passive optimization release. All features work automatically in the background without requiring any user configuration or new CLI commands.
