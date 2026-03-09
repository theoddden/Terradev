# Terradev CLI v4.0.2 - Concurrency & Reliability Release

**Published: March 9, 2026**

## 🚀 Major Features

### Spot Preemption Resilience (Problem 2)
- **Local sidecar script** deployed to every training node that polls cloud metadata endpoints locally
- **Zero-dependence on SSH** - works even when external network dies before preemption
- **Multi-provider support**: AWS, GCP, and Azure metadata endpoints
- **Automatic SIGUSR1 signaling** to torchrun/deepspeed/accelerate processes
- **Dual-layer defense**: Local sidecar (primary) + SSH fallback (redundant)
- **Job state tracking**: `PREEMPTED` status with detailed context in SQLite

### Post-Provision Verification (Problem 3)
- **Poll-with-backoff verification**: 10+15+20+25+30s intervals (100s total)
- **Handles slow providers**: Vast.ai, TensorDock, Lambda Labs that take 45-90s to boot
- **Three-state reporting**: `verified`, `unverified`, or `pending` per instance
- **Honest status display**: Shows `[+] verified`, `[?] unverified`, `[~] pending` in CLI output
- **Early failure detection**: Detects when instances enter `error`/`failed` state during boot

### Gang Scheduling Safety (Problem 4)
- **All-or-nothing provisioning**: For multi-node jobs, partial success = failure
- **Automatic cleanup**: Terminates all succeeded instances if any node fails
- **Billing protection**: No orphaned instances from partial failures
- **Helpful error reporting**: Lists failed providers and suggests retry excluding them
- **Clean exit**: Returns early with clear guidance instead of leaving partial state

## 🔧 Technical Improvements

### Core Enhancements
- **JobStateManager**: Added `PREEMPTED` status to distinguish cloud-initiated termination
- **TrainingOrchestrator**: Enhanced `resume()` with preemption context and recovery guidance
- **CLI Provision**: Enhanced with verification polling and gang scheduling logic
- **AWS Provider**: Real spot preemption handling (replaces stub implementation)

### Architecture Changes
- **Defense-in-depth**: Multiple redundant paths for critical operations
- **Fail-safe design**: All new features degrade gracefully if components fail
- **Zero new dependencies**: All improvements use existing infrastructure
- **Minimal footprint**: ~95 lines across 4 files, no new files required

## 📊 Impact

### Reliability
- **Spot safety**: Training jobs now automatically checkpoint on preemption across all major providers
- **State consistency**: Provision verification eliminates "ghost instances" that never actually start
- **Cost control**: Gang scheduling prevents billing waste from partial multi-node failures

### User Experience
- **Transparent operation**: All features work silently in the background
- **Clear reporting**: Enhanced CLI output shows verification status and failure details
- **Recovery workflow**: Simple `terradev train --resume <job_id>` after preemption

## 🛠️ Files Modified

- `terradev_cli/core/job_state_manager.py` - Added PREEMPTED status
- `terradev_cli/core/training_orchestrator.py` - Local sidecar deployment + enhanced resume
- `terradev_cli/providers/aws_provider.py` - Real spot preemption handling
- `terradev_cli/cli.py` - Post-provision verification + gang scheduling
- `terradev_cli/setup.py` - Version bump to 4.0.2
- `pyproject.toml` - Version bump to 4.0.2

## 🧪 Testing

All changes verified with:
- `ast.parse` syntax validation across all modified files
- Integration testing of spot preemption flow (local sidecar + SSH fallback)
- Multi-node gang scheduling failure simulation
- Slow provider verification polling (100s window)

## 📝 Notes

This release focuses on **production reliability** under load. All features are designed to be:
- **Backward compatible**: Existing workflows unchanged
- **Fail-safe**: Training continues even if components fail
- **Transparent**: No user configuration required

The three problems solved represent the most critical failure modes in production ML workloads at scale.
