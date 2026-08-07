"""Aggregated MCP tool schemas."""

from .compute import TOOLS as _compute_tools
from .core import TOOLS as _core_tools
from .gitops import TOOLS as _gitops_tools
from .governance import TOOLS as _governance_tools
from .inference import TOOLS as _inference_tools
from .integrations import TOOLS as _integrations_tools
from .k8s import TOOLS as _k8s_tools
from .ml import TOOLS as _ml_tools
from .monitoring import TOOLS as _monitoring_tools
from .networking import TOOLS as _networking_tools
from .orchestration import TOOLS as _orchestration_tools
from .pricing import TOOLS as _pricing_tools
from .training import TOOLS as _training_tools

TOOLS = []
from .vllm_lora import TOOLS as _vllm_lora_tools
from .agentic import TOOLS as _agentic_tools
TOOLS.extend(_compute_tools)
TOOLS.extend(_core_tools)
TOOLS.extend(_gitops_tools)
TOOLS.extend(_governance_tools)
TOOLS.extend(_inference_tools)
TOOLS.extend(_integrations_tools)
TOOLS.extend(_k8s_tools)
TOOLS.extend(_ml_tools)
TOOLS.extend(_monitoring_tools)
TOOLS.extend(_networking_tools)
TOOLS.extend(_orchestration_tools)
TOOLS.extend(_pricing_tools)
TOOLS.extend(_training_tools)
TOOLS.extend(_agentic_tools)
TOOLS.extend(_vllm_lora_tools)
