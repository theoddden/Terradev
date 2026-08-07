"""Aggregated MCP handler dispatch table."""

from .compute import HANDLERS as _compute_handlers
from .core import HANDLERS as _core_handlers
from .gitops import HANDLERS as _gitops_handlers
from .governance import HANDLERS as _governance_handlers
from .inference import HANDLERS as _inference_handlers
from .integrations import HANDLERS as _integrations_handlers
from .k8s import HANDLERS as _k8s_handlers
from .ml import HANDLERS as _ml_handlers
from .monitoring import HANDLERS as _monitoring_handlers
from .networking import HANDLERS as _networking_handlers
from .orchestration import HANDLERS as _orchestration_handlers
from .pricing import HANDLERS as _pricing_handlers
from .training import HANDLERS as _training_handlers

HANDLERS = {}
from .unsloth import HANDLERS as _unsloth_handlers
from .weaviate import HANDLERS as _weaviate_handlers
from .vllm_lora import HANDLERS as _vllm_lora_handlers
from .agentic import HANDLERS as _agentic_handlers
HANDLERS.update(_compute_handlers)
HANDLERS.update(_core_handlers)
HANDLERS.update(_gitops_handlers)
HANDLERS.update(_governance_handlers)
HANDLERS.update(_inference_handlers)
HANDLERS.update(_integrations_handlers)
HANDLERS.update(_k8s_handlers)
HANDLERS.update(_ml_handlers)
HANDLERS.update(_monitoring_handlers)
HANDLERS.update(_networking_handlers)
HANDLERS.update(_orchestration_handlers)
HANDLERS.update(_pricing_handlers)
HANDLERS.update(_training_handlers)
HANDLERS.update(_agentic_handlers)
HANDLERS.update(_vllm_lora_handlers)
HANDLERS.update(_weaviate_handlers)
HANDLERS.update(_unsloth_handlers)
