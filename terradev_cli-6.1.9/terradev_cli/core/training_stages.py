#!/usr/bin/env python3
"""
Training Stages — declarative SFT / DPO / GRPO stage configs and command builders.

Supports:
  - Frameworks: unsloth, trl, axolotl, llama-factory, ms-swift, openrlhf
  - DPO variants: dpo, simpo, kto, orpo
  - Per-stage resource estimation (GPU type / count / multi-node)
  - Generated training commands and config files
"""

import json
import logging
import os
import re
import shlex
import tempfile
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


class StageType(Enum):
    SFT = "sft"
    DPO = "dpo"
    GRPO = "grpo"


class Framework(Enum):
    UNSLOTH = "unsloth"
    TRL = "trl"
    AXOLOTL = "axolotl"
    LLAMA_FACTORY = "llama-factory"
    MS_SWIFT = "ms-swift"
    OPENRLHF = "openrlhf"


class DPOAlgorithm(Enum):
    DPO = "dpo"
    SIMPO = "simpo"
    KTO = "kto"
    ORPO = "orpo"
    CPO = "cpo"


STAGE_TYPE_CHOICES = [m.value for m in StageType]
FRAMEWORK_CHOICES = [m.value for m in Framework]
DPO_ALGORITHM_CHOICES = [m.value for m in DPOAlgorithm]


@dataclass
class StageConfig:
    """Single training stage configuration."""

    type: str  # 'sft' | 'dpo' | 'grpo'
    model: str = ""
    data: str = ""
    base_checkpoint: str = ""
    checkpoint: str = ""
    framework: str = "unsloth"
    algorithm: str = "dpo"  # dpo | simpo | kto | orpo | cpo
    reward_fn: str = "verifiable"  # for GRPO
    rollout_provider: str = "auto"
    trainer_provider: str = "auto"
    provider: str = "auto"  # single provider for SFT/DPO, or 'auto'
    gpu_type: str = ""
    gpu_count: int = 1
    node_count: int = 1
    gpus_per_node: int = 8
    spot: bool = False
    max_price: float = 0.0
    output_dir: str = ""
    output_bucket: str = ""  # s3://... for checkpoint upload
    env: Dict[str, str] = field(default_factory=dict)
    extra_args: List[str] = field(default_factory=list)
    name: str = ""
    num_train_epochs: int = 1
    per_device_batch_size: int = 1
    gradient_accumulation_steps: int = 4
    learning_rate: float = 2e-4
    warmup_ratio: float = 0.1
    beta: float = 0.1
    max_seq_length: int = 2048
    lora_rank: int = 64
    lora_alpha: int = 16
    lora_dropout: float = 0.0
    use_4bit: bool = True
    use_16bit: bool = False
    bf16: bool = True
    fp16: bool = False
    seed: int = 42
    deepspeed: str = ""  # path to deepspeed config (optional)
    ray_address: str = ""  # for openrlhf
    num_generations: int = 8  # GRPO group size

    def __post_init__(self):
        if not self.name:
            self.name = f"{self.type}-{self.framework}"
        if not self.output_dir:
            self.output_dir = f"./checkpoints/{self.name}"
        if not self.checkpoint:
            self.checkpoint = self.output_dir


@dataclass
class PipelineConfig:
    """Multi-stage training pipeline configuration."""

    name: str = "training-pipeline"
    stages: List[StageConfig] = field(default_factory=list)
    defaults: Dict[str, Any] = field(default_factory=dict)
    teardown: bool = False
    checkpoint_bucket: str = ""  # base s3:// bucket for inter-stage handoff
    dry_run: bool = False

    @classmethod
    def from_yaml(cls, path: str) -> "PipelineConfig":
        import yaml

        with open(path) as f:
            data = yaml.safe_load(f) or {}
        defaults = data.get("defaults", {})
        stages = []
        for i, raw in enumerate(data.get("stages", [])):
            merged = {**defaults, **raw}
            if "type" not in merged:
                raise ValueError(f"Stage {i} missing 'type'")
            merged["type"] = merged["type"].lower()
            # checkpoint handoff: if next stage doesn't set base_checkpoint, previous checkpoint is used
            stage = StageConfig(**{k: v for k, v in merged.items() if k in StageConfig.__dataclass_fields__})
            stages.append(stage)
        # Resolve base_checkpoint handoffs
        for i, stage in enumerate(stages):
            if i > 0 and not stage.base_checkpoint:
                stage.base_checkpoint = stages[i - 1].checkpoint
            if not stage.checkpoint and stage.output_dir:
                stage.checkpoint = stage.output_dir
        return cls(
            name=data.get("name", "training-pipeline"),
            stages=stages,
            defaults=defaults,
            teardown=data.get("teardown", False),
            checkpoint_bucket=data.get("checkpoint_bucket", ""),
            dry_run=data.get("dry_run", False),
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "stages": [s.__dict__ for s in self.stages],
            "defaults": self.defaults,
            "teardown": self.teardown,
            "checkpoint_bucket": self.checkpoint_bucket,
            "dry_run": self.dry_run,
        }


def _model_billion_params(model_name: str) -> float:
    """Crude model-size heuristic from the model name/id."""
    m = re.search(r"(\d+(?:\.\d+)?)\s*[bB]", model_name)
    if m:
        return float(m.group(1))
    if "8b" in model_name.lower() or "-8B" in model_name:
        return 8.0
    if "70b" in model_name.lower() or "-70B" in model_name:
        return 70.0
    if "13b" in model_name.lower() or "-13B" in model_name:
        return 13.0
    if "7b" in model_name.lower() or "-7B" in model_name:
        return 7.0
    if "1b" in model_name.lower() or "-1B" in model_name:
        return 1.0
    if "405b" in model_name.lower() or "-405B" in model_name:
        return 405.0
    return 8.0


def estimate_resource(
    stage: StageConfig,
) -> Tuple[str, int, int, int, float]:
    """
    Estimate (gpu_type, total_gpus, node_count, gpus_per_node, min_vram_gb) for a stage.

    DPO needs more memory than SFT (ref model copy).
    GRPO needs split rollout/trainer resources.
    """
    params = _model_billion_params(stage.model)

    # Base multiplier on memory vs SFT
    memory_multiplier = 1.0
    if stage.type == StageType.DPO.value:
        if stage.algorithm == "dpo":
            memory_multiplier = 1.8
        elif stage.algorithm in ("simpo", "kto", "orpo", "cpo"):
            memory_multiplier = 1.3
    elif stage.type == StageType.GRPO.value:
        # Rollout memory is mostly inference; trainer memory is training
        memory_multiplier = 1.2

    # Total effective parameter memory in GB (very rough, includes optimizer/gradients)
    # 4 bytes/param + 8 bytes/adam states + 4 bytes/grad ~ 16 bytes/param for full finetune
    # LoRA reduces this dramatically
    if stage.lora_rank > 0:
        effective_params = params * 0.1  # ~10% for LoRA overhead
    else:
        effective_params = params

    vram_gb = effective_params * 16 * memory_multiplier / 1024  # rough
    # at least 1 GB
    vram_gb = max(vram_gb, 1.0)

    # Pick GPU type
    if stage.gpu_type:
        gpu_type = stage.gpu_type
    else:
        if vram_gb > 70:
            gpu_type = "H100"
        elif vram_gb > 40:
            gpu_type = "A100"
        elif vram_gb > 24:
            gpu_type = "A100"
        else:
            gpu_type = "A100"  # default for training (other options: A10, RTX4090, etc.)

    # Node / GPU count
    if stage.gpu_count and stage.node_count:
        total_gpus = stage.gpu_count
        node_count = stage.node_count
        gpus_per_node = max(1, total_gpus // max(1, node_count))
    else:
        # Auto node count for 70B+ or high vram
        gpus_per_node = stage.gpus_per_node or 8
        if vram_gb > 80:
            total_gpus = max(16, gpus_per_node * 2)
        elif vram_gb > 40:
            total_gpus = gpus_per_node
        else:
            total_gpus = 1
        node_count = max(1, total_gpus // gpus_per_node)

    return gpu_type, total_gpus, node_count, gpus_per_node, vram_gb


def _safe_quote(s: str) -> str:
    """Return a shell-safe quoted string."""
    return shlex.quote(s)


def _normalize_path(s: str) -> str:
    return os.path.expanduser(s)


def _write_temp_script(code: str, suffix: str = ".py") -> str:
    fd, path = tempfile.mkstemp(prefix="terradev_train_", suffix=suffix)
    with os.fdopen(fd, "w") as f:
        f.write(code)
    return path


def _write_temp_yaml(content: str) -> str:
    fd, path = tempfile.mkstemp(prefix="terradev_train_", suffix=".yaml")
    with os.fdopen(fd, "w") as f:
        f.write(content)
    return path


def _wrap_command_in_script(
    cmd: List[str],
    extra_env: Optional[Dict[str, str]] = None,
    extra_files: Optional[Dict[str, str]] = None,
    config_file: Optional[str] = None,
    config_index: Optional[int] = None,
    guard_rank_zero: bool = False,
) -> str:
    """
    Wrap an arbitrary CLI command in a throw-away Python script.

    This lets the TrainingOrchestrator run torchrun/deepspeed/accelerate
    against a single Python file even when the underlying framework is a
    console-script CLI.

    extra_files is a dict of {filename: content} that will be written next to
    the wrapper at runtime, making the wrapper safe to copy to remote nodes
    without carrying local temp paths. If config_file and config_index are
    supplied the matching cmd element is rewritten to the runtime path.
    """
    env_dict = dict(extra_env or {})
    files = dict(extra_files or {})
    code = f"""#!/usr/bin/env python3
import os, subprocess, sys
from pathlib import Path

script_dir = Path(__file__).parent.resolve()

# Write any embedded config / data files next to this wrapper on the remote node.
extra_files = {files!r}
for name, content in extra_files.items():
    (script_dir / name).write_text(content)

cmd = {cmd!r}
config_file = {config_file!r}
config_index = {config_index!r}
if config_file and config_index is not None:
    cmd[config_index] = str(script_dir / config_file)

env = os.environ.copy()
env.update({env_dict!r})
"""
    if guard_rank_zero:
        code += """
# CLI-based frameworks are not distributed through torchrun; only run on the
# global rank-0 worker. Other workers exit cleanly so torchrun does not fail.
local_rank = int(os.environ.get("LOCAL_RANK", "0"))
rank = int(os.environ.get("RANK", "0"))
if local_rank == 0 and rank == 0:
    sys.exit(subprocess.call(cmd, env=env))
sys.exit(0)
"""
    else:
        code += """
sys.exit(subprocess.call(cmd, env=env))
"""
    return _write_temp_script(code)


def _base_env(stage: StageConfig) -> Dict[str, str]:
    env = dict(stage.env)
    env.setdefault("PYTHONUNBUFFERED", "1")
    env.setdefault("TOKENIZERS_PARALLELISM", "false")
    if stage.seed:
        env.setdefault("SEED", str(stage.seed))
    return env


def _torch_dtype(stage: StageConfig) -> str:
    if stage.bf16:
        return "bfloat16"
    if stage.fp16:
        return "float16"
    if stage.use_4bit:
        return "bfloat16"
    return "float32"


def _unsloth_sft_script(stage: StageConfig, base_model: str, output_dir: str) -> str:
    dtype = _torch_dtype(stage)
    load_dtype = "None" if not stage.use_4bit else "torch.bfloat16"
    return f'''#!/usr/bin/env python3
import os, json
from unsloth import FastLanguageModel, is_bfloat16_supported
from unsloth import UnslothTrainer, UnslothTrainingArguments
from datasets import load_dataset

model_name = {_safe_quote(base_model)}
output_dir = {_safe_quote(output_dir)}
data_path = {_safe_quote(stage.data)}

# Load model
dtype = {load_dtype}
use_4bit = {str(stage.use_4bit)}
max_seq_length = {stage.max_seq_length}

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=model_name,
    max_seq_length=max_seq_length,
    dtype=dtype,
    load_in_4bit=use_4bit,
)

model = FastLanguageModel.get_peft_model(
    model,
    r={stage.lora_rank},
    lora_alpha={stage.lora_alpha},
    lora_dropout={stage.lora_dropout},
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
)

# Dataset formatting (assumes jsonl with 'text' field; adapt as needed)
dataset = load_dataset("json", data_files=data_path, split="train")
if "text" not in dataset.column_names:
    # fallback: concatenate first string columns
    cols = [c for c in dataset.column_names if isinstance(dataset[0].get(c), str)]
    dataset = dataset.map(lambda ex: {{"text": " ".join(str(ex[c]) for c in cols)}})

def formatting_prompts_func(examples):
    return {{"text": examples["text"]}}

trainer = UnslothTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset,
    dataset_text_field="text",
    max_seq_length=max_seq_length,
    args=UnslothTrainingArguments(
        output_dir=output_dir,
        num_train_epochs={stage.num_train_epochs},
        per_device_train_batch_size={stage.per_device_batch_size},
        gradient_accumulation_steps={stage.gradient_accumulation_steps},
        learning_rate={stage.learning_rate},
        warmup_ratio={stage.warmup_ratio},
        bf16={str(stage.bf16).lower()},
        fp16={str(stage.fp16).lower()},
        seed={stage.seed},
        logging_steps=10,
        save_strategy="epoch",
        optim="adamw_8bit",
    ),
)

trainer.train(resume_from_checkpoint=None)
trainer.save_model(output_dir)
tokenizer.save_pretrained(output_dir)
print(json.dumps({{"status": "completed", "output_dir": output_dir}}))
'''


def _unsloth_dpo_script(stage: StageConfig, base_model: str, output_dir: str) -> str:
    dtype = _torch_dtype(stage)
    return f'''#!/usr/bin/env python3
import os, json
from unsloth import FastLanguageModel
from trl import DPOTrainer
from datasets import load_dataset
from transformers import TrainingArguments

dataset = load_dataset("json", data_files={_safe_quote(stage.data)}, split="train")
if "chosen" not in dataset.column_names or "rejected" not in dataset.column_names:
    raise ValueError("DPO dataset requires 'chosen' and 'rejected' fields")

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name={_safe_quote(base_model)},
    max_seq_length={stage.max_seq_length},
    dtype={"None" if not stage.use_4bit else "torch.bfloat16"},
    load_in_4bit={str(stage.use_4bit).lower()},
)

model = FastLanguageModel.get_peft_model(
    model,
    r={stage.lora_rank},
    lora_alpha={stage.lora_alpha},
    lora_dropout={stage.lora_dropout},
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
)

args = TrainingArguments(
    output_dir={_safe_quote(output_dir)},
    num_train_epochs={stage.num_train_epochs},
    per_device_train_batch_size={stage.per_device_batch_size},
    gradient_accumulation_steps={stage.gradient_accumulation_steps},
    learning_rate={stage.learning_rate},
    warmup_ratio={stage.warmup_ratio},
    bf16={str(stage.bf16).lower()},
    fp16={str(stage.fp16).lower()},
    seed={stage.seed},
    logging_steps=10,
    save_strategy="epoch",
)

trainer = DPOTrainer(
    model=model,
    ref_model=None,  # PEFT-based
    args=args,
    train_dataset=dataset,
    tokenizer=tokenizer,
    beta={stage.beta},
)

trainer.train()
trainer.save_model(output_dir)
tokenizer.save_pretrained(output_dir)
print(json.dumps({{"status": "completed", "output_dir": output_dir}}))
'''


def _trl_cli_command(stage: StageConfig, base_model: str, output_dir: str, trl_cmd: str) -> List[str]:
    cmd = [
        "trl",
        trl_cmd,
        f"--model_name_or_path={base_model}",
        f"--dataset_name_or_path={stage.data}",
        f"--output_dir={output_dir}",
        f"--num_train_epochs={stage.num_train_epochs}",
        f"--per_device_train_batch_size={stage.per_device_batch_size}",
        f"--gradient_accumulation_steps={stage.gradient_accumulation_steps}",
        f"--learning_rate={stage.learning_rate}",
        f"--warmup_ratio={stage.warmup_ratio}",
        f"--seed={stage.seed}",
        f"--max_seq_length={stage.max_seq_length}",
    ]
    if stage.bf16:
        cmd.append("--bf16")
    if stage.fp16:
        cmd.append("--fp16")
    if stage.deepspeed:
        cmd.append(f"--deepspeed={stage.deepspeed}")
    if stage.lora_rank > 0:
        cmd.append(f"--lora_r={stage.lora_rank}")
        cmd.append(f"--lora_alpha={stage.lora_alpha}")
        cmd.append("--use_peft")
    cmd.extend(stage.extra_args)
    return cmd


def _unsloth_grpo_script(stage: StageConfig, base_model: str, output_dir: str) -> str:
    dtype = _torch_dtype(stage)
    return f'''#!/usr/bin/env python3
import json, re, os
from unsloth import FastLanguageModel, GRPOConfig, GRPOTrainer
from datasets import load_dataset

# GRPO via native Unsloth GRPOTrainer (>= unsloth 2025.02)
# The prompt field is mapped to a chat-style list of messages; ground_truth is
# optional and used by the default reward if present.
dataset = load_dataset("json", data_files={_safe_quote(stage.data)}, split="train")

def _make_conversation(ex):
    if "prompt" in ex:
        return {{"prompt": [{{"role": "user", "content": str(ex["prompt"])}}]}}
    if "text" in ex:
        return {{"prompt": [{{"role": "user", "content": str(ex["text"])}}]}}
    if "instruction" in ex:
        return {{"prompt": [{{"role": "user", "content": str(ex["instruction"])}}]}}
    return {{"prompt": [{{"role": "user", "content": str(ex)}}]}}

if "prompt" not in dataset.column_names:
    dataset = dataset.map(_make_conversation, batched=False)

if "ground_truth" not in dataset.column_names:
    dataset = dataset.map(lambda _: {{"ground_truth": ""}}, batched=False)

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name={_safe_quote(base_model)},
    max_seq_length={stage.max_seq_length},
    dtype={_safe_quote(dtype)},
    load_in_4bit={str(stage.use_4bit).lower()},
)

model = FastLanguageModel.get_peft_model(
    model,
    r={stage.lora_rank},
    lora_alpha={stage.lora_alpha},
    lora_dropout={stage.lora_dropout},
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
)

def reward_func(completions, ground_truth=None, **kwargs):
    scores = [0.0] * len(completions)
    if ground_truth:
        for i, (c, gt) in enumerate(zip(completions, ground_truth)):
            if gt and str(gt).strip() and str(gt).strip().lower() in c.lower():
                scores[i] += 1.0
    for i, c in enumerate(completions):
        if re.search(r"<answer>.*?</answer>", c, re.S):
            scores[i] += 0.5
    return scores

training_args = GRPOConfig(
    output_dir={_safe_quote(output_dir)},
    num_train_epochs={stage.num_train_epochs},
    per_device_train_batch_size={stage.per_device_batch_size},
    gradient_accumulation_steps={stage.gradient_accumulation_steps},
    learning_rate={stage.learning_rate},
    warmup_ratio={stage.warmup_ratio},
    num_generations={stage.num_generations},
    max_prompt_length={stage.max_seq_length},
    max_completion_length={stage.max_seq_length},
    bf16={str(stage.bf16).lower()},
    fp16={str(stage.fp16).lower()},
    seed={stage.seed},
    logging_steps=10,
    save_strategy="epoch",
)

trainer = GRPOTrainer(
    model=model,
    processing_class=tokenizer,
    reward_funcs=[reward_func],
    train_dataset=dataset,
    args=training_args,
)

trainer.train()
trainer.save_model(output_dir)
tokenizer.save_pretrained(output_dir)
print(json.dumps({{"status": "completed", "output_dir": output_dir}}))
'''


def _trl_grpo_script(stage: StageConfig, base_model: str, output_dir: str) -> str:
    return f'''#!/usr/bin/env python3
import json
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import GRPOTrainer, GRPOConfig
from datasets import load_dataset

# GRPO in TRL (>=0.15): train reward model on rule/verifiable reward function
dataset = load_dataset("json", data_files={_safe_quote(stage.data)}, split="train")
if "prompt" not in dataset.column_names:
    raise ValueError("GRPO dataset requires 'prompt' field")

model = AutoModelForCausalLM.from_pretrained({_safe_quote(base_model)})
tokenizer = AutoTokenizer.from_pretrained({_safe_quote(base_model)})

# Minimal verifiable reward: count of a target substring in generated output
reward_fn_name = {_safe_quote(stage.reward_fn)}
if reward_fn_name == "verifiable":
    def reward_func(prompts, completions, **kwargs):
        return [1.0 if "42" in c else 0.0 for c in completions]
else:
    def reward_func(prompts, completions, **kwargs):
        return [0.0 for _ in completions]

training_args = GRPOConfig(
    output_dir={_safe_quote(output_dir)},
    num_train_epochs={stage.num_train_epochs},
    per_device_train_batch_size={stage.per_device_batch_size},
    gradient_accumulation_steps={stage.gradient_accumulation_steps},
    learning_rate={stage.learning_rate},
    warmup_ratio={stage.warmup_ratio},
    num_generations={stage.num_generations},
    bf16={str(stage.bf16).lower()},
    fp16={str(stage.fp16).lower()},
    seed={stage.seed},
)

trainer = GRPOTrainer(
    model=model,
    reward_funcs=reward_func,
    args=training_args,
    train_dataset=dataset,
    processing_class=tokenizer,
)

trainer.train()
trainer.save_model(output_dir)
tokenizer.save_pretrained(output_dir)
print(json.dumps({{"status": "completed", "output_dir": output_dir}}))
'''


def _openrlhf_grpo_command(stage: StageConfig, base_model: str, output_dir: str) -> List[str]:
    # Ray-based OpenRLHF GRPO command. Requires a Ray cluster already running (or ray start).
    # rollout-provider and trainer-provider are encoded as Ray resources tags.
    env = {
        "RAY_ADDRESS": stage.ray_address or os.environ.get("RAY_ADDRESS", "auto"),
    }
    cmd = [
        "ray",
        "job",
        "submit",
        "--working-dir",
        ".",
        "--",
        "python",
        "-m",
        "openrlhf.cli.train_ppo",
        f"--actor_model={base_model}",
        f"--dataset={stage.data}",
        f"--output_dir={output_dir}",
        f"--num_epochs={stage.num_train_epochs}",
        f"--lr={stage.learning_rate}",
        f"--train_batch_size={stage.per_device_batch_size * stage.gradient_accumulation_steps}",
        f"--rollout_batch_size={stage.num_generations}",
        f"--seed={stage.seed}",
        "--bf16",
        f"--advantage_estimator=group_norm",
    ]
    if stage.deepspeed:
        cmd.append(f"--deepspeed={stage.deepspeed}")
    cmd.extend(stage.extra_args)
    return cmd, env


def _axolotl_yaml(stage: StageConfig, base_model: str, output_dir: str) -> str:
    base = f"""base_model: {base_model}
model_type: AutoModelForCausalLM
tokenizer_type: AutoTokenizer

datasets:
  - path: {stage.data}
    type: {stage.type}

sequence_len: {stage.max_seq_length}
num_epochs: {stage.num_train_epochs}
micro_batch_size: {stage.per_device_batch_size}
gradient_accumulation_steps: {stage.gradient_accumulation_steps}
learning_rate: {stage.learning_rate}
warmup_ratio: {stage.warmup_ratio}
optimizer: adamw_bnb_8bit
lr_scheduler: cosine
bf16: {str(stage.bf16).lower()}
fp16: {str(stage.fp16).lower()}
seed: {stage.seed}

lora_r: {stage.lora_rank}
lora_alpha: {stage.lora_alpha}
lora_dropout: {stage.lora_dropout}
lora_target_linear: true

output_dir: {output_dir}
"""
    return base


def _llama_factory_yaml(stage: StageConfig, base_model: str, output_dir: str) -> str:
    return f"""model_name_or_path: {base_model}
stage: {stage.type}
do_train: true
dataset: {stage.data}
output_dir: {output_dir}
num_train_epochs: {stage.num_train_epochs}
per_device_train_batch_size: {stage.per_device_batch_size}
gradient_accumulation_steps: {stage.gradient_accumulation_steps}
learning_rate: {stage.learning_rate}
warmup_ratio: {stage.warmup_ratio}
bf16: {str(stage.bf16).lower()}
fp16: {str(stage.fp16).lower()}
seed: {stage.seed}
max_length: {stage.max_seq_length}
{"lora_rank: " + str(stage.lora_rank) if stage.lora_rank > 0 else ""}
{"lora_alpha: " + str(stage.lora_alpha) if stage.lora_rank > 0 else ""}
"""


def _ms_swift_command(stage: StageConfig, base_model: str, output_dir: str) -> List[str]:
    # ms-swift sft / dpo / rlhf CLI
    swift_cmd = "dpo" if stage.type == "dpo" else ("rlhf" if stage.type == "grpo" else "sft")
    cmd = [
        "swift",
        swift_cmd,
        f"--model={base_model}",
        f"--dataset={stage.data}",
        f"--output_dir={output_dir}",
        f"--num_train_epochs={stage.num_train_epochs}",
        f"--per_device_train_batch_size={stage.per_device_batch_size}",
        f"--gradient_accumulation_steps={stage.gradient_accumulation_steps}",
        f"--learning_rate={stage.learning_rate}",
        f"--warmup_ratio={stage.warmup_ratio}",
        f"--seed={stage.seed}",
    ]
    if stage.bf16:
        cmd.append("--bf16")
    if stage.fp16:
        cmd.append("--fp16")
    cmd.extend(stage.extra_args)
    return cmd


def build_stage_command(
    stage: StageConfig,
    base_model: str = "",
) -> Tuple[List[str], Dict[str, str], Dict[str, str], str]:
    """
    Build the training command, environment variables, extra files, and a description.

    Returns:
        (cmd_parts, env_vars, files_to_write, description)
    """
    output_dir = _normalize_path(stage.output_dir)
    base_model = base_model or stage.model or stage.base_checkpoint
    framework = stage.framework.lower()
    files_to_write: Dict[str, str] = {}
    env = _base_env(stage)

    if not base_model:
        raise ValueError(f"Stage '{stage.name}' requires a model or base_checkpoint")
    if not stage.data:
        raise ValueError(f"Stage '{stage.name}' requires a data path")

    if framework == Framework.UNSLOTH.value:
        if stage.type == StageType.SFT.value:
            script = _unsloth_sft_script(stage, base_model, output_dir)
        elif stage.type == StageType.DPO.value:
            script = _unsloth_dpo_script(stage, base_model, output_dir)
        elif stage.type == StageType.GRPO.value:
            script = _unsloth_grpo_script(stage, base_model, output_dir)
        else:
            raise ValueError(f"Unknown stage type: {stage.type}")
        script_path = _write_temp_script(script)
        cmd = [script_path]
        description = f"unsloth {stage.type} on {base_model} -> {output_dir}"

    elif framework == Framework.TRL.value:
        if stage.type == StageType.SFT.value:
            trl_cmd = _trl_cli_command(stage, base_model, output_dir, "sft")
            description = f"trl sft {base_model} -> {output_dir}"
        elif stage.type == StageType.DPO.value:
            # TRL dpo cli; map simpo/kto/orpo/cpo to trl subcommands
            algo = stage.algorithm.lower()
            if algo == "simpo":
                trl_cmd_name = "cpo"  # TRL CPO covers SimPO-style contrastive objectives
            elif algo == "kto":
                trl_cmd_name = "kto"
            elif algo == "orpo":
                trl_cmd_name = "orpo"
            elif algo == "cpo":
                trl_cmd_name = "cpo"
            else:
                trl_cmd_name = "dpo"
            trl_cmd = _trl_cli_command(stage, base_model, output_dir, trl_cmd_name)
            if trl_cmd_name == "dpo":
                trl_cmd.append(f"--beta={stage.beta}")
            description = f"trl {trl_cmd_name} {base_model} -> {output_dir}"
        elif stage.type == StageType.GRPO.value:
            script = _trl_grpo_script(stage, base_model, output_dir)
            script_path = _write_temp_script(script)
            cmd = [script_path]
            description = f"trl grpo {base_model} -> {output_dir}"
        else:
            raise ValueError(f"Unknown stage type: {stage.type}")

        if stage.type != StageType.GRPO.value:
            # Wrap the TRL console CLI so TrainingOrchestrator can launch it
            # with torchrun/deepspeed/accelerate as a single Python entrypoint.
            script_path = _wrap_command_in_script(trl_cmd)
            cmd = [script_path]

    elif framework == Framework.OPENRLHF.value:
        if stage.type != StageType.GRPO.value:
            raise ValueError("openrlhf framework is currently targeted at GRPO/RL stages")
        openrlhf_cmd, env = _openrlhf_grpo_command(stage, base_model, output_dir)
        script_path = _wrap_command_in_script(openrlhf_cmd, extra_env=env)
        cmd = [script_path]
        description = f"openrlhf grpo {base_model} -> {output_dir}"

    elif framework == Framework.AXOLOTL.value:
        yaml_content = _axolotl_yaml(stage, base_model, output_dir)
        config_name = "axolotl_config.yaml"
        axolotl_cmd = ["axolotl", "train", config_name]
        script_path = _wrap_command_in_script(
            axolotl_cmd,
            extra_files={config_name: yaml_content},
            config_file=config_name,
            config_index=2,
            guard_rank_zero=True,
        )
        cmd = [script_path]
        description = f"axolotl {stage.type} {base_model} -> {output_dir}"

    elif framework == Framework.LLAMA_FACTORY.value:
        yaml_content = _llama_factory_yaml(stage, base_model, output_dir)
        config_name = "llama_factory_config.yaml"
        factory_cmd = ["llamafactory-cli", "train", config_name]
        script_path = _wrap_command_in_script(
            factory_cmd,
            extra_files={config_name: yaml_content},
            config_file=config_name,
            config_index=2,
            guard_rank_zero=True,
        )
        cmd = [script_path]
        description = f"llama-factory {stage.type} {base_model} -> {output_dir}"

    elif framework == Framework.MS_SWIFT.value:
        swift_cmd = _ms_swift_command(stage, base_model, output_dir)
        script_path = _wrap_command_in_script(swift_cmd)
        cmd = [script_path]
        description = f"ms-swift {stage.type} {base_model} -> {output_dir}"

    else:
        raise ValueError(f"Unsupported framework: {framework}")

    # Ensure the launch command is a single executable / script path that the
    # TrainingOrchestrator can wrap with torchrun, deepspeed, or accelerate.
    # The orchestrator adds the launcher (torchrun/deepspeed) based on config.framework.
    if stage.deepspeed:
        # deepspeed is selected in training_pipeline.py via config.framework
        pass

    return cmd, env, files_to_write, description


def framework_help() -> str:
    return (
        "Supported training frameworks:\n"
        "  - unsloth:  fast LoRA SFT / DPT / GRPO (unsloth + TRL)\n"
        "  - trl:      HuggingFace TRL SFT/DPO/KTO/ORPO/CPO/GRPO\n"
        "  - openrlhf: Ray-based OpenRLHF GRPO/RLHF\n"
        "  - axolotl:  YAML-driven training\n"
        "  - llama-factory: LLaMA-Factory CLI\n"
        "  - ms-swift: ModelScope Swift CLI\n"
    )
