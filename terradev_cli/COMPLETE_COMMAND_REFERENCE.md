# Complete Terradev CLI Command Reference

**All commands and subcommands for Terradev CLI v6.0.1**

---

## Main Commands

### **provision**

```bash
terradev provision [OPTIONS]
```

### **status**

```bash
terradev status [OPTIONS]
```

---

### **agentic-serving** - `terradev agent agentic-serving`

```bash
terradev agent agentic-serving [OPTIONS] COMMAND [ARGS]...
```

### **configure** - `terradev agent agentic-serving configure`

```bash
terradev agent agentic-serving configure [OPTIONS]
```

### **helm-values** - `terradev agent agentic-serving helm-values`

```bash
terradev agent agentic-serving helm-values [OPTIONS]
```

### **k8s** - `terradev agent agentic-serving k8s`

```bash
terradev agent agentic-serving k8s [OPTIONS]
```

### **launch-args** - `terradev agent agentic-serving launch-args`

```bash
terradev agent agentic-serving launch-args [OPTIONS]
```

### **lmcache-env** - `terradev agent agentic-serving lmcache-env`

```bash
terradev agent agentic-serving lmcache-env [OPTIONS]
```

### **show-config** - `terradev agent agentic-serving show-config`

```bash
terradev agent agentic-serving show-config [OPTIONS]
```

### **cost** - `terradev agent cost`

```bash
terradev agent cost [OPTIONS]
```

### **deploy** - `terradev agent deploy`

```bash
terradev agent deploy [OPTIONS]
```

### **langchain** - `terradev agent langchain`

```bash
terradev agent langchain [OPTIONS] COMMAND [ARGS]...
```

### **create-langgraph** - `terradev agent langchain create-langgraph`

```bash
terradev agent langchain create-langgraph [OPTIONS] GRAPH_NAME
```

### **create-pipeline** - `terradev agent langchain create-pipeline`

```bash
terradev agent langchain create-pipeline [OPTIONS] PIPELINE_NAME
```

### **create-workflow** - `terradev agent langchain create-workflow`

```bash
terradev agent langchain create-workflow [OPTIONS] WORKFLOW_NAME
```

### **test** - `terradev agent langchain test`

```bash
terradev agent langchain test [OPTIONS]
```

### **langgraph** - `terradev agent langgraph`

```bash
terradev agent langgraph [OPTIONS] COMMAND [ARGS]...
```

### **create-workflow** - `terradev agent langgraph create-workflow`

```bash
terradev agent langgraph create-workflow [OPTIONS] WORKFLOW_NAME
```

### **deploy** - `terradev agent langgraph deploy`

```bash
terradev agent langgraph deploy [OPTIONS] WORKFLOW_NAME
```

### **status** - `terradev agent langgraph status`

```bash
terradev agent langgraph status [OPTIONS] WORKFLOW_ID
```

### **test** - `terradev agent langgraph test`

```bash
terradev agent langgraph test [OPTIONS]
```

### **letta** - `terradev agent letta`

```bash
terradev agent letta [OPTIONS] COMMAND [ARGS]...
```

### **chat** - `terradev agent letta chat`

```bash
terradev agent letta chat [OPTIONS]
```

### **create** - `terradev agent letta create`

```bash
terradev agent letta create [OPTIONS]
```

### **delete** - `terradev agent letta delete`

```bash
terradev agent letta delete [OPTIONS]
```

### **list** - `terradev agent letta list`

```bash
terradev agent letta list [OPTIONS]
```

### **remember** - `terradev agent letta remember`

```bash
terradev agent letta remember [OPTIONS]
```

### **status** - `terradev agent letta status`

```bash
terradev agent letta status [OPTIONS]
```

### **list** - `terradev agent list`

```bash
terradev agent list [OPTIONS]
```

### **plan** - `terradev agent plan`

```bash
terradev agent plan [OPTIONS]
```

### **scale** - `terradev agent scale`

```bash
terradev agent scale [OPTIONS]
```

### **skill** - `terradev agent skill`

```bash
terradev agent skill [OPTIONS] COMMAND [ARGS]...
```

### **attach** - `terradev agent skill attach`

```bash
terradev agent skill attach [OPTIONS]
```

### **init** - `terradev agent skill init`

```bash
terradev agent skill init [OPTIONS]
```

### **status** - `terradev agent status`

```bash
terradev agent status [OPTIONS]
```

### **teardown** - `terradev agent teardown`

```bash
terradev agent teardown [OPTIONS]
```

### **vector-db** - `terradev agent vector-db`

```bash
terradev agent vector-db [OPTIONS] COMMAND [ARGS]...
```

### **down** - `terradev agent vector-db down`

```bash
terradev agent vector-db down [OPTIONS]
```

### **up** - `terradev agent vector-db up`

```bash
terradev agent vector-db up [OPTIONS]
```

### **report** - `terradev canary report`

```bash
terradev canary report [OPTIONS]
```

### **tail** - `terradev canary tail`

```bash
terradev canary tail [OPTIONS]
```

### **model-details** - `terradev cost-scaler model-details`

```bash
terradev cost-scaler model-details [OPTIONS] MODEL_ID
```

### **start** - `terradev cost-scaler start`

```bash
terradev cost-scaler start [OPTIONS]
```

### **status** - `terradev cost-scaler status`

```bash
terradev cost-scaler status [OPTIONS]
```

### **crud** - `terradev database crud`

```bash
terradev database crud [OPTIONS]
```

### **down** - `terradev database down`

```bash
terradev database down [OPTIONS]
```

### **qdrant** - `terradev database qdrant`

```bash
terradev database qdrant [OPTIONS] COMMAND [ARGS]...
```

### **create-collection** - `terradev database qdrant create-collection`

```bash
terradev database qdrant create-collection [OPTIONS]
```

### **delete-collection** - `terradev database qdrant delete-collection`

```bash
terradev database qdrant delete-collection [OPTIONS]
```

### **scroll** - `terradev database qdrant scroll`

```bash
terradev database qdrant scroll [OPTIONS]
```

### **search** - `terradev database qdrant search`

```bash
terradev database qdrant search [OPTIONS]
```

### **upsert** - `terradev database qdrant upsert`

```bash
terradev database qdrant upsert [OPTIONS]
```

### **search** - `terradev database search`

```bash
terradev database search [OPTIONS]
```

### **sql** - `terradev database sql`

```bash
terradev database sql [OPTIONS]
```

### **up** - `terradev database up`

```bash
terradev database up [OPTIONS]
```

### **weaviate** - `terradev database weaviate`

```bash
terradev database weaviate [OPTIONS] COMMAND [ARGS]...
```

### **create-collection** - `terradev database weaviate create-collection`

```bash
terradev database weaviate create-collection [OPTIONS]
```

### **delete-collection** - `terradev database weaviate delete-collection`

```bash
terradev database weaviate delete-collection [OPTIONS]
```

### **hybrid-search** - `terradev database weaviate hybrid-search`

```bash
terradev database weaviate hybrid-search [OPTIONS]
```

### **insert** - `terradev database weaviate insert`

```bash
terradev database weaviate insert [OPTIONS]
```

### **list-collections** - `terradev database weaviate list-collections`

```bash
terradev database weaviate list-collections [OPTIONS]
```

### **query** - `terradev database weaviate query`

```bash
terradev database weaviate query [OPTIONS]
```

### **up** - `terradev database weaviate up`

```bash
terradev database weaviate up [OPTIONS]
```

### **approve** - `terradev environments approve`

```bash
terradev environments approve [OPTIONS] PROMOTION_ID
```

### **history** - `terradev environments history`

```bash
terradev environments history [OPTIONS]
```

### **list** - `terradev environments list`

```bash
terradev environments list [OPTIONS]
```

### **promote** - `terradev environments promote`

```bash
terradev environments promote [OPTIONS] ARTIFACT_NAME
```

### **compare** - `terradev eval compare`

```bash
terradev eval compare [OPTIONS] MODEL_A MODEL_B
```

### **evaluation** - `terradev eval evaluation`

```bash
terradev eval evaluation [OPTIONS]
```

### **baseten** - `terradev gateway baseten`

```bash
terradev gateway baseten [OPTIONS] COMMAND [ARGS]...
```

### **chat** - `terradev gateway baseten chat`

```bash
terradev gateway baseten chat [OPTIONS]
```

### **configure** - `terradev gateway baseten configure`

```bash
terradev gateway baseten configure [OPTIONS]
```

### **delete** - `terradev gateway baseten delete`

```bash
terradev gateway baseten delete [OPTIONS] ENDPOINT_ID
```

### **deploy** - `terradev gateway baseten deploy`

```bash
terradev gateway baseten deploy [OPTIONS]
```

### **list** - `terradev gateway baseten list`

```bash
terradev gateway baseten list [OPTIONS]
```

### **models** - `terradev gateway baseten models`

```bash
terradev gateway baseten models [OPTIONS]
```

### **status** - `terradev gateway baseten status`

```bash
terradev gateway baseten status [OPTIONS] ENDPOINT_ID
```

### **huggingface** - `terradev gateway huggingface`

```bash
terradev gateway huggingface [OPTIONS] COMMAND [ARGS]...
```

### **chat** - `terradev gateway huggingface chat`

```bash
terradev gateway huggingface chat [OPTIONS]
```

### **configure** - `terradev gateway huggingface configure`

```bash
terradev gateway huggingface configure [OPTIONS]
```

### **delete** - `terradev gateway huggingface delete`

```bash
terradev gateway huggingface delete [OPTIONS] ENDPOINT_ID
```

### **deploy** - `terradev gateway huggingface deploy`

```bash
terradev gateway huggingface deploy [OPTIONS]
```

### **list** - `terradev gateway huggingface list`

```bash
terradev gateway huggingface list [OPTIONS]
```

### **models** - `terradev gateway huggingface models`

```bash
terradev gateway huggingface models [OPTIONS]
```

### **status** - `terradev gateway huggingface status`

```bash
terradev gateway huggingface status [OPTIONS] ENDPOINT_ID
```

### **inferx** - `terradev gateway inferx`

```bash
terradev gateway inferx [OPTIONS] COMMAND [ARGS]...
```

### **chat** - `terradev gateway inferx chat`

```bash
terradev gateway inferx chat [OPTIONS]
```

### **configure** - `terradev gateway inferx configure`

```bash
terradev gateway inferx configure [OPTIONS]
```

### **delete** - `terradev gateway inferx delete`

```bash
terradev gateway inferx delete [OPTIONS] ENDPOINT_ID
```

### **deploy** - `terradev gateway inferx deploy`

```bash
terradev gateway inferx deploy [OPTIONS]
```

### **list** - `terradev gateway inferx list`

```bash
terradev gateway inferx list [OPTIONS]
```

### **models** - `terradev gateway inferx models`

```bash
terradev gateway inferx models [OPTIONS]
```

### **status** - `terradev gateway inferx status`

```bash
terradev gateway inferx status [OPTIONS] ENDPOINT_ID
```

### **serve** - `terradev gateway serve`

```bash
terradev gateway serve [OPTIONS]
```

### **siliconflow** - `terradev gateway siliconflow`

```bash
terradev gateway siliconflow [OPTIONS] COMMAND [ARGS]...
```

### **chat** - `terradev gateway siliconflow chat`

```bash
terradev gateway siliconflow chat [OPTIONS]
```

### **configure** - `terradev gateway siliconflow configure`

```bash
terradev gateway siliconflow configure [OPTIONS]
```

### **delete** - `terradev gateway siliconflow delete`

```bash
terradev gateway siliconflow delete [OPTIONS] ENDPOINT_ID
```

### **deploy** - `terradev gateway siliconflow deploy`

```bash
terradev gateway siliconflow deploy [OPTIONS]
```

### **list** - `terradev gateway siliconflow list`

```bash
terradev gateway siliconflow list [OPTIONS]
```

### **models** - `terradev gateway siliconflow models`

```bash
terradev gateway siliconflow models [OPTIONS]
```

### **status** - `terradev gateway siliconflow status`

```bash
terradev gateway siliconflow status [OPTIONS] ENDPOINT_ID
```

### **status** - `terradev gateway status`

```bash
terradev gateway status [OPTIONS]
```

### **bootstrap** - `terradev gitops bootstrap`

```bash
terradev gitops bootstrap [OPTIONS]
```

### **init** - `terradev gitops init`

```bash
terradev gitops init [OPTIONS]
```

### **sync** - `terradev gitops sync`

```bash
terradev gitops sync [OPTIONS]
```

### **validate** - `terradev gitops validate`

```bash
terradev gitops validate [OPTIONS]
```

### **deploy** - `terradev infer deploy`

```bash
terradev infer deploy [OPTIONS]
```

### **endpoint** - `terradev infer endpoint`

```bash
terradev infer endpoint [OPTIONS] MODEL_PATH
```

### **failover** - `terradev infer failover`

```bash
terradev infer failover [OPTIONS]
```

### **route** - `terradev infer route`

```bash
terradev infer route [OPTIONS]
```

### **status** - `terradev infer status`

```bash
terradev infer status [OPTIONS]
```

### **deploy** - `terradev inferx deploy`

```bash
terradev inferx deploy [OPTIONS]
```

### **inferx-configure** - `terradev inferx inferx-configure`

```bash
terradev inferx inferx-configure [OPTIONS]
```

### **inferx-delete** - `terradev inferx inferx-delete`

```bash
terradev inferx inferx-delete [OPTIONS]
```

### **inferx-optimize** - `terradev inferx inferx-optimize`

```bash
terradev inferx inferx-optimize [OPTIONS]
```

### **inferx-quote** - `terradev inferx inferx-quote`

```bash
terradev inferx inferx-quote [OPTIONS]
```

### **inferx-status** - `terradev inferx inferx-status`

```bash
terradev inferx inferx-status [OPTIONS]
```

### **list** - `terradev inferx list`

```bash
terradev inferx list [OPTIONS]
```

### **usage** - `terradev inferx usage`

```bash
terradev inferx usage [OPTIONS]
```

### **create** - `terradev k8s create`

```bash
terradev k8s create [OPTIONS] CLUSTER_NAME
```

### **destroy** - `terradev k8s destroy`

```bash
terradev k8s destroy [OPTIONS] CLUSTER_NAME
```

### **info** - `terradev k8s info`

```bash
terradev k8s info [OPTIONS] CLUSTER_NAME
```

### **list** - `terradev k8s list`

```bash
terradev k8s list [OPTIONS]
```

### **add-input** - `terradev lineage add-input`

```bash
terradev lineage add-input [OPTIONS] EXECUTION_ID
```

### **add-output** - `terradev lineage add-output`

```bash
terradev lineage add-output [OPTIONS] EXECUTION_ID
```

### **auto** - `terradev lineage auto`

```bash
terradev lineage auto [OPTIONS]
```

### **complete** - `terradev lineage complete`

```bash
terradev lineage complete [OPTIONS] EXECUTION_ID
```

### **diff** - `terradev lineage diff`

```bash
terradev lineage diff [OPTIONS] VERSION1 VERSION2
```

### **export** - `terradev lineage export`

```bash
terradev lineage export [OPTIONS]
```

### **graph** - `terradev lineage graph`

```bash
terradev lineage graph [OPTIONS] ARTIFACT_ID
```

### **production** - `terradev lineage production`

```bash
terradev lineage production [OPTIONS]
```

### **register** - `terradev lineage register`

```bash
terradev lineage register [OPTIONS]
```

### **show** - `terradev lineage show`

```bash
terradev lineage show [OPTIONS] MODEL_IDENTIFIER
```

### **trace** - `terradev lineage trace`

```bash
terradev lineage trace [OPTIONS]
```

### **pool** - `terradev local pool`

```bash
terradev local pool [OPTIONS]
```

### **register** - `terradev local register`

```bash
terradev local register [OPTIONS]
```

### **scan** - `terradev local scan`

```bash
terradev local scan [OPTIONS]
```

### **activate** - `terradev lora activate`

```bash
terradev lora activate [OPTIONS]
```

### **add** - `terradev lora add`

```bash
terradev lora add [OPTIONS]
```

### **cost-report** - `terradev lora cost-report`

```bash
terradev lora cost-report [OPTIONS]
```

### **drift-check** - `terradev lora drift-check`

```bash
terradev lora drift-check [OPTIONS]
```

### **list** - `terradev lora list`

```bash
terradev lora list [OPTIONS]
```

### **lorax** - `terradev lora lorax`

```bash
terradev lora lorax [OPTIONS] COMMAND [ARGS]...
```

### **deploy** - `terradev lora lorax deploy`

```bash
terradev lora lorax deploy [OPTIONS]
```

### **generate** - `terradev lora lorax generate`

```bash
terradev lora lorax generate [OPTIONS]
```

### **list-adapters** - `terradev lora lorax list-adapters`

```bash
terradev lora lorax list-adapters [OPTIONS]
```

### **load-adapter** - `terradev lora lorax load-adapter`

```bash
terradev lora lorax load-adapter [OPTIONS]
```

### **sync-registry** - `terradev lora lorax sync-registry`

```bash
terradev lora lorax sync-registry [OPTIONS]
```

### **test** - `terradev lora lorax test`

```bash
terradev lora lorax test [OPTIONS]
```

### **unload-adapter** - `terradev lora lorax unload-adapter`

```bash
terradev lora lorax unload-adapter [OPTIONS]
```

### **peft** - `terradev lora peft`

```bash
terradev lora peft [OPTIONS] COMMAND [ARGS]...
```

### **delete** - `terradev lora peft delete`

```bash
terradev lora peft delete [OPTIONS]
```

### **import** - `terradev lora peft import`

```bash
terradev lora peft import [OPTIONS]
```

### **list** - `terradev lora peft list`

```bash
terradev lora peft list [OPTIONS]
```

### **validate** - `terradev lora peft validate`

```bash
terradev lora peft validate [OPTIONS]
```

### **register** - `terradev lora register`

```bash
terradev lora register [OPTIONS]
```

### **remove** - `terradev lora remove`

```bash
terradev lora remove [OPTIONS]
```

### **rollback** - `terradev lora rollback`

```bash
terradev lora rollback [OPTIONS]
```

### **sync** - `terradev lora sync`

```bash
terradev lora sync [OPTIONS]
```

### **versions** - `terradev lora versions`

```bash
terradev lora versions [OPTIONS]
```

### **list-workloads** - `terradev migrate list-workloads`

```bash
terradev migrate list-workloads [OPTIONS]
```

### **migration** - `terradev migrate migration`

```bash
terradev migrate migration [OPTIONS]
```

### **deepeval** - `terradev ml deepeval`

```bash
terradev ml deepeval [OPTIONS] COMMAND [ARGS]...
```

### **evaluate** - `terradev ml deepeval evaluate`

```bash
terradev ml deepeval evaluate [OPTIONS]
```

### **init** - `terradev ml deepeval init`

```bash
terradev ml deepeval init [OPTIONS]
```

### **install** - `terradev ml deepeval install`

```bash
terradev ml deepeval install [OPTIONS]
```

### **metrics** - `terradev ml deepeval metrics`

```bash
terradev ml deepeval metrics [OPTIONS]
```

### **run** - `terradev ml deepeval run`

```bash
terradev ml deepeval run [OPTIONS]
```

### **dvc** - `terradev ml dvc`

```bash
terradev ml dvc [OPTIONS] COMMAND [ARGS]...
```

### **add-data** - `terradev ml dvc add-data`

```bash
terradev ml dvc add-data [OPTIONS] DATA_PATH
```

### **add-remote** - `terradev ml dvc add-remote`

```bash
terradev ml dvc add-remote [OPTIONS] REMOTE_SPEC
```

### **init** - `terradev ml dvc init`

```bash
terradev ml dvc init [OPTIONS]
```

### **pull** - `terradev ml dvc pull`

```bash
terradev ml dvc pull [OPTIONS]
```

### **push** - `terradev ml dvc push`

```bash
terradev ml dvc push [OPTIONS]
```

### **status** - `terradev ml dvc status`

```bash
terradev ml dvc status [OPTIONS]
```

### **test** - `terradev ml dvc test`

```bash
terradev ml dvc test [OPTIONS]
```

### **guardrails** - `terradev ml guardrails`

```bash
terradev ml guardrails [OPTIONS] COMMAND [ARGS]...
```

### **chat** - `terradev ml guardrails chat`

```bash
terradev ml guardrails chat [OPTIONS]
```

### **generate-config** - `terradev ml guardrails generate-config`

```bash
terradev ml guardrails generate-config [OPTIONS]
```

### **k8s** - `terradev ml guardrails k8s`

```bash
terradev ml guardrails k8s [OPTIONS]
```

### **test** - `terradev ml guardrails test`

```bash
terradev ml guardrails test [OPTIONS]
```

### **kserve** - `terradev ml kserve`

```bash
terradev ml kserve [OPTIONS] COMMAND [ARGS]...
```

### **test** - `terradev ml kserve test`

```bash
terradev ml kserve test [OPTIONS]
```

### **langfuse** - `terradev ml langfuse`

```bash
terradev ml langfuse [OPTIONS] COMMAND [ARGS]...
```

### **configure** - `terradev ml langfuse configure`

```bash
terradev ml langfuse configure [OPTIONS]
```

### **datasets** - `terradev ml langfuse datasets`

```bash
terradev ml langfuse datasets [OPTIONS]
```

### **export-training-data** - `terradev ml langfuse export-training-data`

```bash
terradev ml langfuse export-training-data [OPTIONS]
```

### **k8s** - `terradev ml langfuse k8s`

```bash
terradev ml langfuse k8s [OPTIONS]
```

### **otel-env** - `terradev ml langfuse otel-env`

```bash
terradev ml langfuse otel-env [OPTIONS]
```

### **quality** - `terradev ml langfuse quality`

```bash
terradev ml langfuse quality [OPTIONS]
```

### **score** - `terradev ml langfuse score`

```bash
terradev ml langfuse score [OPTIONS]
```

### **scores** - `terradev ml langfuse scores`

```bash
terradev ml langfuse scores [OPTIONS]
```

### **test** - `terradev ml langfuse test`

```bash
terradev ml langfuse test [OPTIONS]
```

### **trace** - `terradev ml langfuse trace`

```bash
terradev ml langfuse trace [OPTIONS] TRACE_ID
```

### **traces** - `terradev ml langfuse traces`

```bash
terradev ml langfuse traces [OPTIONS]
```

### **mlflow-legacy** - `terradev ml mlflow-legacy`

```bash
terradev ml mlflow-legacy [OPTIONS] COMMAND [ARGS]...
```

### **create-experiment** - `terradev ml mlflow-legacy create-experiment`

```bash
terradev ml mlflow-legacy create-experiment [OPTIONS] EXPERIMENT_NAME
```

### **export** - `terradev ml mlflow-legacy export`

```bash
terradev ml mlflow-legacy export [OPTIONS] EXPERIMENT_ID
```

### **list-experiments** - `terradev ml mlflow-legacy list-experiments`

```bash
terradev ml mlflow-legacy list-experiments [OPTIONS]
```

### **list-runs** - `terradev ml mlflow-legacy list-runs`

```bash
terradev ml mlflow-legacy list-runs [OPTIONS] EXPERIMENT_ID
```

### **test** - `terradev ml mlflow-legacy test`

```bash
terradev ml mlflow-legacy test [OPTIONS]
```

### **ollama** - `terradev ml ollama`

```bash
terradev ml ollama [OPTIONS] COMMAND [ARGS]...
```

### **chat** - `terradev ml ollama chat`

```bash
terradev ml ollama chat [OPTIONS] MODEL
```

### **generate** - `terradev ml ollama generate`

```bash
terradev ml ollama generate [OPTIONS] MODEL
```

### **info** - `terradev ml ollama info`

```bash
terradev ml ollama info [OPTIONS] MODEL
```

### **list** - `terradev ml ollama list`

```bash
terradev ml ollama list [OPTIONS]
```

### **ps** - `terradev ml ollama ps`

```bash
terradev ml ollama ps [OPTIONS]
```

### **pull** - `terradev ml ollama pull`

```bash
terradev ml ollama pull [OPTIONS] MODEL
```

### **phoenix** - `terradev ml phoenix`

```bash
terradev ml phoenix [OPTIONS] COMMAND [ARGS]...
```

### **k8s** - `terradev ml phoenix k8s`

```bash
terradev ml phoenix k8s [OPTIONS]
```

### **otel-env** - `terradev ml phoenix otel-env`

```bash
terradev ml phoenix otel-env [OPTIONS]
```

### **projects** - `terradev ml phoenix projects`

```bash
terradev ml phoenix projects [OPTIONS]
```

### **snippet** - `terradev ml phoenix snippet`

```bash
terradev ml phoenix snippet [OPTIONS]
```

### **spans** - `terradev ml phoenix spans`

```bash
terradev ml phoenix spans [OPTIONS]
```

### **test** - `terradev ml phoenix test`

```bash
terradev ml phoenix test [OPTIONS]
```

### **trace** - `terradev ml phoenix trace`

```bash
terradev ml phoenix trace [OPTIONS]
```

### **qdrant** - `terradev ml qdrant`

```bash
terradev ml qdrant [OPTIONS] COMMAND [ARGS]...
```

### **collections** - `terradev ml qdrant collections`

```bash
terradev ml qdrant collections [OPTIONS]
```

### **count** - `terradev ml qdrant count`

```bash
terradev ml qdrant count [OPTIONS]
```

### **create-collection** - `terradev ml qdrant create-collection`

```bash
terradev ml qdrant create-collection [OPTIONS]
```

### **info** - `terradev ml qdrant info`

```bash
terradev ml qdrant info [OPTIONS]
```

### **k8s** - `terradev ml qdrant k8s`

```bash
terradev ml qdrant k8s [OPTIONS]
```

### **test** - `terradev ml qdrant test`

```bash
terradev ml qdrant test [OPTIONS]
```

### **ray** - `terradev ml ray`

```bash
terradev ml ray [OPTIONS] COMMAND [ARGS]...
```

### **dashboard** - `terradev ml ray dashboard`

```bash
terradev ml ray dashboard [OPTIONS]
```

### **install** - `terradev ml ray install`

```bash
terradev ml ray install [OPTIONS]
```

### **list-nodes** - `terradev ml ray list-nodes`

```bash
terradev ml ray list-nodes [OPTIONS]
```

### **start** - `terradev ml ray start`

```bash
terradev ml ray start [OPTIONS]
```

### **status** - `terradev ml ray status`

```bash
terradev ml ray status [OPTIONS]
```

### **stop** - `terradev ml ray stop`

```bash
terradev ml ray stop [OPTIONS]
```

### **test** - `terradev ml ray test`

```bash
terradev ml ray test [OPTIONS]
```

### **sglang** - `terradev ml sglang`

```bash
terradev ml sglang [OPTIONS] COMMAND [ARGS]...
```

### **detect** - `terradev ml sglang detect`

```bash
terradev ml sglang detect [OPTIONS] MODEL_PATH
```

### **install** - `terradev ml sglang install`

```bash
terradev ml sglang install [OPTIONS]
```

### **router** - `terradev ml sglang router`

```bash
terradev ml sglang router [OPTIONS] MODEL_PATH
```

### **sglang-optimize** - `terradev ml sglang sglang-optimize`

```bash
terradev ml sglang sglang-optimize [OPTIONS] MODEL_PATH
```

### **start** - `terradev ml sglang start`

```bash
terradev ml sglang start [OPTIONS] MODEL_PATH
```

### **test** - `terradev ml sglang test`

```bash
terradev ml sglang test [OPTIONS]
```

### **vllm** - `terradev ml vllm`

```bash
terradev ml vllm [OPTIONS] COMMAND [ARGS]...
```

### **analyze** - `terradev ml vllm analyze`

```bash
terradev ml vllm analyze [OPTIONS]
```

### **auto-optimize** - `terradev ml vllm auto-optimize`

```bash
terradev ml vllm auto-optimize [OPTIONS]
```

### **benchmark** - `terradev ml vllm benchmark`

```bash
terradev ml vllm benchmark [OPTIONS]
```

### **import-adapter** - `terradev ml vllm import-adapter`

```bash
terradev ml vllm import-adapter [OPTIONS] ADAPTER_ID
```

### **import-model** - `terradev ml vllm import-model`

```bash
terradev ml vllm import-model [OPTIONS] MODEL_ID
```

### **lora** - `terradev ml vllm lora`

```bash
terradev ml vllm lora [OPTIONS] COMMAND [ARGS]...
```

### **link** - `terradev ml vllm lora link`

```bash
terradev ml vllm lora link [OPTIONS]
```

### **list** - `terradev ml vllm lora list`

```bash
terradev ml vllm lora list [OPTIONS]
```

### **load** - `terradev ml vllm lora load`

```bash
terradev ml vllm lora load [OPTIONS]
```

### **sync** - `terradev ml vllm lora sync`

```bash
terradev ml vllm lora sync [OPTIONS]
```

### **unload** - `terradev ml vllm lora unload`

```bash
terradev ml vllm lora unload [OPTIONS]
```

### **optimize** - `terradev ml vllm optimize`

```bash
terradev ml vllm optimize [OPTIONS]
```

### **wandb** - `terradev ml wandb`

```bash
terradev ml wandb [OPTIONS] COMMAND [ARGS]...
```

### **create-dashboard** - `terradev ml wandb create-dashboard`

```bash
terradev ml wandb create-dashboard [OPTIONS]
```

### **create-project** - `terradev ml wandb create-project`

```bash
terradev ml wandb create-project [OPTIONS] PROJECT_NAME
```

### **create-report** - `terradev ml wandb create-report`

```bash
terradev ml wandb create-report [OPTIONS]
```

### **dashboard-status** - `terradev ml wandb dashboard-status`

```bash
terradev ml wandb dashboard-status [OPTIONS]
```

### **list-projects** - `terradev ml wandb list-projects`

```bash
terradev ml wandb list-projects [OPTIONS]
```

### **list-runs** - `terradev ml wandb list-runs`

```bash
terradev ml wandb list-runs [OPTIONS]
```

### **setup-alerts** - `terradev ml wandb setup-alerts`

```bash
terradev ml wandb setup-alerts [OPTIONS]
```

### **test** - `terradev ml wandb test`

```bash
terradev ml wandb test [OPTIONS]
```

### **classify** - `terradev model-router classify`

```bash
terradev model-router classify [OPTIONS] TEXT
```

### **configure** - `terradev model-router configure`

```bash
terradev model-router configure [OPTIONS]
```

### **llmd-config** - `terradev model-router llmd-config`

```bash
terradev model-router llmd-config [OPTIONS]
```

### **stats** - `terradev model-router stats`

```bash
terradev model-router stats [OPTIONS]
```

### **test** - `terradev model-router test`

```bash
terradev model-router test [OPTIONS]
```

### **gateway** - `terradev observe gateway`

```bash
terradev observe gateway [OPTIONS] GATEWAY_ENDPOINT
```

### **status** - `terradev observe status`

```bash
terradev observe status [OPTIONS] TRACE_ID
```

### **evict** - `terradev orchestrator evict`

```bash
terradev orchestrator evict [OPTIONS] MODEL_ID
```

### **infer** - `terradev orchestrator infer`

```bash
terradev orchestrator infer [OPTIONS] MODEL_ID
```

### **load** - `terradev orchestrator load`

```bash
terradev orchestrator load [OPTIONS] MODEL_ID
```

### **register** - `terradev orchestrator register`

```bash
terradev orchestrator register [OPTIONS] MODEL_ID MODEL_PATH
```

### **start** - `terradev orchestrator start`

```bash
terradev orchestrator start [OPTIONS]
```

### **status** - `terradev orchestrator status`

```bash
terradev orchestrator status [OPTIONS]
```

### **export-example** - `terradev providers export-example`

```bash
terradev providers export-example [OPTIONS]
```

### **list-profiles** - `terradev providers list-profiles`

```bash
terradev providers list-profiles [OPTIONS]
```

### **load-profiles** - `terradev providers load-profiles`

```bash
terradev providers load-profiles [OPTIONS]
```

### **remove-profile** - `terradev providers remove-profile`

```bash
terradev providers remove-profile [OPTIONS] NAME
```

### **show-profile** - `terradev providers show-profile`

```bash
terradev providers show-profile [OPTIONS] NAME
```

### **start** - `terradev record start`

```bash
terradev record start [OPTIONS]
```

### **stop** - `terradev record stop`

```bash
terradev record stop [OPTIONS]
```

### **deploy** - `terradev retrain deploy`

```bash
terradev retrain deploy [OPTIONS]
```

### **detect** - `terradev retrain detect`

```bash
terradev retrain detect [OPTIONS]
```

### **drift** - `terradev retrain drift`

```bash
terradev retrain drift [OPTIONS]
```

### **history** - `terradev retrain history`

```bash
terradev retrain history [OPTIONS]
```

### **job** - `terradev schedule job`

```bash
terradev schedule job [OPTIONS] COMMAND GPU_TYPE
```

### **list** - `terradev schedule list`

```bash
terradev schedule list [OPTIONS]
```

### **windows** - `terradev schedule windows`

```bash
terradev schedule windows [OPTIONS]
```

### **configure** - `terradev sso configure`

```bash
terradev sso configure [OPTIONS]
```

### **status** - `terradev sso status`

```bash
terradev sso status [OPTIONS]
```

### **test** - `terradev sso test`

```bash
terradev sso test [OPTIONS]
```

### **resume** - `terradev train resume`

```bash
terradev train resume [OPTIONS]
```

### **start** - `terradev train start`

```bash
terradev train start [OPTIONS] [SCRIPT_ARGS]...
```

### **status** - `terradev train status`

```bash
terradev train status [OPTIONS]
```

### **stop** - `terradev train stop`

```bash
terradev train stop [OPTIONS]
```

### **unsloth** - `terradev train unsloth`

```bash
terradev train unsloth [OPTIONS] COMMAND [ARGS]...
```

### **run** - `terradev train unsloth run`

```bash
terradev train unsloth run [OPTIONS]
```

### **start** - `terradev train unsloth start`

```bash
terradev train unsloth start [OPTIONS]
```

### **stop** - `terradev train unsloth stop`

```bash
terradev train unsloth stop [OPTIONS]
```

### **create** - `terradev triggers create`

```bash
terradev triggers create [OPTIONS] NAME PIPELINE
```

### **disable** - `terradev triggers disable`

```bash
terradev triggers disable [OPTIONS] NAME
```

### **enable** - `terradev triggers enable`

```bash
terradev triggers enable [OPTIONS] NAME
```

### **fire** - `terradev triggers fire`

```bash
terradev triggers fire [OPTIONS] EVENT_TYPE
```

### **list** - `terradev triggers list`

```bash
terradev triggers list [OPTIONS]
```

### **env** - `terradev vault env`

```bash
terradev vault env [OPTIONS] PROVIDER
```

### **get** - `terradev vault get`

```bash
terradev vault get [OPTIONS] PROVIDER KEY
```

### **list** - `terradev vault list`

```bash
terradev vault list [OPTIONS]
```

### **remove** - `terradev vault remove`

```bash
terradev vault remove [OPTIONS] PROVIDER [KEY]
```

### **run** - `terradev vault run`

```bash
terradev vault run [OPTIONS] COMMAND...
```

### **set** - `terradev vault set`

```bash
terradev vault set [OPTIONS] PROVIDER KEY
```

### **sync** - `terradev vault sync`

```bash
terradev vault sync [OPTIONS]
```

### **verify** - `terradev vault verify`

```bash
terradev vault verify [OPTIONS]
```

### **register** - `terradev warm-pool register`

```bash
terradev warm-pool register [OPTIONS] MODEL_ID
```

### **start** - `terradev warm-pool start`

```bash
terradev warm-pool start [OPTIONS]
```

### **status** - `terradev warm-pool status`

```bash
terradev warm-pool status [OPTIONS]
```
