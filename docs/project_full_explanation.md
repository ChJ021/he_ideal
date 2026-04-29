# HETune-LLM 项目完整说明

本文档基于当前仓库源码、配置、测试、脚本和 native OpenFHE runner 整理，目标是说明 HETune-LLM 的完整架构、各功能模块的代码实现方式、主要运行流程、输出产物和扩展点。

## 1. 项目定位

HETune-LLM 是一个面向 CKKS 同态加密友好 Transformer 推理的实验框架。它不直接把 HuggingFace 模型全部改写为 HE 执行，而是先在 plaintext PyTorch 中模拟不同近似算子的精度影响，再结合静态或导入的 HE 成本模型，为每一层可替换算子选择合适的近似实现，最终输出 schedule、评估指标、HE 成本分析和可选 OpenFHE 部署结果。

当前 MVP 覆盖：

- HuggingFace sequence-classification 模型加载。
- GLUE/SST-2 数据加载、校准、验证。
- GELU、LayerNorm、Softmax 近似候选。
- 单算子替换敏感度分析。
- 静态 CKKS 成本和 OpenFHE/SEAL profile 成本导入。
- uniform baseline、additive greedy、validated greedy 调度。
- HE level/bootstrapping 可行性分析。
- LayerNorm 参数蒸馏。
- OpenFHE 外部 runner 部署编排。
- 报告、图表、manifest、metrics、schedule 等产物输出。

## 2. 总体架构

整体数据流如下：

```text
configs/*.yaml
  -> load_experiment_config
  -> HFSequenceClassifierAdapter 加载模型并发现 OperatorKey
  -> ApproximationRegistry 注册候选算子
  -> collect_operator_calibration_stats 收集 GELU/LayerNorm 输入统计
  -> SensitivityProfiler 做单点替换 profiling
  -> StaticCostModel 或 ProfiledHECostModel 估计 HE 成本
  -> Uniform/Base/Greedy/ValidatedGreedy/HEUniform 生成 schedule
  -> PlaintextEvaluator 验证 schedule 精度
  -> 可选 DistillationRunner 微调 LayerNorm 替换模块参数
  -> HEAnalysisRunner 做 profile coverage、level、bootstrap 分析
  -> HEDeploymentRunner 调 native OpenFHE runner
  -> outputs/runs/<experiment_id>/ 写入所有产物
```

项目的核心解耦层是 `SchedulePlan`。模型适配器只负责发现稳定的 `OperatorKey` 和按 schedule 替换模块；调度器只输出 `OperatorKey -> candidate_id`；算子注册表负责把 `candidate_id` 映射到 plaintext 实现和 HE 成本元数据。这样每层替换选择不会硬编码进模型代码。

## 3. 目录结构

```text
src/hetune/
  core/          稳定 ID、成本向量、schedule、sensitivity、路径集合、序列化。
  operators/     GELU、LayerNorm、Softmax 近似候选和 registry。
  models/        HuggingFace adapter、BERT/DistilBERT attention wrapper。
  profiling/     accuracy/KL/flip 指标、敏感度 profiling、校准统计 hooks。
  cost/          静态成本模型、profiled HE 成本导入。
  scheduling/    uniform/base/greedy/validated greedy/HE-aware 策略和 bootstrap planner。
  execution/     plaintext PyTorch schedule evaluator。
  experiments/   配置加载、数据加载、pipeline orchestration、报告、图表、HE 分析、蒸馏。
  deployment/    HE 部署配置、后端协议、OpenFHE 外部 runner 编排、forward artifact 导出。
  benchmarking/  SEAL profile schema 和 deterministic benchmark profile 生成。
  security/      schedule input-independent 基础校验。
  utils/         project root 和路径解析。

configs/
  experiments/       实验入口配置。
  models/            HF model id/name。
  datasets/          GLUE/SST-2 配置。
  approximations/    启用的 candidate_id 列表。
  ckks/              CKKS 参数、profile 路径、bootstrap 成本。
  deployment/        deploy-he 配置。
  he_backend/        OpenFHE 本地路径和 native runner 参数。

native/openfhe_runner/
  src/main.cpp       OpenFHE C++ runner，支持 schedule workload 和 DistilBERT forward 两种模式。

tests/
  unit/              核心类型、候选、调度、HE 分析、部署、蒸馏等单元测试。
  integration/       不依赖 HuggingFace 下载的离线调度链路测试。
```

## 4. 入口命令

CLI 在 `src/hetune/cli.py` 中定义，`pyproject.toml` 暴露脚本：

```text
hetune = "hetune.cli:app"
```

主要命令：

- `hetune run-activate --config ...`：只优化 GELU 和 LayerNorm。
- `hetune run-softmax --config ...`：只优化 attention Softmax。
- `hetune run-all --config ...`：联合优化 GELU、LayerNorm、Softmax。
- `hetune run-all-he --config ...`：先跑 all-nonlinear 搜索，再跑 HE profile 分析。
- `hetune run-he --config ...`：读取已有 schedules，只做 HE 成本、coverage、bootstrap 分析。
- `hetune bench-seal-profile --ckks-config ... --output ...`：生成标准 SEAL profile CSV 和 metadata。
- `hetune deploy-he --config ...`：按部署配置调用 OpenFHE 外部 runner。
- `hetune profile/tune/evaluate/distill --config ...`：分别执行 pipeline 的单个阶段。

`main.py` 和 `src/hetune/__main__.py` 都只是导入 `hetune.cli.app` 并执行。

## 5. 配置系统

`src/hetune/experiments/config.py` 的 `load_experiment_config()` 读取实验 YAML，并以实验 YAML 所在目录作为相对路径基准加载：

- `model_config`
- `dataset_config`
- `approximation_config`
- `ckks_config`

读取结果封装为 `LoadedExperimentConfig`，包含 project root、experiment、model、dataset、approximations、ckks 五类配置。

典型实验配置字段：

- `experiment_id`：输出目录名。
- `operator_scope` / `operator_types`：决定优化哪些算子。
- `device`、`sequence_length`、`batch_size`、`calibration_size`、`validation_size`。
- `accuracy_tolerance` 和 `scheduler.*`：validated greedy 的阈值与约束。
- `cost_weights`：`CostVector.weighted()` 使用的成本权重。
- `distillation.*`：是否启用 LayerNorm 参数蒸馏。

scope 默认映射在 `ExperimentRunner._operator_types_for_scope()` 和 `DistillationRunner` 中保持一致：

- `activation_norm` -> `["gelu", "layernorm"]`
- `softmax_only` -> `["softmax"]`
- `all_nonlinear` -> `["gelu", "layernorm", "softmax"]`

## 6. 核心数据类型

### 6.1 OperatorKey

`src/hetune/core/ids.py`

`OperatorKey` 是可替换算子的稳定标识：

- `model_id`
- `layer_index`
- `operator_type`
- `name`
- `path`

其 `id` 由上述字段拼成：

```text
<model_id>.layer<layer_index>.<operator_type>.<name>
```

模型发现、敏感度记录、schedule、报告都依赖这个稳定 ID。

### 6.2 CostVector

`src/hetune/core/types.py`

`CostVector` 表示 HE 成本：

- `latency_ms`
- `rotations`
- `ct_ct_mults`
- `ct_pt_mults`
- `rescale_count`
- `relin_count`
- `depth`
- `bootstrap_count`
- `memory_mb`

它支持：

- `weighted(weights)`：按权重折算总成本。
- `__add__()`：汇总 schedule 总成本。
- `to_dict()` / `from_dict()`：CSV/YAML/JSON 输出输入。

### 6.3 ApproximationSpec

`ApproximationSpec` 描述一个候选近似实现：

- `operator_type`
- `candidate_id`
- `approximation_family`
- `quality_rank`
- `degree`
- `valid_input_range`
- `depth`
- `rotation_requirement`
- `supports_plaintext_simulation`
- `supports_ckks_backend`
- `expected_accuracy_risk`
- `implementation_backend`
- `cost_hint`

调度器按 `quality_rank` 从高到低排序。`.base` 候选是 plaintext reference，默认不参与自动搜索。

### 6.4 ScheduleEntry 和 SchedulePlan

`ScheduleEntry` 是一个算子的选择：

- `operator_key`
- `candidate_id`
- `ckks_param_id`
- `scale_id`
- `level_budget`
- `bootstrap_policy`
- `layout_id`

`SchedulePlan` 包含：

- `metadata`
- `entries`
- `constraints`

所有 schedule YAML 都通过 `SchedulePlan.to_dict()` 和 `SchedulePlan.from_dict()` 读写。

### 6.5 SensitivityRecord

`SensitivityRecord` 记录单算子替换结果：

- baseline/candidate accuracy
- accuracy drop
- logit KL
- label flip rate
- hidden L2、attention KL 预留字段

`sensitivity_score` 是上述风险项的加和，用于 greedy 排序。

### 6.6 ExperimentPaths

`ExperimentPaths` 统一定义输出目录：

```text
outputs/runs/<experiment_id>/
  configs/
  profiles/
  schedules/
  evaluations/
  he_analysis/
  he_deployment/
  distillation/
  figures/
  reports/
  logs/
```

`ensure()` 会创建所有目录。

## 7. 算子候选实现

### 7.1 Registry

`src/hetune/operators/registry.py`

`ApproximationRegistry` 保存 `candidate_id -> provider`，并按 `operator_type` 分组。`query()` 默认排除 `.base` 候选，也可以按 enabled ids 或 `supports_ckks_backend` 过滤。

`build_default_registry()` 注册：

- `gelu_providers()`
- `layernorm_providers()`
- `softmax_providers()`

当 `ckks_only=True` 时，只保留支持 CKKS 后端的候选和 `.base` reference。

### 7.2 GELU

`src/hetune/operators/gelu.py`

候选：

- `gelu.base`：调用 `torch.nn.functional.gelu`。
- `gelu.exact.high.v1`：校准区间上的 Chebyshev degree 13 近似，负尾部置 0，正尾部近似 identity。
- `gelu.chebyshev.degree11.v1`：与 high 相同的校准 Chebyshev 构建，degree 11。
- `gelu.chebyshev.degree9.v1`：与 high 相同的校准 Chebyshev 构建，degree 9。
- `gelu.chebyshev.degree5.v1`：与 high 相同的校准 Chebyshev 构建，degree 5。
- 旧版 `gelu.poly.degree*` 手写固定多项式已移除；GELU 搜索空间只保留校准 Chebyshev 多项式系列。

高质量候选会读取 `context["calibration_stats"]`，用 `abs_p99` 或 `abs_p95` 决定 Chebyshev 拟合 scale。系数由 `_gelu_coefficients()` 动态用 NumPy 拟合并 LRU cache。

### 7.3 LayerNorm

`src/hetune/operators/layernorm.py`

LayerNorm 候选替换的是模块，不是函数。`LayerNormProvider.build_layernorm_module()` 返回新的 torch module。

候选：

- `layernorm.base`：深拷贝原始 LayerNorm。
- `layernorm.exact.high.v1`：对 `rsqrt(var + eps)` 做校准 Chebyshev degree 9 近似。
- `layernorm.newton.low_iter.v1`：低迭代近似 `1 / (1 + 0.5 * (var - 1))`。
- `layernorm.centered.mid_cost.v1`：只中心化再 affine。
- `layernorm.affine.low_cost.v1`：跳过均值方差，只做 `x * weight + bias`。

`ApproxLayerNorm` 内部创建实际 `torch.nn.Module`，复制原始 weight/bias。高质量候选通过校准统计中的 `var_p01/var_p95/var_p99/var_mean` 决定 rsqrt 近似区间。

### 7.4 Softmax

`src/hetune/operators/softmax.py`

Softmax 候选替换 attention 模块里的 softmax，不直接替换 `torch.softmax` 全局函数。

候选：

- `softmax.base`：`torch.softmax(scores, dim=-1)`。
- `softmax.exact.high.v1`：shift 到 `[-8, 0]` 后用 degree 4 多项式近似 exp，再归一化。
- `softmax.clipped.stable.v1`：clipped exact softmax，不支持 CKKS backend。
- `softmax.poly_exp.degree2.v1`：degree 2 exp surrogate。
- `softmax.power.degree2.v1`：power attention 风格低成本候选。

所有非 base 实现都会处理 attention mask，并保证输出非负且最后一维归一化。

## 8. HuggingFace 模型适配

### 8.1 加载与发现

`src/hetune/models/hf_adapter.py`

`HFSequenceClassifierAdapter.load()`：

1. 检查 CUDA 可用性，不可用则回落 CPU。
2. 用 `AutoTokenizer.from_pretrained()` 加载 tokenizer。
3. 用 `AutoModelForSequenceClassification.from_pretrained()` 加载模型。
4. 移动到 device，设为 eval。
5. 调 `discover_operators()`。

`discover_operators()` 支持：

- BERT/RoBERTa/ALBERT-like：发现每层 GELU、attention softmax、attention output LayerNorm、FFN output LayerNorm。
- DistilBERT：发现每层 GELU、attention softmax、`sa_layer_norm`、`output_layer_norm`。
- 其他模型：退化为扫描 `torch.nn.LayerNorm`。

### 8.2 应用 schedule

`apply_schedule()` 的流程：

1. 先 `restore_original()`，清掉上次替换。
2. 遍历 schedule entries。
3. `.base` candidate 直接跳过，相当于保留原模块。
4. 每个算子构造 context，包含 operator id/path/type 和校准统计。
5. 首次替换前深拷贝原始对象到 `_originals`。
6. 按类型替换：
   - GELU：替换函数或包装成 module。
   - LayerNorm：用 provider 构造替换 module。
   - Softmax：用 attention wrapper 包住原 attention module。

评估结束后 `PlaintextEvaluator` 会调用 `restore_original()`，避免 schedule 之间互相污染。

### 8.3 参数覆盖

`apply_parameter_overrides()` 用于蒸馏后的 LayerNorm 参数覆盖。payload 每项包含：

- `operator_id`
- `operator_path`
- `candidate_id`
- `parameter_name`
- `tensor`

函数会找到 schedule 对应模块，并把 tensor 拷贝到该模块的 `torch.nn.Parameter`。

### 8.4 Attention wrappers

`src/hetune/models/attention_wrappers.py`

`build_attention_wrapper()` 支持：

- DistilBERT `MultiHeadSelfAttention` / `DistilBertSdpaAttention`
- BERT `BertSelfAttention` / `BertSdpaSelfAttention`

wrapper 复制原始 attention forward 的主要逻辑，只把 softmax 一步替换为 provider 给出的 `replacement_softmax`。对 unsupported 模块会抛 `TypeError`，避免静默替换错误。

## 9. Profiling 和评估

### 9.1 PlaintextEvaluator

`src/hetune/execution/evaluator.py`

`PlaintextEvaluator.run()`：

1. 如果传入 schedule，调用 adapter 应用 schedule。
2. 如果传入 parameter overrides，也一并应用。
3. 用 DataLoader 遍历 dataset。
4. 移除 labels，把 batch 送到 device。
5. 收集 logits 和 labels。
6. 恢复原模型。
7. 返回 `EvaluationResult(logits, labels, accuracy)`。

它是 profiling、validated greedy、evaluate、distillation pre-result 的共同执行器。

### 9.2 metrics

`src/hetune/profiling/metrics.py`

实现：

- `accuracy(logits, labels)`
- `softmax(logits)`
- `logit_kl(reference_logits, candidate_logits)`
- `label_flip_rate(reference_logits, candidate_logits)`

validated greedy 同时限制 accuracy drop、logit KL 和 label flip rate。

### 9.3 校准统计

`src/hetune/profiling/calibration.py`

`collect_operator_calibration_stats()` 给 GELU 和 LayerNorm 注册 hooks：

- 对 module 目标用 forward pre-hook 捕获输入。
- 对函数式 GELU 的 fallback path，改在上游 dense/lin1 输出 hook 捕获。

统计字段包括：

- mean/std/min/max
- abs_p95/abs_p99
- rms_mean
- LayerNorm variance 的 min/p01/p05/mean/p95/p99

结果写入 `profiles/operator_calibration_stats.csv`，并通过 `adapter.set_calibration_stats()` 供高质量 GELU/LayerNorm 近似读取。

### 9.4 敏感度 profiling

`src/hetune/profiling/profiler.py`

`SensitivityProfiler.profile_all()`：

1. 先跑 baseline schedule-less 原模型，得到 baseline logits/accuracy。
2. 对每个 operator 和 registry 中对应类型的每个非 base candidate：
   - 构造只替换该 operator 的 `SchedulePlan`。
   - evaluator 跑该单点替换。
   - 记录 accuracy drop、logit KL、label flip rate。
3. 保存为 `profiles/sensitivity_matrix.csv`。

## 10. 成本模型

### 10.1 StaticCostModel

`src/hetune/cost/static.py`

静态成本模型直接读取 provider spec 中的 `cost_hint`，可：

- `estimate(operator_key, candidate_id)`
- `estimate_schedule(schedule)`
- `weighted_cost(operator_key, candidate_id)`
- `export_candidate_costs(output_path)`

它用于 plaintext tuning 或没有 profile 数据时的 fallback。

### 10.2 ProfiledHECostModel

`src/hetune/cost/profiled.py`

profiled 成本模型用于 HE-aware tuning 和 `run-he`。它读取 CSV 或 JSON profile，并按以下条件过滤：

- `ckks_param_id`
- `backend_id`

如果 candidate 在 profile 中有记录，优先使用 profile 成本；否则回落到 `StaticCostModel`。

严格模式字段：

- `profile_required`
- `profile_min_coverage`

当 strict profile 开启时，以下情况会抛 `ProfileValidationError` 或标记 strict check 失败：

- 没有配置 profile path。
- profile 文件不存在。
- profile 中没有匹配当前 backend/ckks 的行。
- 最终非 base schedule 使用了缺失 profile 的 candidate。
- coverage rate 低于最小阈值。

`coverage_for_schedule()` 会输出：

- non-base entries 数量。
- profile/static fallback entries 数量。
- 使用过且有/缺失 profile 的 candidate ids。
- strict check 是否通过及原因。

### 10.3 SEAL profile 生成

`src/hetune/benchmarking/seal.py`

当前 `bench-seal-profile` 并不调用真实 SEAL 执行，而是用 `CostHintSealBenchmarkBackend` 基于 registry cost hints 生成 deterministic CSV。它定义了标准 profile schema：

```text
backend_id, ckks_param_id, candidate_id,
latency_ms, rotations, ct_ct_mults, ct_pt_mults,
rescale_count, relin_count, depth, bootstrap_count, memory_mb
```

同时生成 metadata JSON，记录 SEAL 路径、版本、CKKS 参数、repetitions、warmups。

## 11. 调度策略

### 11.1 UniformPolicy

`UniformPolicy` 对所有 operator 选择同一质量档：

- `high`：最高 `quality_rank` 的非 base candidate。
- `mid`：排序中间 candidate。
- `low`：最低 `quality_rank` candidate。

输出 `uniform_high.yaml`、`uniform_mid.yaml`、`uniform_low.yaml`。

### 11.2 BasePolicy

`BasePolicy` 给每类 operator 选择 `<operator_type>.base`，用于 plaintext reference，不是 HE 部署 schedule。

输出 `base_reference.yaml`。

### 11.3 HEUniformPolicy

`HEUniformPolicy` 只考虑 `supports_ckks_backend=True` 的候选，并用 `level_cost(cost) <= available_levels` 过滤不可行候选。质量档选择逻辑与 UniformPolicy 一致。

当高质量候选单算子 level 超预算时，它会自动降到当前 CKKS level 能容纳的最高质量候选。

### 11.4 GreedyDowngradePolicy

这是 additive greedy：

1. 从 uniform high 或传入的 initial schedule 开始。
2. 对每个 operator 的低质量候选计算成本节省。
3. 用 `saving / sensitivity_penalty` 排序。
4. 按排序尝试 downgrade。
5. 用单点 accuracy drop 累加估算总 drop，不重新跑组合评估。
6. 不超过 `max_accuracy_drop` 就接受。

输出 `hetune_additive_greedy.yaml`。它主要作为对照，因为单点误差会相互作用，风险比 validated greedy 更高。

### 11.5 ValidatedGreedyDowngradePolicy

这是默认主策略：

1. 从 high 或 HE-aware high schedule 开始。
2. 用 base schedule 评估原始 reference。
3. 评估 high schedule。
4. 根据单点 sensitivity 和成本节省排序候选 downgrade。
5. 每个候选先做 precheck：
   - protected operator type。
   - 是否真的是 downgrade。
   - 是否低于 operator type 最低质量 rank。
   - 是否超过每层 downgrade 数。
6. 构造 trial schedule，并可选调用 HE schedule constraint checker。
7. 真正跑一次组合 schedule 评估。
8. 同时检查：
   - `max_accuracy_drop`
   - `max_logit_kl`
   - `max_label_flip_rate`
9. 通过才接受，否则记录拒绝原因。

输出：

- `hetune_generated.yaml`
- `validated_greedy_decisions.csv`

decision log 包含每个候选的 from/to candidate、accepted、reason、benefit、cost saving、single accuracy drop、组合 accuracy/KL/flip 等。

### 11.6 HE planner

`src/hetune/scheduling/he_planner.py`

HE planner 不改变 candidate 选择本身，而是分析一个 schedule 是否能放入 CKKS level budget，并标注 bootstrap：

- `available_levels()`：从配置读取 `available_levels`，否则用 modulus chain 长度估算。
- `level_cost()`：取 `max(depth, rescale_count)`。
- `bootstrap_cost()`：读取配置中的 bootstrap cost。
- `analyze_schedule_feasibility()`：
  - 给每个 entry 估成本和 level cost。
  - 调 `build_bootstrap_plan_with_annotations()`。
  - 标注 `level_budget`、`bootstrap_policy`、`layout_id=ckks_profiled`。
  - 汇总总成本，包含估计 bootstrap cost。

若单个 operator level cost 超过 `available_levels`，schedule infeasible。若累计 level 超预算，且支持 bootstrapping，则在对应 operator 前插入 `bootstrap_before`；若不支持，标记 infeasible。

## 12. 实验主流程

`src/hetune/experiments/runner.py` 的 `ExperimentRunner` 是主编排器。

### 12.1 初始化

`__init__()`：

1. 加载实验配置。
2. 决定 operator scope/types。
3. 构造 `ExperimentPaths`。
4. 创建输出目录。
5. 写配置快照到 `configs/`。
6. 写初始 manifest。

### 12.2 `run()`

完整 pipeline：

```text
profile()
  -> tune()
  -> _run_distillation_if_enabled()
  -> evaluate()
  -> write_manifest()
  -> write_artifacts_index()
```

### 12.3 `profile()`

1. `_build_runtime(calibration_split)` 加载模型、registry、evaluator。
2. 加载 calibration dataset。
3. `_ensure_calibration_stats()` 收集或读取校准统计。
4. `SensitivityProfiler.profile_all()` 跑单点替换。
5. 写 `sensitivity_matrix.csv`。
6. 导出候选成本表到 `outputs/cost_tables/<ckks_param_id>/candidate_costs.csv`。
7. 写 sensitivity heatmap。

### 12.4 `tune()`

1. 加载 calibration runtime 和 dataset。
2. 确保校准统计存在。
3. 确保 sensitivity profile 存在，不存在则先 profile。
4. 加载 sensitivity records。
5. 生成 constraints。
6. 构造成本模型。
7. 写 base reference schedule。
8. 如果 HE-aware：
   - 使用 `ProfiledHECostModel`。
   - 使用 `HEUniformPolicy` 生成 high/mid/low。
   - 对 schedule 做 HE annotation。
   - validated greedy 中启用 HE constraint checker。
9. 写 uniform schedules。
10. 写 additive greedy schedule。
11. 可选写 combination diagnostics。
12. 跑 validated greedy。
13. 用 `SecurityValidator` 检查 schedule。
14. 写最终 `hetune_generated.yaml`。
15. 写 selection summary 和 softmax selection。

### 12.5 `evaluate()`

1. 构建 validation runtime。
2. 读取校准统计。
3. 加载 validation dataset。
4. 加载最终 schedule，不存在则先 tune。
5. 评估 base、hetune generated、可选 distilled、uniform low/mid/high。
6. 对每个 schedule 估成本；HE-aware 时使用 HE feasibility 结果总成本。
7. 写 `evaluations/metrics.csv`。
8. 写 `reports/report.md`。
9. 写 artifacts index。

metrics 每行包括 schedule、accuracy、profile coverage 信息和成本向量。

## 13. HE 分析流程

`src/hetune/experiments/he_analysis.py`

`HEAnalysisRunner` 与主 runner 不同，它不加载模型和数据，只读取已有 schedule YAML，因此可在调度完成后离线运行。

流程：

1. 加载 experiment config。
2. build registry。
3. 如果有 `hetune_generated.yaml`，确保生成 `base_reference.yaml`。
4. 创建 `ProfiledHECostModel`。
5. 搜索 schedule：
   - base
   - hetune_generated
   - uniform_low
   - uniform_mid
   - uniform_high
6. 对每个 schedule 调 `analyze_schedule_feasibility()`。
7. 汇总：
   - `he_metrics.csv`
   - `he_cost_breakdown.csv`
   - `profile_coverage.csv`
   - `bootstrap_plan.csv`
8. 写 `figures/he_cost_breakdown.png`。
9. 写 `reports/he_report.md`。
10. 更新 manifest 和 artifacts index。
11. 若 strict profile 开启且最终 schedule 不通过，抛 `ProfileValidationError`。

## 14. 蒸馏流程

`src/hetune/experiments/distillation.py`

蒸馏用于在 schedule 已经替换 LayerNorm 后，微调替换模块的 `weight` 和 `bias`，降低近似带来的精度损失。

`DistillationRunner.run()`：

1. 检查 `distillation.enabled`。
2. 读取指定 schedule，默认 `hetune_generated`。
3. 构建 teacher adapter 和 student adapter。
4. 加载 train/validation dataset。
5. 先用 schedule 评估 student 的 pre-distill accuracy。
6. 给 student 应用 schedule，teacher 保持原始模型。
7. 冻结 student 全部参数，只放开非 base LayerNorm 替换模块的 weight/bias。
8. 注册 hidden hooks，用于对齐指定 target LayerNorm 输出。
9. 训练若干 epoch，loss 为：
   - KL(student logits / temperature, teacher logits / temperature)
   - CE(student logits, labels)
   - hidden MSE(student hidden, teacher hidden)
10. 用 validation accuracy 选择 best payload。
11. 保存：
    - `distillation/summary.csv`
    - `distillation/overrides.pt`
    - `distillation/overrides_summary.csv`
    - `distillation/report.md`

`ExperimentRunner.evaluate()` 会检测 `overrides.pt`，若存在则额外评估 `hetune_generated_distilled`。

## 15. 部署流程

### 15.1 Python 编排

`src/hetune/deployment/`

部署入口是 `HEDeploymentRunner`：

1. `load_deployment_config()` 加载 deployment YAML。若传入的是 experiment YAML，则生成默认 deployment 配置。
2. 加载 backend YAML，例如 `configs/he_backend/openfhe_local.yaml`。
3. 构造 `OpenFHEExternalBackend`。
4. 解析 cases，默认：
   - `high` -> `uniform_high`
   - `pre_distill` -> `hetune_generated`
   - `post_distill` -> `hetune_generated` + `distillation/overrides.pt`
5. 检查 backend availability。
6. 对每个 case 构造 `DeploymentCaseRequest`。
7. 调后端 `run_case()`。
8. 写：
   - `he_deployment/metadata.json`
   - `he_deployment/<case>/request.json`
   - `he_deployment/<case>/result.json`
   - `he_deployment/<case>/metrics.csv`
   - `he_deployment/<case>/latency.csv`
   - `he_deployment/comparison.csv`
   - `he_deployment/deployment_report.md`

当 runner mode 是 `openfhe_distilbert_forward` 时，`_build_request()` 会先调用 `export_distilbert_forward_artifact()`。

### 15.2 Forward artifact

`src/hetune/deployment/forward_artifact.py`

该导出器为 native runner 准备 DistilBERT forward 所需的二进制 blob：

- 输入 embeddings。
- attention mask。
- labels。
- 每层 q/k/v/out linear 的 weight/bias。
- 每层 LayerNorm、FFN lin1/lin2 的 weight/bias。
- pre_classifier 和 classifier 的 weight/bias。
- schedule entries 和 CKKS public config。

manifest 写为 `manifest.json`，blob 写为 `.f32` 或 `.i32` 二进制文件。

隐私边界当前定义为 `client_embedding`：导出的是 embedding 后的明文张量，native runner 本地加密它们。生产场景可把这些文件替换为客户端产生的 ciphertext。

### 15.3 OpenFHEExternalBackend

`src/hetune/deployment/backend.py`

该后端只是 Python 到 native runner 的桥：

- 检查 OpenFHE root/install/config cmake/native runner 是否存在且可执行。
- 将 request 写成 JSON。
- 执行：

```text
<runner_path> --request request.json --output result.json
```

- 设置 `LD_LIBRARY_PATH=<openfhe_install>/lib`。
- 解析 native runner 输出的 result JSON 为 `DeploymentCaseResult`。

如果 `fail_on_plaintext_accuracy_fallback=true`，Python 层要求 native result 的 `accuracy_source` 必须是 `native_decrypted_logits`。

## 16. Native OpenFHE runner

`native/openfhe_runner/src/main.cpp`

runner 支持两种模式。

### 16.1 `openfhe_schedule_workload`

这是成本 workload 模式：

1. 读取 request。
2. 尝试从 run 目录的 `he_analysis/he_metrics.csv` 读取 schedule 成本计数。
3. 如果没有 metrics，则用 regex 从 schedule YAML 的 `candidate_id` 粗略统计成本。
4. 创建小型 CKKS context。
5. 按 cost counts 执行若干 EvalMult、EvalMult plaintext、EvalAtIndex。
6. 输出 latency 和成本 counts。
7. accuracy 为空，metadata 表示可用 plaintext metrics fallback。

这个模式验证 OpenFHE 执行环境和基本 ciphertext op latency，不执行完整 Transformer。

### 16.2 `openfhe_distilbert_forward`

这是 DistilBERT encrypted tensor forward 模式：

1. 读取 forward manifest 和所有 blobs。
2. 创建 `HEEngine`，初始化 CKKS context、keys、rotation keys。
3. 对每个样本：
   - 加密每个 token 的 embedding。
   - 每层执行 attention：
     - q/k/v linear，可选 fuse QKV。
     - BSGS/diagonal linear kernel。
     - 多头 dot product。
     - polynomial exp/softmax surrogate。
     - out linear。
   - residual + LayerNorm 近似。
   - FFN lin1。
   - GELU 多项式。
   - FFN lin2。
   - residual + LayerNorm 近似。
   - CLS token pre_classifier、GELU、classifier。
   - decrypt logits。
4. 写 predictions CSV 和 logits CSV。
5. 用 labels 计算 native decrypted logits accuracy。
6. 输出总 latency、p50、p95、stage timing、rotation key count、成本计数。

`HEEngine` 封装 OpenFHE ciphertext 操作，并在 `mul`、`mul_plain`、`rotate` 等方法中累计成本计数。

当前 native runner 为实验实现，JSON/YAML 解析多处使用 regex，适合项目内固定 schema，不等价于通用 parser。

## 17. 报告与产物

主实验输出：

```text
outputs/runs/<experiment_id>/
  manifest.yaml
  configs/
    experiment.yaml
    model.yaml
    dataset.yaml
    approximations.yaml
    ckks.yaml
  profiles/
    operator_calibration_stats.csv
    sensitivity_matrix.csv
    combination_diagnostics.csv
  schedules/
    base_reference.yaml
    uniform_low.yaml
    uniform_mid.yaml
    uniform_high.yaml
    hetune_additive_greedy.yaml
    hetune_generated.yaml
    validated_greedy_decisions.csv
    selection_summary.csv
    softmax_selection.csv
  evaluations/
    metrics.csv
  distillation/
    summary.csv
    overrides.pt
    overrides_summary.csv
    report.md
  he_analysis/
    he_metrics.csv
    he_cost_breakdown.csv
    profile_coverage.csv
    bootstrap_plan.csv
  he_deployment/
    comparison.csv
    deployment_report.md
  figures/
    sensitivity_heatmap.png
    he_cost_breakdown.png
  reports/
    report.md
    he_report.md
    artifacts_index.md
```

`write_artifacts_index()` 会列出关键产物路径和是否存在，适合作为每次 run 的检查入口。

## 18. 安全校验

`src/hetune/security/validators.py`

`SecurityValidator` 当前是 MVP 级别检查：

- schedule 不能为空。
- `constraints.input_independent` 不能为 false。
- 每个 entry 必须有 candidate id。

它不证明 HE 安全性，只保证 schedule 不是显式 input-dependent，并捕获空 schedule 等明显错误。真正 HE 安全参数来自 CKKS 配置中的 `security_bits`、modulus chain、poly modulus degree 等。

## 19. 测试覆盖

测试主要覆盖：

- `test_schedule.py`：uniform/base/greedy/validated greedy 行为。
- `test_registry_cost.py`：registry 排序、base 排除、静态成本排序、CKKS-only 过滤。
- `test_high_approximations.py`：高质量 GELU/LayerNorm 近似优于低阶候选。
- `test_softmax_candidates.py`：Softmax 归一化、成本排序、高质量候选误差。
- `test_calibration_stats.py`：hook 收集统计和 coverage。
- `test_he_planner.py`：HE uniform 候选选择、bootstrap 插入、单算子超预算。
- `test_he_analysis.py`：profile 过滤、严格 coverage、run-he 离线输出。
- `test_deployment.py`：部署配置、backend availability、三 case 输出、forward artifact request、native accuracy gate。
- `test_distillation_utils.py`：overrides payload 保存、加载、应用。
- `test_serialization.py`：schedule dict roundtrip。
- `test_paths_and_scope.py`：输出路径和 scope 映射。
- `tests/integration/test_offline_pipeline.py`：不依赖 HuggingFace 的离线 schedule pipeline。

## 20. 典型运行流程

### 20.1 快速 smoke test

```bash
uv sync --extra dev
uv run pytest tests/unit
uv run hetune run-activate --config configs/experiments/bert_tiny_sst2.yaml
```

BERT-Tiny 配置适合检查 pipeline，不适合报告有意义 SST-2 精度。

### 20.2 DistilBERT all nonlinear

```bash
uv run hetune run-all --config configs/experiments/distilbert_sst2_all_nonlinear.yaml
```

该命令会：

1. 加载 DistilBERT SST-2。
2. 收集校准统计。
3. 做 GELU/LayerNorm/Softmax 单点 sensitivity。
4. 生成 base、uniform、additive greedy、validated greedy schedules。
5. 可选蒸馏 LayerNorm 参数。
6. 在 validation split 上评估。
7. 写报告和图表。

### 20.3 HE-aware all nonlinear

```bash
uv run hetune run-all-he --config configs/experiments/distilbert_sst2_all_nonlinear_he.yaml
```

该命令比 `run-all` 多了：

- CKKS-only candidate filtering。
- profile required / profile coverage 严格检查。
- HEUniformPolicy 初始 high schedule。
- validated greedy 中检查 level 和 bootstrap 约束。
- 运行完成后执行 HEAnalysisRunner。

### 20.4 只分析已有 schedule 的 HE 成本

```bash
uv run hetune run-he --config configs/experiments/distilbert_sst2_all_nonlinear_he.yaml
```

前提是对应 run 目录已有 `schedules/hetune_generated.yaml` 等 schedule 文件。

### 20.5 OpenFHE 部署

```bash
scripts/install_openfhe.sh
scripts/build_openfhe_runner.sh
uv run hetune deploy-he --config configs/deployment/distilbert_sst2_openfhe.yaml
```

部署配置默认对比：

- `high`：`uniform_high`
- `pre_distill`：`hetune_generated`
- `post_distill`：`hetune_generated` + `distillation/overrides.pt`

若 OpenFHE 或 runner 不可用，默认 fail fast。可使用 `--allow-unavailable-backend` 生成 infeasible 报告。

## 21. 扩展指南

### 21.1 新增近似候选

1. 在对应 `operators/*.py` 中新增 provider candidate。
2. 填好 `ApproximationSpec`，尤其是：
   - `candidate_id`
   - `quality_rank`
   - `supports_ckks_backend`
   - `expected_accuracy_risk`
   - `cost_hint`
3. 在 plaintext 实现中处理校准 context。
4. 在 `configs/approximations/*.yaml` 中启用 candidate。
5. 添加单元测试，至少覆盖输出合法性、误差相对关系、成本排序。
6. 若支持 native runner，需要同步 C++ 的 candidate 行为或成本统计。

### 21.2 新增模型结构

1. 在 `HFSequenceClassifierAdapter.discover_operators()` 增加 model_type 分支。
2. 为每层生成稳定 `OperatorKey`。
3. 若替换 Softmax，需要在 `attention_wrappers.py` 添加 wrapper。
4. 添加离线或小模型测试，验证 operator discovery 和 apply schedule。

### 21.3 新增数据集

1. 新增 `configs/datasets/<name>.yaml`。
2. 确认 `text_fields`、`label_column`、split 名称。
3. 如非 accuracy metric，需要扩展 `profiling/metrics.py` 和 reporting/evaluate。

### 21.4 接入真实 profile

1. 生成包含标准 columns 的 CSV/JSON。
2. 确保 `backend_id` 和 `ckks_param_id` 与 ckks config 匹配。
3. 在 ckks config 设置 `backend_profile_path`。
4. 对 deployable 结论开启：
   - `scheduler.profile_required: true`
   - `scheduler.profile_min_coverage: 1.0`

### 21.5 强化部署

当前 Python 编排和 native runner 已有接口边界，但 native runner 内部 schema 解析较简化。若要用于更严格实验，应优先：

- 引入正式 JSON parser。
- 避免用 regex 解析 YAML。
- 补齐更多 model architecture。
- 扩展 bootstrap 参数调优和监控；`openfhe_distilbert_forward` 已按 schedule policy 执行真实 CKKS bootstrap，轻量 workload 仍只计数。
- 将 client-produced ciphertext 接入 forward artifact 边界。

## 22. 当前实现限制

- 调度主流程仍是 plaintext PyTorch 仿真，不是端到端 HE 推理。
- `bench-seal-profile` 当前基于 cost hint 生成 deterministic profile，不是真实 SEAL benchmark。
- native runner 支持的是项目内固定 DistilBERT forward artifact schema。
- `softmax.clipped.stable.v1` 不支持 CKKS backend，HE-aware registry 会过滤它。
- `SecurityValidator` 只是基础 schedule 校验，不是完整安全证明。
- 蒸馏只训练 LayerNorm 替换模块 weight/bias，不训练 GELU/Softmax。
- HE planner 的 bootstrap placement 仍是基于 level cost 的模型化分析；真实执行目前仅接入 OpenFHE DistilBERT forward 路径。

## 23. 一句话总结

HETune-LLM 的核心设计是把“模型中的可替换算子发现”“近似候选实现”“精度敏感度 profiling”“HE 成本/可行性建模”“每层 schedule 搜索”“可选蒸馏”“报告与部署”拆成清晰模块，并通过 `OperatorKey` 和 `SchedulePlan` 串联。实验主链路负责找出精度可接受且成本更低的每层近似组合；HE 分析链路负责验证 profile 覆盖和 CKKS level/bootstrapping 可行性；部署链路负责把已有 schedule 交给外部 OpenFHE runner 做真实环境验证。
