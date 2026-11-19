# llmprobe：多服务 LLM 推理性能测试工具设计

- 日期：2026-11-19
- 状态：待确认
- 选定方向：Python 异步 CLI

## 1. 背景与目标

需要一套不依赖推理框架内部实现的压测工具，用同一套工作负载测试：

- 本地 vLLM / llama.cpp / 其他 OpenAI-compatible 服务
- Qwen、DeepSeek 等官方 API
- 第三方模型网关

核心目标是以可重复、可比较的方式测量不同并发和输入/输出长度下的服务性能。

### 成功标准

1. 一个 YAML 配置可以同时声明多个待测服务，并对它们执行完全相同的请求矩阵。
2. 能可靠测量流式请求的 TTFT、TPOT、ITL、E2EL、请求吞吐和 token 吞吐。
3. 能显式组合并发数、输入长度和输出长度，并支持 warmup 和冷却时间。
4. 原始请求级数据与聚合结果均落盘，便于回归比较和后续二次分析。
5. 对网络错误、HTTP 错误、流中断、超时和不支持 usage 的服务有明确行为。
6. 单元测试和本地 mock 服务集成测试可覆盖核心计时逻辑。

## 2. 非目标

第一版不包含：

- 模型输出质量、准确率或业务效果评估
- GPU 利用率、显存、KV cache、服务端排队时间等引擎内部指标
- 模型启动时间、加载时间、离线 batch 推理
- 多模态、embedding、rerank、transcription
- 多轮对话、工具调用链路和 agent 工作负载
- Web UI、实时图表、数据库
- 分布式或多进程压测
- open-loop 固定 QPS / 到达率模式
- 自动重试隐藏故障（warmup 的能力探测除外）
- 自动启动或停止被测服务

## 3. 可选方案与决策

### A. 独立 Python 异步 CLI（已选）

使用 `httpx.AsyncClient` 直接调用 OpenAI-compatible HTTP API，自行解析 SSE 并计时。配置驱动，支持多 target、参数扫描和 JSONL/CSV/JSON 输出。

优点：跨服务、计时口径可控、依赖少、适合 CI。缺点：需要自行维护不同 API 方言的兼容层。

### B. CLI + Web 面板

在 A 上增加浏览器界面、实时图表和历史数据库。可视化和交互更好，但会把第一版范围扩大到一个前后端项目。以后可在稳定结果格式之上增加。

### C. 封装 vLLM bench / GuideLLM

开发量小，但指标口径、端点兼容和配置受上游约束，且不适合直接作为多服务统一比较层。仅作参考，不作为实现基础。

## 4. 用户接口

安装后提供 `llmprobe` 命令：

```bash
llmprobe run configs/compare.yaml
llmprobe run configs/compare.yaml --dry-run
llmprobe validate configs/compare.yaml
```

- `run`：校验配置、打印待执行矩阵、执行 warmup 和正式测量、即时显示进度、落盘结果并在终端打印汇总。
- `--dry-run`：只展开并打印 target × workload 矩阵，不建立连接。
- `validate`：只做配置和 URL 校验。
- 默认输出目录为 `results/`，每次运行生成独立的 `run_id` 子目录。
- 有配置错误时退出码为 2；全部 target/cell 失败时退出码为 1；至少一个 cell 有成功请求时退出码为 0，即使存在部分失败。
- Ctrl-C 会停止调度、等待已有请求收尾或取消它们，并尽量写出已完成的原始数据和部分汇总。

## 5. 配置设计

示例：

```yaml
run:
  name: qwen-compare
  seed: 42
  output_dir: results
  cooldown_s: 2
  save_generated_text: false
  save_itl: false

workload:
  dataset: random
  requests_per_cell: 60
  warmup_requests: 3
  cells:
    - 128-128-1
    - 128-128-4
    - 1024-128-1
    - 1024-128-4
  system_prompt: ""
  tokenizer: /models/Qwen3-8B/tokenizer.json
  stream_options: auto

targets:
  - name: vllm-qwen
    backend: openai-chat
    base_url: http://127.0.0.1:8000/v1
    model: Qwen/Qwen3.6-35B-A3B
    api_key_env: OPENAI_API_KEY
    tokenizer: qwen3.6-35b
    headers: {}
    metrics_url: http://127.0.0.1:8000/metrics
    metrics_headers: {}
    timeout_s: 300
    max_tokens_field: max_tokens
    request_params:
      temperature: 0
      ignore_eos: true

  - name: deepseek
    backend: openai-chat
    base_url: https://api.deepseek.com/v1
    model: deepseek-chat
    api_key_env: DEEPSEEK_API_KEY
    timeout_s: 300
    max_tokens_field: max_tokens
    request_params:
      temperature: 0
```

### 配置规则

- `base_url` 是 API base，通常以 `/v1` 结尾，可带网关自定义前缀。
- 默认端点：`openai-chat` 为 `{base_url}/chat/completions`，`openai-completions` 为 `{base_url}/completions`。
- 完整 URL 始终通过 `rstrip("/")` 后拼接，避免双斜杠。
- `name` 在单次运行内唯一；默认使用配置中的顺序。
- `requests_per_cell` 必须大于等于最大并发。
- `dataset` 第一版仅支持 `random`，默认 `random`。
- `warmup_requests` 不计入任何聚合指标，按 target × cell 执行，详细语义见 6.4。
- `cells` 中第一项是目标“内容 token 数”，包含 `system_prompt`，但不包含 chat template 和特殊 token。chat target 将 system prompt 放入 system message；completions target 将 system prompt 与实际 prompt 拼接。若已知 system prompt 已超过目标长度，配置校验失败。
- `cells` 中第二项是生成 token 数上限，`cells` 中第三项是并发数。没有 tokenizer 时输入长度按确定性文本长度近似；所有实际 token 数优先采用服务端 usage，并记录来源。
- `cells` 的每一项都是一个独立组合，不再根据并发、输入长度、输出长度列表自动做笛卡尔积。
- `workload.tokenizer` 是全局参考 tokenizer，可指向内置别名、本地 `tokenizer.json`、包含 `tokenizer.json` 的模型目录，或 HF repo id。默认安装不强制引入 tokenizer 依赖，缺省时使用内置的确定性文本生成器。
- `targets[].tokenizer` 可覆盖全局 tokenizer。相同 tokenizer 的 target 共享同一组 prompt；不同 tokenizer 的 target 各自生成目标 token 长度的 prompt，使不同模型的输入 token 长度更准确。实际 token 数仍以服务端 usage 为准。
- 内置 tokenizer 位于 `src/llmprobe/tokenizers/`，提供 `qwen3.8-flash-next`、`qwen3.6-35b`、`deepseek-v4.1-flash`、`glm-5.3` 别名；来源、源提交、SHA256、许可证和直接文件 URL 记录在 `manifest.json`，许可证文本随包分发。
- `max_tokens_field` 是 target 级配置，支持 `max_tokens` 和 `max_completion_tokens`，默认 `max_tokens` 以获得更广的兼容性。这样同一矩阵可以公平比较使用不同字段的官方 API。
- `stream_options: auto` 先尝试 `stream_options.include_usage=true` 获取精确 token 数。能力探测在正式测量前通过一个被排除的 probe/warmup 请求完成；若服务明确不支持该参数，则去掉该字段重试并记住该 target 的能力。正式测量不会因探测而重复。
- `request_params` 原样合并进请求体，用于 `temperature`、`top_p`、`ignore_eos` 等厂商扩展。保留字段 `model`、`messages`、`prompt`、`stream`、`stream_options`、`max_tokens`、`max_completion_tokens` 不允许覆盖，配置校验时报错；生成长度只通过 `max_tokens_field` 控制。
- API key 推荐通过 `api_key_env` 从环境变量读取，也支持直接配置 `api_key`；写日志和结果前必须脱敏。headers 支持 `${ENV_VAR}` 展开；缺失环境变量属于配置错误。
- `timeout_s` 是单个请求从发起到流结束的总超时。
- 可选 `http2`、`verify_tls`、`ca_bundle`，用于网关和自签名本地服务。
- 可选 `metrics_url` 指向 vLLM Prometheus endpoint；配置后按 cell 采集 speculative decoding/MTP 指标。`metrics_headers` 用于独立的 metrics 认证，支持 `${ENV_VAR}` 展开。

## 6. 工作负载模型

### 6.1 矩阵

workload cell 为显式配置项：

```text
(input_length, output_length, concurrency)
```

每次正式测量是该 cell 下的一个 target。执行顺序为：

```text
for cell in cells:
    for target in targets:
        warmup(target, cell)
    for target in targets:
        measure(target, cell)
    cooldown_s
```

这样同一 cell 下各 target 的测量时间最接近，减少机器温度、网络波动等时间漂移对比较的影响。cooldown 只发生在整个 cell 的所有 target 完成之后。

### 6.2 并发模式

第一版只实现 closed-loop 固定并发：

- 对每个 cell 创建最多 `concurrency` 个异步 worker。
- 总请求数固定为 `requests_per_cell`。
- worker 从原子递增的请求序号队列取任务，直到所有请求完成。
- 每个请求使用独立连接槽位，但不为请求重建 `AsyncClient`。
- 连接池上限至少为运行中最大并发；HTTP keep-alive 默认开启。
- 记录客户端的调度等待时间，避免把 worker 排队时间误认为服务延迟。

### 6.3 Prompt 生成（vLLM 风格 random 数据集）

`dataset: random` 使用与 vLLM bench random 数据集相同的思路：用固定 seed 从 tokenizer 词表生成确定性 token ID 序列，再 decode 为文本后发送。使用相同有效 tokenizer 的 target 收到完全相同的文本；不同 tokenizer 的 target 各自生成目标 token 长度的 prompt。实际 token 数用各服务的 usage 校正。

同一 cell 内：

- 使用全局 `seed` 和稳定 cell key 生成确定性 prompt 集。
- 同一 cell 内，使用相同有效 tokenizer 的 target 共享完全一致的 prompt 集；不同 tokenizer 组分别生成 prompt，以对齐各自的目标 token 长度。
- 每个请求内容不同，降低 prefix cache 命中导致的偏差。
- warmup 使用单独的 prompt 集。

有本地/内置 tokenizer 时，按随机 token ID 序列 decode/re-encode 到目标内容 token 数；无 tokenizer 时使用可复现的随机词序列，按字符启发式接近目标长度。target 级 tokenizer 覆盖时，每个 target 使用自己的 tokenizer 生成 prompt。两类情况都在结果中记录目标长度、实际 usage 长度以及 token 数来源。`random-range-ratio`、prefix repetition 和 ShareGPT 等数据集留作后续扩展。

`ignore_eos` 等强制生成长度的能力只能通过 provider 专属 `request_params` 开启。不能强制长度时，以服务实际生成 token 为准。

### 6.4 Warmup 与能力探测

- 每个 target 在第一个正式 cell 之前执行一次能力 probe，仅当 `stream_options: auto` 时需要。probe 使用最小输出长度和并发 1，不计入正式结果。
- 对每个 target × cell，在正式测量之前执行 `warmup_requests` 个相同并发、相同输入/输出长度的请求；warmup 使用独立 prompt 集，不计入任何指标或吞吐时间窗口。
- probe 和 workload warmup 都不属于正式测量；这样既能为每个长度/并发准备服务状态，也不改变正式样本数。
- 若 `warmup_requests: 0`，仍保留 `stream_options: auto` 的 probe，因为它是能力探测而非负载预热。
- warmup/probe 失败时，auto 模式可在明确识别为 `stream_options` 不受支持时降级；其他错误将该 target 标记为不可用并跳过其剩余 cell，其他 target 和 cell 继续执行。

### 6.5 vLLM speculative decoding / MTP metrics

- 当 target 配置 `metrics_url` 时，在每个 cell 的 warmup 完成后、正式请求开始前抓取一次 Prometheus 文本，在正式请求结束后再抓取一次。
- 通过计数器差值计算 `spec_decode_num_draft_tokens`、`spec_decode_num_accepted_tokens`、`spec_decode_num_drafts` 和 per-position acceptance rate；warmup、probe 和相邻 cell 不计入该 cell 窗口。
- `Draft accept % = accepted / drafted`；`Mean accept len = 1 + accepted / drafts`。若指标缺失、MTP 未开启或 endpoint 不可用，benchmark 继续执行，并在 cell 结果中标记 unavailable。
- 指标是服务实例级累计值；同一实例上的其他并发流量会进入差值。需要严格 A/B 时应隔离服务实例。

## 7. API 兼容层

### 7.1 第一版 backend

1. `openai-chat`：`POST {base_url}/chat/completions`
   - body: `model`, `messages`, `stream=true`, `max_tokens`/`max_completion_tokens`
   - 默认 user message 由合成 prompt 构成，可附加可选 system prompt
2. `openai-completions`：`POST {base_url}/completions`
   - body: `model`, `prompt`, `stream=true`, `max_tokens`

两者都支持：

- Bearer token（由 `api_key_env` 提供）
- 自定义 headers
- `request_params` 合并
- 兼容的 SSE 解析
- usage 获取与降级计数

后续通过 backend 接口扩展 Responses API、embeddings 等，不改变 runner、指标和存储层。

### 7.2 SSE 解析要求

- 支持 UTF-8 字符跨网络 chunk 边界
- 支持 CRLF/LF、多行 `data:`、空行分隔、注释/keep-alive
- 忽略 `[DONE]`
- 不把最终 usage-only chunk 计作 token 事件
- 检测 HTTP 非 2xx、`error` 对象、流内错误、无效 JSON、连接中断
- 只要求 HTTP 200 且至少收到一个 token-bearing delta 才算成功；不强制某些兼容服务不发送的 `[DONE]`

token-bearing delta 包括：

- chat `delta.content`
- `delta.reasoning`、`delta.reasoning_content`
- `delta.tool_calls` / `delta.function_call`
- completion `choices[].text`

这与 vLLM bench 的口径一致：reasoning 和工具参数也是正在解码的 token，不能把它们全部计入 TTFT。

## 8. 指标定义

所有时间使用单调时钟 `time.perf_counter()`，单位为秒，输出时转换为 ms。

### 8.1 请求级指标

- `client_queue_s`：请求被 worker 领取前的客户端调度等待时间，不计入 TTFT。
- `start_s`：请求即将发送时的客户端时间。
- `ttft_s`：从 `start_s` 到第一个 token-bearing delta 的时间。
- `e2el_s`：从 `start_s` 到最后一个 token-bearing delta 的时间；不包含流关闭和最终 usage chunk 的额外网络尾部延迟。
- `stream_end_s`：从 `start_s` 到 HTTP 流完全关闭（或失败/超时）的时间，用于确定 worker 槽位释放和 cell 实际结束时间。
- `itl_ms`：相邻 token-bearing delta 的间隔列表，属于事件级 ITL；一个 chunk 可能包含多个 token。
- `input_tokens`：优先级为服务端 usage、参考 tokenizer 对文本内容的估算、不可用。tokenizer 估算可能不包含 chat template，必须标记 `estimated=true`。
- `output_tokens`：优先级为服务端 usage、参考 tokenizer 对全部可见生成内容（content/reasoning/tool 参数）的计数、token 事件计数。后两者必须标记来源并说明估算性质。
- `tpot_s`：`(e2el_s - ttft_s) / (output_tokens - 1)`；`output_tokens <= 1` 时不产生 TPOT。
- `success`、`error_kind`、`error`、HTTP 状态码、token count source。

worker 槽位在 `stream_end_s` 才释放，以允许读取最终 usage chunk；延迟指标仍截止到最后一个 token 事件。

TTFT 包含服务端排队、prefill、首 token 调度和网络传输，这正是调用方观察到的首 token 延迟。

### 8.2 Cell 级聚合

只对成功请求计算延迟分位数：

- 完成/失败请求数、错误率
- benchmark duration：从该 cell 第一条正式请求发出，到所有正式请求的 HTTP 流关闭/失败，即最晚的 `stream_end_s`；不使用最后一个 token 事件时间
- request throughput = 成功请求数 / duration
- output token throughput = 成功请求的 output tokens 总和 / duration；event 计数来源时结果可用但标记为近似
- total token throughput = (input tokens + output tokens) / duration；只有所有成功请求的 input/output token 均可确定时才输出，否则为 null
- TTFT：mean、std、p50、p90、p95、p99
- TPOT：mean、std、p50、p90、p95、p99
- ITL：mean、p50、p90、p95、p99
- E2EL：mean、std、p50、p90、p95、p99
- 平均实际输入和输出 token 数
- decode tokens/s per stream = `1000 / mean_tpot_ms`（存在 TPOT 时）
- peak concurrent requests
- token count source 分布：`usage` / `tokenizer` / `events`

分位数采用线性插值；样本数少于 2 时只输出可定义的值，不伪造统计量。

### 8.3 文件粒度

每个 target/cell 产生一行正式聚合结果。需要独立重复实验时，重新运行同一配置。

## 9. 结果与报告

每次运行写入：

```text
results/<run_id>/
  config.effective.yaml
  run.json
  requests.jsonl
  summary.csv
  summary.json
  summary.md
  errors.log
```

### `run.json`

- `schema_version`
- `run_id`、开始/结束时间、工具版本
- 完整有效配置（密钥和敏感 header 已脱敏）
- target 能力探测结果（如是否支持 stream usage）
- `interrupted` 中断状态

### `requests.jsonl`

每个正式请求一行，包含：

- target、cell_id、request_index
- 目标 input/output 长度与实际 token 数
- `client_queue_s`、`ttft_ms`、`e2el_ms`、`tpot_ms`、ITL 摘要
- `output_tokens`、`input_tokens`、token count source
- 成功/失败、错误类型和脱敏后的错误信息
- 可选原始 `itl_ms` 列表（`save_itl: true`）
- 可选生成文本（`save_generated_text: true`）

### `summary.csv`

每个 cell/target 一行，扁平包含第 8 节的核心指标，便于 pandas、Excel 和 CI 消耗。

### `summary.json`

包含每个 cell/target 的行级聚合。`run.json` 保存运行元数据和 target 能力，`summary.json` 专门承载稳定、易比较的指标结构。

### `summary.md`

人类可读的汇总表格，每行对应一个 target/cell，包含以下核心指标：

- completed / failed
- request throughput
- output token throughput
- TTFT p50 / p95
- TPOT p50
- ITL p95
- E2EL p95

终端只打印精简表格，完整分位数和数据写入文件。

## 10. 错误处理与边界行为

- 连接失败、超时、TLS 错误：记为 failed，不重试。
- HTTP 429/5xx：记为 failed，保留状态码；单个请求失败不会中止当前队列，runner 继续执行剩余请求、cell 和 target。
- SSE JSON 损坏、流内 error、异常断流：记为 failed。
- 一个 target 的所有请求都失败时，继续下一个 target，但该 cell 的聚合结果只包含失败计数。
- warmup 失败时：
  - `stream_options: auto` 仅在可明确判断为不支持 usage 参数时降级；
  - 其他错误直接报错并跳过该 target，避免用错误配置继续压测。
- 输出 token 数为 0 或缺失时，TTFT 仍可统计，但 TPOT 和输出吞吐标记为不可用或按实际可用计数计算，并记录来源。
- 计时节点的网络前提、连接复用和客户端侧 backpressure 需要写入 README。
- 结果文件写入过程中不允许包含 API key、Authorization 明文或展开后的敏感 header。

## 11. 架构

```text
src/llmprobe/
  cli.py             # argparse、子命令、退出码
  config.py          # stdlib dataclass 配置模型、YAML 加载、校验、环境变量展开
  models.py          # Target、Cell、RequestResult、CellSummary 等数据模型
  workload.py        # 矩阵展开、cooldown、cell 执行顺序
  promptgen.py       # 确定性 prompt 生成与长度控制
  tokenization.py    # 可选 tokenizer 加载、内置别名和计数降级
  tokenizers/        # 内置常用模型 tokenizer.json + manifest
  client.py          # httpx client、请求构造、超时、流式请求
  sse.py             # 增量 SSE parser
  metrics.py         # 请求级/聚合级指标计算
  report.py          # JSONL/JSON/CSV/Markdown/终端输出
  runner.py          # 编排 target、cell、warmup、取消和关闭
```

依赖：

- 运行时：`httpx`、`PyYAML`、`rich`（配置模型和校验使用标准库 dataclass）
- 可选：`tokenizers` + `huggingface_hub`（tokenizer 精确 prompt 长度、HF repo id 加载）
- 开发：`pytest`、`ruff`；集成测试使用标准库 `http.server` 提供可控 SSE mock 服务

采用 `src/` layout，`pyproject.toml` 管理项目和 console script，目标 Python >= 3.11。CLI 使用标准库 `argparse`，不额外引入 CLI 框架。配置层使用标准库 dataclass 和手写校验，减少运行时依赖，并保证当前无网络沙箱中也可测试。

## 12. 测试策略

### 单元测试

- YAML 配置校验、默认值、唯一 target、矩阵展开
- prompt 生成的确定性和跨 target 一致性
- tokenizer 缺失时的降级
- SSE parser：任意 chunk 切分、UTF-8 多字节、CRLF、多行 data、注释、`[DONE]`、usage-only chunk、坏 JSON
- TTFT/TPOT/ITL/E2EL/吞吐和分位数的公式
- token 计数优先级与来源标记
- 敏感字段脱敏

### 集成测试

用测试内 mock OpenAI-compatible SSE 服务覆盖：

- 正常 chat/completions 流
- 无 usage、无 `[DONE]`、reasoning delta、tool call delta
- 慢首 token 和固定 ITL
- HTTP 4xx/5xx、超时、中途断流、错误 JSON
- warmup/probe 触发 stream_options 降级
- warmup 与 cooldown 的执行顺序
- 并发上限与请求总数
- usage 缺失时的 input/output token 降级与 null 语义

集成测试使用可控等待时间，断言 TTFT/ITL 在合理容差内，不访问公网。

### 手工验收

- 对本地 vLLM 和 llama.cpp 各跑一组 `1/4/8` 并发
- 对 DeepSeek 或 Qwen 官方 API 用小请求量跑通鉴权与 usage
- 验证 `summary.csv` 可直接用 pandas 读取
- 验证 Ctrl-C 后结果文件可解析

## 13. 风险与约束

- 跨公网服务的 TTFT 包含客户端到服务的网络延迟；结果用于同一环境下的相对比较，不应直接等同服务端性能。
- ITL 是 chunk 级事件间隔；若服务端一次返回多个 token，ITL 不是严格逐 token 间隔。TPOT 使用服务端 usage 时更准确。
- 不同服务对 `max_tokens` 的解释、EOS 和 reasoning 可见性不同，因此必须同时报告实际输出 token 数与生成结束原因（如服务提供）。
- 官方 API 可能有速率限制，默认并发应保守；工具不会自动退避重试。
- 单进程 async 客户端在极高并发下可能成为瓶颈；第一版以常见 1–1024 并发为目标，并记录 client queue time 供诊断。
- 当前工作目录不是可用的 git 仓库；设计文档先落盘，无法按要求提交。初始化仓库或恢复到可用 `.git` 后再提交。

## 14. 后续扩展

- open-loop 固定/泊松请求速率
- `random-range-ratio`、prefix repetition、ShareGPT 等数据集
- Responses API、embeddings、rerank
- 从 vLLM `/metrics` 读取服务端队列和 KV cache 指标
- `llmprobe compare` 跨 run 比较和 Markdown/HTML 报告
- 多进程压测和分布式负载生成
- 可恢复运行与实验数据库
