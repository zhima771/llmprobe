# llmprobe

多 target、OpenAI-compatible 的 LLM 推理服务性能测试 CLI。用于比较本地 vLLM、llama.cpp、Qwen/DeepSeek 官方 API 和第三方网关的流式服务性能。

## 安装

```bash
uv sync --extra dev
```

需要 tokenizer 辅助构造更接近目标长度的 prompt 时：

```bash
uv sync --extra dev --extra tokenizer
```

仓库包含 `uv.lock`，`uv sync` 默认按锁定版本安装依赖。`tokenizer` 是可选 extra；启用后可以在 workload 或单个 target 上指定 tokenizer。

### 内置 tokenizer

`src/llmprobe/tokenizers/` 内置了以下常用 tokenizer，可直接使用别名：

- `qwen3.8-flash-next`
- `qwen3.6-35b`
- `deepseek-v4.1-flash`
- `glm-5.3`

```yaml
workload:
  tokenizer: qwen3.6-35b

targets:
  - name: qwen
    base_url: http://127.0.0.1:8000/v1
    model: Qwen3.6-35B-A3B
    tokenizer: qwen3.6-35b

  - name: deepseek
    base_url: http://127.0.0.1:8001/v1
    model: DeepSeek-V4.1-Flash-1017
    tokenizer: deepseek-v4.1-flash
```

`target.tokenizer` 会覆盖 `workload.tokenizer`。相同 tokenizer 的 target 共享同一组 prompt；不同 tokenizer 的 target 会各自生成目标 token 长度的 prompt，因此同一并发下各模型的计算负载更接近。若要求所有 target 接收完全相同的 prompt 文本，只设置全局 `workload.tokenizer`，不要在 target 上覆盖。

内置文件来源、SHA256 和许可证说明见 `src/llmprobe/tokenizers/README.md` 与 `manifest.json`。也可以使用 `tokenizer.json` 文件路径、包含 `tokenizer.json` 的模型目录，或 Hugging Face repo id。

## 快速开始

```bash
cp configs/vllm-chat.yaml config.yaml
uv run llmprobe validate config.yaml
uv run llmprobe run config.yaml --dry-run
uv run llmprobe run config.yaml
```

结果写入 `results/<run_id>/`：

- `config.effective.yaml`：脱敏后的有效配置
- `summary.csv`：每个 target/cell 一行
- `summary.json`：每个 target/cell 的聚合明细
- `requests.jsonl`：请求级数据
- `summary.md`：可读报告
- `run.json`：配置能力、运行元数据
- `errors.log`：失败请求

## 示例配置

仓库提供三份完整示例配置，每份都包含工具当前支持的全部字段；`api_key` 与 `api_key_env` 互斥，因此以注释形式展示：

| 文件 | 用途 | 运行方式 |
|---|---|---|
| `configs/vllm-chat.yaml` | 单个本地 vLLM/OpenAI-compatible chat 服务；并发 1/2/4/8，输入 128/1024，输出 128，启用 warmup 和 `ignore_eos` | 修改 `base_url`/`model` 后直接运行 |
| `configs/compare.yaml` | 多 target 对比：本地 vLLM + DeepSeek 官方 API；3 个显式 cell，比较并发 1/4/8 下的表现 | 设置 `DEEPSEEK_API_KEY` 后运行 |
| `configs/per-target-tokenizers.yaml` | 每个 target 使用各自 tokenizer；包含 Qwen3.8-Flash-Next、DeepSeek-V4.1-Flash、GLM-5.3 示例 | 修改本地端口/model 后运行，需 tokenizer extra |

```bash
# 1. 单服务
uv run llmprobe validate configs/vllm-chat.yaml
uv run llmprobe run configs/vllm-chat.yaml

# 2. 多服务对比
export DEEPSEEK_API_KEY='...'
uv run llmprobe validate configs/compare.yaml
uv run llmprobe run configs/compare.yaml

# 3. 每个 target 使用各自 tokenizer
uv run --extra tokenizer llmprobe validate configs/per-target-tokenizers.yaml
uv run --extra tokenizer llmprobe run configs/per-target-tokenizers.yaml --dry-run
uv run --extra tokenizer llmprobe run configs/per-target-tokenizers.yaml
```

`per-target-tokenizers.yaml` 中的 `127.0.0.1:8000/8001/8002` 只是占位地址，需要改成实际服务地址。

## 配置参数参考

完整结构：

```yaml
run:
  name: my-run
  seed: 42
  output_dir: results
  cooldown_s: 2
  save_generated_text: false
  save_itl: false

workload:
  dataset: random
  requests_per_cell: 16
  warmup_requests: 2
  cells:
    - 4096-512-2
    - 4096-512-4
    - 8192-512-2
  system_prompt: ""
  tokenizer: null
  stream_options: auto

targets:
  - name: local-vllm
    backend: openai-chat
    base_url: http://127.0.0.1:8000/v1
    model: Qwen/Qwen3-8B
    tokenizer: qwen3.6-35b
    api_key_env: null
    api_key: null
    headers: {}
    timeout_s: 300
    max_tokens_field: max_tokens
    request_params:
      temperature: 0
    http2: false
    verify_tls: true
    ca_bundle: null
```

### `run`

| 参数 | 默认值 | 说明 |
|---|---|---|
| `name` | `run` | 运行名称，用于生成 `results/<run_id>/` 目录 |
| `seed` | `42` | prompt 随机种子；相同 seed 和配置可复现 prompt |
| `output_dir` | `results` | 结果根目录 |
| `cooldown_s` | `0` | 每个 cell 的全部 target 完成后等待秒数 |
| `save_generated_text` | `false` | 是否把模型生成文本写入 `requests.jsonl`；可能包含敏感内容 |
| `save_itl` | `false` | 是否写入完整 ITL 序列；高并发/长输出时文件会明显变大 |

### `workload`

| 参数 | 默认值 | 说明 |
|---|---|---|
| `dataset` | `random` | v1 仅支持 `random`，使用 vLLM 风格确定性随机 prompt |
| `requests_per_cell` | `50` | 每个 target × cell 的正式请求数；必须 ≥ 最大并发 |
| `warmup_requests` | `2` | 每个 target × cell 的预热请求数；不计入指标；`0` 表示关闭 |
| `cells` | `[512-128-1]` | 显式 workload 组合列表，格式为 `输入长度-输出上限-并发` |
| `system_prompt` | `""` | 可选 system prompt，会计入输入 token 预算 |
| `tokenizer` | `null` | 全局参考 tokenizer，可用内置别名、`tokenizer.json`、模型目录或 HF repo id |
| `stream_options` | `auto` | `auto`：probe 探测 usage；`include_usage`：强制发送；`none`：关闭 |

`workload.cells` 直接列出要测试的组合，不再自动做笛卡尔积。每个 cell 使用：

```text
输入内容 token 数-输出 token 上限-并发
```

三项含义：

- 第一项：目标输入内容 token 数，包含 `system_prompt`，不含 chat template。
- 第二项：生成 token 数上限，对应请求中的 `max_tokens` / `max_completion_tokens`。
- 第三项：同时进行的请求数，也就是该 cell 的 closed-loop 并发。

每个列表项就是一组独立测试，不会与其他项自动组合。例如：

```yaml
cells:
  - 4096-512-2
  - 4096-512-4
  - 8192-1024-8
```

会生成 3 个 cell，ID 分别为：

```text
in4096-out512-conc2
in4096-out512-conc4
in8192-out1024-conc8
```

也支持等价的列表或映射写法：

```yaml
cells:
  - [4096, 512, 2]
  - {input_length: 4096, output_length: 512, concurrency: 4}
```

每个 cell 对每个 target 正式测量一次；需要独立重复时，重新运行配置即可。请求数计算：

```text
cell 数 = len(cells)
正式请求 = cell 数 × target 数 × requests_per_cell
预热请求 = cell 数 × target 数 × warmup_requests
```

### `targets`

| 参数 | 必填 | 说明 |
|---|---:|---|
| `name` | 是 | target 唯一名称，出现在结果中 |
| `backend` | 否 | `openai-chat`（默认）或 `openai-completions` |
| `base_url` | 是 | API base，通常包含 `/v1`；工具会追加对应 endpoint |
| `model` | 是 | 服务端实际模型名；显示名和权重名不一致时以服务端为准 |
| `tokenizer` | 否 | 覆盖 `workload.tokenizer`；建议使用实际权重对应的 tokenizer |
| `api_key_env` | 否 | 读取 API Key 的环境变量名（推荐） |
| `api_key` | 否 | 直接在 YAML 中填写 API Key；使用方便但不推荐，结果文件会自动脱敏 |
| `headers` | 否 | 额外 HTTP Header；支持 `${ENV_VAR}` 展开 |
| `timeout_s` | 否 | 单请求总超时，默认 `300` 秒 |
| `max_tokens_field` | 否 | 发送 `max_tokens`（默认）或 `max_completion_tokens` |
| `request_params` | 否 | 合并进请求 body 的额外参数，如 `temperature`、`top_p`、`ignore_eos` |
| `http2` | 否 | 是否启用 HTTP/2，默认 `false`；需要 `h2` 依赖 |
| `verify_tls` | 否 | 是否校验 TLS，默认 `true` |
| `ca_bundle` | 否 | 自定义 CA 文件路径 |

`request_params` 不能覆盖 `model`、`messages`、`prompt`、`stream`、`stream_options`、`max_tokens`、`max_completion_tokens` 等协议保留字段。

### API Key 和代理

```bash
export DEEPSEEK_API_KEY='...'
```

推荐使用 `api_key_env`。如果环境变量不方便，也可以直接写 `api_key`：

```yaml
targets:
  - name: gateway
    base_url: http://127.0.0.1:8000/v1
    model: qwen-3.6-35b
    api_key: sk-your-key
```

直接写 `api_key` 会把明文留在 YAML 源文件中，因此要注意文件权限和版本控制；`config.effective.yaml`、`run.json` 等结果文件中的 `api_key` 会被替换为 `<redacted>`。不能同时设置 `api_key` 和 `api_key_env`，也不能同时设置 API Key 和 `Authorization` header。

如果 `ALL_PROXY` 使用非标准 `socks://`，访问非 loopback 服务前请改为 `socks5://host:port` 或 unset；loopback 地址会自动忽略代理。URL query 值会在 dry-run 和持久化配置中脱敏，不要把密钥放在 URL path 或 query key 中。

### 终端结果列

| 列 | 说明 |
|---|---|
| `Target` | target 名称 |
| `Cell` | 输入/输出/并发组合 |
| `OK/Err` | 成功/失败请求数 |
| `Req/s` | 请求吞吐 |
| `Out tok/s` | 所有并发请求的聚合输出 token 吞吐 |
| `Decode tok/s` | 单流 decode 吞吐，`1000 / mean TPOT ms` |
| `TTFT avg/p95` | 首 token 延迟平均值/p95 |
| `TPOT avg/p95` | 每输出 token 时间平均值/p95 |
| `ITL avg/p95` | token 间延迟平均值/p95 |
| `E2EL avg/p95` | 端到端延迟平均值/p95 |

## 指标

- **TTFT**：请求发送到第一个 token-bearing delta 的时间。
- **TPOT**：`(E2EL - TTFT) / (output_tokens - 1)`。
- **ITL**：相邻 token-bearing chunk 的间隔，属于事件级指标。
- **E2EL**：请求发送到最后一个 token-bearing delta。
- **Request throughput**：成功请求数 / benchmark duration。
- **Output token throughput**：成功请求输出 token 总数 / benchmark duration。
- **Total token throughput**：输入和输出 token 总数 / benchmark duration；输入 token 不可知时为 null。
- **Decode tok/s**：`1000 / mean TPOT ms`，即单流 decode 吞吐；同时写入 CSV/JSON/Markdown。

## 配置要点

- `base_url` 通常应包含 `/v1`；实际请求端点会在此后追加 `/chat/completions` 或 `/completions`。
- `dataset: random` 是 v1 唯一支持的数据集，参考 vLLM benchmark random 数据集的思路：使用固定 seed、稳定 cell key 和请求序号生成确定性随机 prompt。
- 有 tokenizer 时（tokenizer-backed）：从 tokenizer 词表生成确定性 token ID 序列，decode 为文本并尽力 re-encode 到目标内容 token 数。使用相同有效 tokenizer 的 target 收到相同文本；不同 tokenizer 组会生成各自的目标长度 prompt。
- 无 tokenizer 时（fallback approximation）：使用确定性随机词序列和约 4 字符/token 的启发式接近目标长度；没有服务端 usage 时通常无法得到输入 token 数。
- `workload.tokenizer` 是全局参考 tokenizer；`targets[].tokenizer` 可覆盖它。tokenizer 只用于构造 prompt 和估算，实际输入/输出 token 数优先采用各服务的 usage，并记录 token count source 与 `estimated` 标记。
- 使用内置别名时不需要复制模型权重；只需要 `tokenizer.json`。
- `random-range-ratio` 在 v1 中尚未实现；每个 cell 的输入目标、输出上限和并发都是显式固定值。
- `cells` 使用 `输入长度-输出上限-并发` 三元组，例如 `4096-512-2`。
- `requests_per_cell` 必须不小于最大并发。
- `warmup_requests` 按 target × cell 执行，不计入结果。
- `stream_options: auto` 会用一个排除在正式结果外的 probe 探测 usage 支持；`include_usage` 强制发送，`none` 强制关闭。
- `max_tokens_field` 是 target 级配置，可选 `max_tokens` 或 `max_completion_tokens`。
- API key 通过环境变量设置，不要写入配置文件：

```bash
export DEEPSEEK_API_KEY=...
```

- 运行时依赖已包含 `socksio`。如果环境的 `ALL_PROXY` 使用非标准 `socks://` scheme，httpx 不会自动把它规范化为 `socks5://`；访问非 loopback target 前请改成 `ALL_PROXY=socks5://host:port`，或 unset `ALL_PROXY`。loopback target 会自动忽略代理环境变量。
- `base_url` 中的 URL query 值会脱敏：`--dry-run` 输出以及持久化的 `config.effective.yaml`、`run.json` 中，所有 query value 都替换为 `<redacted>`（URL 编码后为 `%3Credacted%3E`），query key 保留。不要把密钥放在 URL path 或 query key 中。

## 常见限制

- 公网 API 的 TTFT 包含网络延迟，适合同环境相对比较。
- 服务端可能一次返回多个 token，因此 ITL 是 chunk 级间隔；有 usage 时 TPOT 更准确。
- 无 usage 且无 tokenizer 时无法得到输入 token 数，total token throughput 为 null。
- 第一版只支持 closed-loop 固定并发，不支持固定 QPS。
- 不自动重试；429/5xx 会记录为失败。
- 极高并发下单进程客户端可能成为瓶颈，应观察 client queue time。

## 开发测试

```bash
UV_CACHE_DIR=/tmp/uv-cache uv sync --extra dev
uv run pytest -q
uv run ruff check src tests
```

完整本地检查还可以运行：

```bash
python3 -m compileall -q src tests
uv run llmprobe --version
uv run llmprobe run configs/vllm-chat.yaml --dry-run
```

## License

本项目代码采用 MIT License，见根目录 `LICENSE`。

`src/llmprobe/tokenizers/` 中的第三方 tokenizer 文件不适用本项目 MIT License，
其来源、许可证和通知见 `src/llmprobe/tokenizers/manifest.json`、
`THIRD_PARTY_NOTICES.md` 和 `licenses/`。
