# llmbench

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

`src/llmbench/tokenizers/` 内置了以下常用 tokenizer，可直接使用别名：

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

内置文件来源、SHA256 和许可证说明见 `src/llmbench/tokenizers/README.md` 与 `manifest.json`。也可以使用 `tokenizer.json` 文件路径、包含 `tokenizer.json` 的模型目录，或 Hugging Face repo id。

## 快速开始

```bash
cp configs/vllm-chat.yaml config.yaml
uv run llmbench validate config.yaml
uv run llmbench run config.yaml --dry-run
uv run llmbench run config.yaml
```

结果写入 `results/<run_id>/`：

- `config.effective.yaml`：脱敏后的有效配置
- `summary.csv`：每个 target/cell/repeat 一行
- `summary.json`：明细与跨 repeat 聚合
- `requests.jsonl`：请求级数据
- `summary.md`：可读报告
- `run.json`：配置能力、运行元数据
- `errors.log`：失败请求

## 指标

- **TTFT**：请求发送到第一个 token-bearing delta 的时间。
- **TPOT**：`(E2EL - TTFT) / (output_tokens - 1)`。
- **ITL**：相邻 token-bearing chunk 的间隔，属于事件级指标。
- **E2EL**：请求发送到最后一个 token-bearing delta。
- **Request throughput**：成功请求数 / benchmark duration。
- **Output token throughput**：成功请求输出 token 总数 / benchmark duration。
- **Total token throughput**：输入和输出 token 总数 / benchmark duration；输入 token 不可知时为 null。
- **Decode tok/s per stream**：`1000 / mean TPOT ms`。

## 配置要点

- `base_url` 通常应包含 `/v1`；实际请求端点会在此后追加 `/chat/completions` 或 `/completions`。
- `dataset: random` 是 v1 唯一支持的数据集，参考 vLLM benchmark random 数据集的思路：使用固定 seed、稳定 cell key 和请求序号生成确定性随机 prompt。
- 有 tokenizer 时（tokenizer-backed）：从参考 tokenizer 词表生成确定性 token ID 序列，decode 为文本并尽力 re-encode 到目标内容 token 数；所有 target 收到相同文本。
- 无 tokenizer 时（fallback approximation）：使用确定性随机词序列和约 4 字符/token 的启发式接近目标长度；没有服务端 usage 时通常无法得到输入 token 数。
- `workload.tokenizer` 是全局参考 tokenizer；`targets[].tokenizer` 可覆盖它。tokenizer 只用于构造 prompt 和估算，实际输入/输出 token 数优先采用各服务的 usage，并记录 token count source 与 `estimated` 标记。
- 使用内置别名时不需要复制模型权重；只需要 `tokenizer.json`。
- `random-range-ratio` 在 v1 中尚未实现；同一 cell 的 `input_lengths` / `output_lengths` 是固定目标值，不是随机区间。
- `concurrencies`、`input_lengths`、`output_lengths` 是扫描列表。
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
uv run llmbench --version
uv run llmbench run configs/vllm-chat.yaml --dry-run
```
