# Bundled Tokenizers

这些 `tokenizer.json` 用于 `llmprobe` 的 vLLM 风格 `random` prompt 生成。
工具只读取 tokenizer 元数据，不会加载模型权重。

可用别名：

- `qwen3.8-flash-next`
- `qwen3.6-35b`
- `deepseek-v4.1-flash`
- `glm-5.3`

在配置中可以直接写别名：

```yaml
workload:
  tokenizer: qwen3.6-35b
```

或在 target 上覆盖：

```yaml
targets:
  - name: deepseek
    tokenizer: deepseek-v4.1-flash
```

上游来源、文件和 SHA256 见 `manifest.json`。这些文件受各自上游许可证约束：

- `Qwen/Qwen3.8-Flash-Next`
- `Qwen/Qwen3.6-35B-A3B`
- `deepseek-ai/DeepSeek-V4.1-Flash-1017` (MIT)
- `zai-org/GLM-5.3`

更新 tokenizer 时请同步更新 `manifest.json` 中的 SHA256 和大小。
