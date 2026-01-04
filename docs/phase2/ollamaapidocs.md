# Ollama API Notes (logprobs + token stats)

This file captures quick reference notes for getting token-level probabilities
from Ollama. Keep this alongside `ollamaapi.md` for local Docker setup.

Key idea
- Use `/api/generate` with `logprobs: true`, `num_predict: 1`, and
  `options.top_logprobs` to retrieve the next-token distribution.

Example payload
```json
{
  "model": "llama3.2:1b",
  "prompt": "The capital of France is",
  "stream": false,
  "num_predict": 1,
  "logprobs": true,
  "options": {
    "temperature": 0,
    "top_logprobs": 20
  }
}
```

Expected response fields (subset)
- `response`: generated token text (first token when `num_predict` is 1)
- `logprobs.top_logprobs`: map (or list of maps) from token -> logprob
- `logprobs.token_logprobs`: list of logprobs for generated token(s)

Usage in this repo
- The backend `OllamaClient` parses `logprobs.top_logprobs` and converts
  logprobs to probabilities.
- The `/gd` UI shows per-token `p(correct)` and `-ln(p)` from that output.
