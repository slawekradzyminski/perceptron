# Ollama API (Docker) - Llama 3.2 1B

Status check (verified)
- Date: 2026-01-03
- Container: `ollama-llama`
- Model: `llama3.2:1b` pulled and responding on port `11434`

Quick start (run locally)
```bash
# 1) Pull Ollama image
docker pull ollama/ollama:0.13.2

# 2) Start server
docker run -d --name ollama-llama \
  -p 11434:11434 \
  -v ollama-llama:/root/.ollama \
  ollama/ollama:0.13.2

# 3) Pull the model
docker exec ollama-llama ollama pull llama3.2:1b

# 4) Confirm model list
docker exec ollama-llama ollama list

# 5) Smoke test
curl -s http://localhost:11434/api/generate \
  -d '{"model":"llama3.2:1b","prompt":"Say hello in one short sentence.","stream":false}'
```

Stopping / cleanup
```bash
docker stop ollama-llama
# optional: remove container
# docker rm ollama-llama
```

Using Ollama in the lab
- The Chapter 2 exercises (2.7-2.17) can be computed from static PDF tables.
- If you want live values, point the backend adapter at the Ollama server on
  `http://localhost:11434` and request logits or next-token probabilities.

Build a custom Llama image (like awesome-localstack)
- The `awesome-localstack/ollama` folder contains a reusable Dockerfile and
  `wait-and-pull.sh` that pre-pull models during image build.
- To create a Llama image, reuse that Dockerfile and pass a Llama model name.

Example (from that template):
```bash
# From /Users/admin/IdeaProjects/awesome-localstack/ollama
# Build an image that pre-pulls Llama 3.2 1B

docker build \
  --build-arg OLLAMA_MODEL=llama3.2:1b \
  -t slawekradzyminski/llama3.2:1b .

# Run it

docker run -d --name ollama-llama \
  -p 11434:11434 \
  -v ollama-llama:/root/.ollama \
  slawekradzyminski/llama3.2:1b
```

Notes
- The Ollama server exposes HTTP endpoints on port 11434.
- Mounting `/root/.ollama` keeps pulled models between restarts.
