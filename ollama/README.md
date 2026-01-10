# Ollama Image (Pre-pulled Model)

This folder builds a custom Ollama image with the model pre-pulled.

## Build

```
cd ollama
./push_image.sh
```

This builds and pushes `slawekradzyminski/llama3.2:1b` by default.

## Customize

```
cd ollama
OLLAMA_MODEL=llama3.2:1b \
OLLAMA_EXTRA_MODELS="" \
  docker build -t slawekradzyminski/llama3.2:1b -f Dockerfile .
```

## Use with compose

```
OLLAMA_IMAGE=slawekradzyminski/llama3.2:1b docker compose up -d --build
```
