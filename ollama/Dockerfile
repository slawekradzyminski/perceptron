FROM ollama/ollama:latest

ARG OLLAMA_MODEL=llama3.2:1b
ARG OLLAMA_EXTRA_MODELS=""

COPY wait-and-pull.sh /usr/local/bin/wait-and-pull.sh
RUN chmod +x /usr/local/bin/wait-and-pull.sh \
    && export OLLAMA_MODEL="${OLLAMA_MODEL}" \
    && export OLLAMA_EXTRA_MODELS="${OLLAMA_EXTRA_MODELS}" \
    && /usr/local/bin/wait-and-pull.sh

EXPOSE 11434
ENTRYPOINT ["ollama", "serve"]
