# Phase 2 - Gradient Descent Lab (Chapter 2)

This folder captures the Chapter 2 findings and a concrete implementation plan
for expanding the Perceptron Visual Lab into a Gradient Descent Lab. The core
thesis from Chapter 2 is:

"Gradient descent works because it uses the full gradient vector in the full
parameter space. Many confusing pictures (local minima, wormholes) are artifacts
of looking at low-dimensional slices of a high-dimensional loss surface."

Running Llama (live via Ollama)
- The Chapter 2 token-loss exercises (2.7-2.17) are computed live via Ollama.
- Run Llama locally in Docker and query next-token probabilities from logprobs.
- Set `OLLAMA_BASE_URL` and `OLLAMA_MODEL` (default: `llama3.2:1b`) and use `/gd`.
- See `ollamaapi.md` for Docker commands and model pull steps.

Contents
- chapter2_findings.md: distilled takeaways from Chapter 2
- exercises_mapping.md: how the lab features answer Exercises 2.1-2.18
- implementation_plan.md: build plan for the new gradient descent features
