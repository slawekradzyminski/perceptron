# Phase 2 - Gradient Descent Lab (Chapter 2)

This folder captures the Chapter 2 findings and a concrete implementation plan
for expanding the Perceptron Visual Lab into a Gradient Descent Lab. The core
thesis from Chapter 2 is:

"Gradient descent works because it uses the full gradient vector in the full
parameter space. Many confusing pictures (local minima, wormholes) are artifacts
of looking at low-dimensional slices of a high-dimensional loss surface."

Running Llama (optional)
- The Chapter 2 token-loss exercises (2.7-2.17) can be completed from the PDF
  tables alone, but you can also compute them live.
- We can run Llama via Ollama in Docker to avoid local GPU constraints.
- The implementation plan describes a backend adapter to switch between
  "static table" and "live model" modes.

Contents
- chapter2_findings.md: distilled takeaways from Chapter 2
- exercises_mapping.md: how the lab features answer Exercises 2.1-2.18
- implementation_plan.md: build plan for the new gradient descent features
- ollamaapi.md: Docker + Ollama instructions (Llama 3.2 1B)
