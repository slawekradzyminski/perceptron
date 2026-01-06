# Chapter 2 - Exercises Q&A

Note (2026-01-04): Numeric values for exercises 2.7–2.17 below are from live Ollama (llama3.2:1b) via top-logprobs. Values can differ from the book because Ollama returns only top-k candidates; when the correct token is missing, we record p=0 and CE≈27.631.

## 2.1 What is a local minimum, and why were early AI pioneers like Geoffrey Hinton worried about them?

A local minimum is a point in parameter space where the loss is lower than in its immediate neighbourhood (small nudges in any direction make the loss go up), but it is not necessarily the lowest possible loss overall (the global minimum).
Early neural nets were trained with gradient methods on non-convex loss surfaces, so pioneers worried optimisation would get stuck in a "good enough" dip and never reach a much better solution elsewhere, especially with limited compute, smaller models, and less robust training tricks.

## 2.2 Why are local minima less of a concern for modern neural networks than once thought?

Main reasons:
- High-dimensional geometry: In very high dimensions, true "bad" local minima are relatively rare compared with saddle points and broad flat regions. Many critical points are not traps; they have escape directions.
- SGD noise helps: Mini-batch stochastic gradient descent introduces randomness that helps the optimiser jiggle out of shallow traps and saddles.
- Overparameterisation: Modern nets have so many parameters that there are typically many low-loss solutions; often they are connected by paths of similarly low loss (valleys or flat basins), so getting stuck in a terrible minimum is less common.
- Better training practice: Initialisation schemes, normalisation, residual connections, adaptive optimisers, learning-rate schedules, and other techniques make optimisation smoother and more stable.

## 2.3 How does cross-entropy penalise our model differently than the L1 loss?

Assume p is the model's probability assigned to the correct answer.

L1-style loss (as used in the chapter): roughly 1 - p.
It increases linearly as the model gets less confident.

Cross-entropy: -log(p).
It increases nonlinearly and explodes as p approaches 0.

So cross-entropy punishes "confidently wrong" predictions far more than L1. That strongly encourages the model to move probability mass away from wrong answers and towards the correct one, especially when it is currently very sure of the wrong thing.

## 2.4 In Figure 2.11, we tried training our Llama model one parameter at a time. What went wrong?

Optimising one parameter at a time (coordinate-wise search) fails because neural network parameters are highly coupled:
- The best value for one weight depends on the current values of many others.
- After you adjust parameter A, the "best" value you previously found for parameter B may no longer be best (and vice versa).
- You can get oscillation or extremely slow progress because you are not moving in the true downhill direction in the full space.

Gradient descent works because it uses the full gradient vector, updating all parameters in a coordinated way.

## 2.5 Why can't we just train Llama by testing all combinations of parameter values?

Because the search space is astronomically large:
- Llama has hundreds of millions to billions of parameters.
- Even if each parameter had only (say) 10 candidate values, the number of combinations is 10^N, which is completely infeasible.
- Parameters are also effectively continuous (floating point), making exhaustive search even more impossible.

So brute force is not just slow, it is fundamentally intractable.

## 2.6 When training on short texts, why does it look like a wormhole opens up in our loss landscape (Figure 2.21)?

Because you are looking at a 2D slice of an extremely high-dimensional loss surface (often using two directions, like alpha and beta). During training:
- The optimiser moves to a new region of parameter space where the geometry is different.
- The 2D slice you are plotting can suddenly intersect a narrow low-loss valley that was not intersected before.
- With short texts, the loss is dominated by a small number of token predictions, so updates can create sharp, highly localised improvements for that specific batch, making the valley look like it appears suddenly.

So the "wormhole" is mainly a visual artefact of slicing: you did not create a magical tunnel; you moved so that your plotted slice now cuts through a different (and better) part of a huge landscape.

## 2.7 Fill in the missing cross-entropy values for "The capital of France is Paris".

Live Ollama (top-logprobs) per-token cross-entropy values:
- The: p=0.000000 → CE=27.6310
- capital: p=0.000000 → CE=27.6310
- of: p=0.893609 → CE=0.1125
- France: p=0.054923 → CE=2.9018
- is: p=0.623987 → CE=0.4716
- Paris: p=0.715623 → CE=0.3346

## 2.8 Average the token losses; do they match PyTorch?

Mean cross-entropy for the six tokens above:
- Average CE = 9.8471 (live Ollama).

## 2.9 Which predicted tokens have the highest loss values (France/Paris table)? Why?

Highest loss: "The" and "capital" (p=0.000000, CE=27.6310).
Next highest: "France" (p=0.054923, CE=2.9018).
These tokens are missing from top-logprobs or have low p(correct), so the penalty is large.

## 2.10 Which predicted tokens have the lowest loss values (France/Paris table)? Why?

Lowest loss is for "of" (p=0.893609, CE=0.1125).
Second lowest is "Paris" (p=0.715623, CE=0.3346).
These have the highest p(correct), so the penalty is smallest.

## 2.11 What would the L1 loss be for the France/Paris example, and how would it change contributions?

L1 per token is 1 - p(correct). The mean L1 for this example is:
- Average L1 = 0.6186.

Compared with cross-entropy, L1 grows linearly and compresses extremes, so very low p(correct) tokens ("capital") no longer dominate as strongly as they do under cross-entropy.

## 2.12 Fill in the missing cross-entropy values for "An apple a day keeps the doctor away".

Per-token cross-entropy values (live Ollama):
- An: p=0.000000 → CE=27.6310
- apple: p=0.000000 → CE=27.6310
- a: p=0.728502 → CE=0.3168
- day: p=0.991261 → CE=0.0088
- keeps: p=0.939659 → CE=0.0622
- the: p=0.985881 → CE=0.0142
- doctor: p=0.963943 → CE=0.0367
- away: p=0.943730 → CE=0.0579

## 2.13 Average the token losses; do they match PyTorch?

Mean cross-entropy for the apple/day table:
- Average CE = 6.9698 (live Ollama).

## 2.14 Which predicted tokens have the highest loss values (apple/day table)? Why?

Highest loss is for "An" and "apple" (p=0.000000, CE=27.6310).
These are the least likely correct tokens in that context.

## 2.15 Which predicted tokens have the lowest loss values (apple/day table)? Why?

Lowest loss is for "day" (p=0.991261, CE=0.0088).
Second lowest is "the" (p=0.985881, CE=0.0142).
These have the highest p(correct), so their penalty is minimal.

## 2.16 How would switching to L1 loss change contributions (apple/day table)?

Average L1 for the apple/day example is:
- Average L1 = 0.3059.

L1 compresses the spread of contributions; the very low-probability tokens still matter, but they do not dominate as much as they do under cross-entropy.

## 2.17 Which predictions have the highest and lowest loss values in "I had a perfectly wonderful evening, but this wasn't it"?

From live Ollama:
- Highest loss: "I", "perfectly", and "wasn't" (p=0.000000, CE=27.6310).
- Other high-loss tokens: "this" (CE=4.6577), "had" (CE=4.5026), "wonderful" (CE=4.3760).
- Lowest loss: "a" (p=0.356168, CE=1.0324).
- Next lowest: the comma token after “evening” (CE=1.3359).

## 2.18 Compare logits between Llama 1B and Llama 3B (Ollama)

Setup (2026-01-04):
1) Pulled both models:
   - `docker exec ollama-llama ollama pull llama3.2:1b`
   - `docker exec ollama-llama ollama pull llama3.2:3b`
2) Queried each model with the same prompt via Ollama `/api/generate`:
   - Prompt: `She looked at me and`
   - Params: `logprobs=true`, `raw=true`, `top_logprobs=10`, `temperature=0`, `num_predict=1`

Top candidates (live Ollama)

Llama 3.2:1B (top 10)

| Rank | Token | Prob | Logprob |
| --- | --- | --- | --- |
| 1 | ` said` | 0.506116 | -0.680989 |
| 2 | ` smiled` | 0.188399 | -1.669194 |
| 3 | ` asked` | 0.036917 | -3.299081 |
| 4 | ` I` | 0.036121 | -3.320882 |
| 5 | ` whispered` | 0.017216 | -4.061894 |
| 6 | ` laughed` | 0.016041 | -4.132615 |
| 7 | ` her` | 0.010780 | -4.530026 |
| 8 | ` raised` | 0.010412 | -4.564759 |
| 9 | ` saw` | 0.009262 | -4.681878 |
| 10 | ` then` | 0.008269 | -4.795213 |

Llama 3.2:3B (top 10)

| Rank | Token | Prob | Logprob |
| --- | --- | --- | --- |
| 1 | ` smiled` | 0.327549 | -1.116116 |
| 2 | ` said` | 0.233741 | -1.453541 |
| 3 | ` I` | 0.115495 | -2.158532 |
| 4 | ` asked` | 0.044368 | -3.115241 |
| 5 | ` raised` | 0.021367 | -3.845916 |
| 6 | ` saw` | 0.021132 | -3.856958 |
| 7 | ` then` | 0.016756 | -4.088976 |
| 8 | ` she` | 0.012679 | -4.367808 |
| 9 | ` my` | 0.011626 | -4.454521 |
| 10 | ` whispered` | 0.009974 | -4.607743 |

Interpretation:
- The 1B model is more peaked (overconfident) on ` said` (~0.51), while the 3B model spreads probability more evenly and puts more mass on ` smiled` and ` I`.
- This is consistent with the intuition that larger models can represent more plausible continuations and maintain higher-entropy distributions.

Notes on other models:
- `gemma3:1b`, `qwen3:1.7b`, and `ministral-3:3b` currently return `logprobs: null` via Ollama, so direct logits tables are not available through this backend for those models.
  For those, use a backend that exposes full logits (e.g., Transformers) or a model that supports `top_logprobs` in Ollama.
