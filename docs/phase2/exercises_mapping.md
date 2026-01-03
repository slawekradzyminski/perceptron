# Chapter 2 Exercises Mapping

This maps every exercise in Chapter 2 to concrete lab features. The second
half (2.7-2.18) relies on token-level cross-entropy tables; these can be
reproduced from the PDF tables or computed live using Llama via Ollama
(Docker) if you want dynamic values.

2.1 Local minima and early concerns
- Feature: 2D loss surface with multiple minima + GD path
- Outcome: show how local minima can trap simple methods

2.2 Why local minima are less of a concern in modern nets
- Feature: random 2D loss slices in high-dimensional parameter space
- Outcome: show that most points are saddle-like and that slices are incomplete

2.3 Cross-entropy vs L1 loss
- Feature: plot L1 and cross-entropy vs p(correct)
- Outcome: demonstrate stronger penalties for confidently wrong predictions

2.4 One-parameter-at-a-time training fails
- Feature: coordinate descent demo with two coupled parameters
- Outcome: show the optimum for parameter 1 shifts after updating parameter 2

2.5 Brute-force parameter search is infeasible
- Feature: combinatorial growth calculator (values^params)
- Outcome: illustrate why grid search fails for large models

2.6 "Wormhole" effect on short texts
- Feature: loss-slice visualization before/after a few GD steps
- Outcome: show slices changing as theta changes, not a literal wormhole

2.7 Fill in missing cross-entropy values (France/Paris table)
- Feature: token table view with per-position p(correct) and -ln(p)
- Outcome: compute per-token cross-entropy directly from the table

2.8 Average the token losses, match PyTorch loss
- Feature: per-token loss list + mean loss output
- Outcome: verify average matches model-reported loss

2.9 Highest-loss tokens (France/Paris)
- Feature: highlight max -ln(p) rows
- Outcome: identify which predictions were most "confidently wrong"

2.10 Lowest-loss tokens (France/Paris)
- Feature: highlight min -ln(p) rows
- Outcome: identify which predictions were most confident and correct

2.11 L1 loss alternative (France/Paris)
- Feature: toggle L1 vs CE and compare per-token contributions
- Outcome: show L1 compresses extremes; CE amplifies low p(correct)

2.12 Fill in missing cross-entropy values (apple/day table)
- Feature: token table view with per-position p(correct) and -ln(p)
- Outcome: compute per-token cross-entropy directly from the table

2.13 Average the token losses, match PyTorch loss
- Feature: per-token loss list + mean loss output
- Outcome: verify average matches model-reported loss

2.14 Highest-loss tokens (apple/day)
- Feature: highlight max -ln(p) rows
- Outcome: identify which predictions were most "confidently wrong"

2.15 Lowest-loss tokens (apple/day)
- Feature: highlight min -ln(p) rows
- Outcome: identify which predictions were most confident and correct

2.16 L1 loss alternative (apple/day)
- Feature: toggle L1 vs CE and compare per-token contributions
- Outcome: show L1 flattens differences vs CE

2.17 Mixed-sentence table ("I had a perfectly wonderful evening, but this wasn't it")
- Feature: token table view with per-position p(correct) and -ln(p)
- Outcome: identify highest/lowest loss tokens and explain why

2.18 Compute a loss landscape for a model other than Llama-3.2-1B
- Feature: use the loss-slice explorer on a small model (MLP/XOR)
- Outcome: produce a contour plot and describe what it shows
