# Chapter 2 - Exercises Q&A

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

Using the table probabilities (p(correct)) from the chapter, the per-token cross-entropy values are:
- The: -ln(0.026724) = 3.6222
- capital: -ln(0.000169) = 8.6856
- of: -ln(0.568659) = 0.5645
- France: -ln(0.012172) = 4.4086
- is: -ln(0.141121) = 1.9581
- Paris: -ln(0.391531) = 0.9377

## 2.8 Average the token losses; do they match PyTorch?

Mean cross-entropy for the six tokens above:
- Average CE = 3.3628 (from the table values).

The book reports 3.3752. The small difference is due to rounding in the printed probabilities.

## 2.9 Which predicted tokens have the highest loss values (France/Paris table)? Why?

Highest loss is for "capital" (p = 0.000169, CE = 8.6856).
Next highest is "France" (p = 0.012172, CE = 4.4086).
These tokens are the least probable correct next tokens, so they carry the largest penalties.

## 2.10 Which predicted tokens have the lowest loss values (France/Paris table)? Why?

Lowest loss is for "of" (p = 0.568659, CE = 0.5645).
Second lowest is "Paris" (p = 0.391531, CE = 0.9377).
These have the highest p(correct), so the penalty is smallest.

## 2.11 What would the L1 loss be for the France/Paris example, and how would it change contributions?

L1 per token is 1 - p(correct). The mean L1 for this example is:
- Average L1 = 0.8099.

Compared with cross-entropy, L1 grows linearly and compresses extremes, so very low p(correct) tokens ("capital") no longer dominate as strongly as they do under cross-entropy.

## 2.12 Fill in the missing cross-entropy values for "An apple a day keeps the doctor away".

Per-token cross-entropy values:
- An: -ln(0.001692) = 6.3818
- apple: -ln(0.000615) = 7.3939
- a: -ln(0.649742) = 0.4312
- day: -ln(0.986583) = 0.0135
- keeps: -ln(0.483511) = 0.7267
- the: -ln(0.850819) = 0.1616
- doctor: -ln(0.738021) = 0.3038
- away: -ln(0.954824) = 0.0462

## 2.13 Average the token losses; do they match PyTorch?

Mean cross-entropy for the apple/day table:
- Average CE = 1.9323.

This matches the reported PyTorch loss (1.9323) to rounding.

## 2.14 Which predicted tokens have the highest loss values (apple/day table)? Why?

Highest loss is for "apple" (p = 0.000615, CE = 7.3939).
Second highest is "An" (p = 0.001692, CE = 6.3818).
These are the least likely correct tokens in that context.

## 2.15 Which predicted tokens have the lowest loss values (apple/day table)? Why?

Lowest loss is for "day" (p = 0.986583, CE = 0.0135).
Second lowest is "away" (p = 0.954824, CE = 0.0462).
These have the highest p(correct), so their penalty is minimal.

## 2.16 How would switching to L1 loss change contributions (apple/day table)?

Average L1 for the apple/day example is:
- Average L1 = 0.4168.

L1 compresses the spread of contributions; the very low-probability tokens still matter, but they do not dominate as much as they do under cross-entropy.

## 2.17 Which predictions have the highest and lowest loss values in "I had a perfectly wonderful evening, but this wasn't it"?

From the table:
- Highest loss: "perfectly" (CE = 9.2520), because p(correct) is extremely small.
- Other high-loss tokens: "I" (4.9784), "wonderful" (4.6985), "this" (4.0740), "but" (4.0289).
- Lowest loss: "wasn't" (CE = 0.0030), because p(correct) is near 1.
- Next lowest: "it" (CE = 1.5600).

## 2.18 Compute and visualize a loss landscape for a model other than Llama-3.2-1B.

We use a small two-layer MLP on XOR (or a coupled quadratic toy loss) and plot a 2D slice of the loss surface by choosing two random directions (alpha/beta) in parameter space. The resulting contour plot shows:
- A low-loss valley around the trained parameters.
- Different slice geometry before vs after training, illustrating why 2D views can look like "new valleys" appear even though the model just moved in parameter space.
