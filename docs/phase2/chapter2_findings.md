# Chapter 2 Findings (Gradient Descent)

Summary
- Gradient descent succeeds because it uses the full gradient vector over the
  full parameter space.
- Local minima and "wormholes" often look dramatic in 2D slices, but those
  visuals can mislead because the true loss surface is extremely
  high-dimensional.
- Looking at single-parameter or 2-parameter slices is useful for intuition,
  but it hides parameter interactions and can make optimization seem fragile.
- Cross-entropy penalizes "confidently wrong" predictions more strongly than
  L1 loss, which helps learning.
- When you update one parameter at a time (coordinate descent), the optimum
  for a given parameter can shift after other parameters change. This makes
  greedy 1D sweeps inefficient for coupled models.
- Loss landscapes can appear to shift or reveal new valleys as training
  progresses because you are effectively viewing a new slice after each
  parameter update.

Implications for the lab
- Use small, controllable models that expose the math explicitly (Adaline,
  logistic neuron, tiny MLP).
- Provide both analytic gradients and finite-difference gradients so users see
  the link between local slope measurements and the full gradient vector.
- Show 2D slices of loss surfaces using random directions (alpha, beta) in
  parameter space to explain why slices can look deceptive.
- Demonstrate why cross-entropy is preferred over L1 by plotting both against
  p(correct).
