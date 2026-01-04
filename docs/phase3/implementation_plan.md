# Implementation Plan for Chapter 3 Lab

## Goal
Build a small, explicit “backprop lab” that reproduces the chapter’s worked examples:
- Softmax + cross-entropy with the TinyGPS (city) classifier
- Chain rule backprop for parameters `m,b`
- Single-neuron regression trained by gradient descent with MSE and L1
- (Optional) run the chapter’s supporting code: Llama forward pass + Map of Language

## 0) Repo scaffold (agent should create)
```
ch3_backprop_lab/
  pyproject.toml          # numpy, matplotlib, pandas, pytest
  README.md

  ch3lab/
    __init__.py

    maths/
      softmax.py          # stable softmax, CE, gradients
      losses.py           # MSE, L1, CE helpers
      finite_diff.py      # numerical gradient checker

    models/
      tinygps.py          # K-class linear logits -> softmax
      single_neuron.py    # y=mx+b for MSE and L1

    training/
      sgd.py              # SGD step + training loops
      metrics.py          # accuracy, mean loss

    experiments/
      ex_city_walkthrough.py     # prints tables like the book
      ex_city_train.py           # trains + compares to book tables
      ex_regression_mse.py       # step tables + plots
      ex_regression_l1.py        # step tables + plots
      ex_llama_forwardpass.py    # optional (HF Transformers)
      ex_map_of_language.md      # instructions + notes

    viz/
      plots.py            # scatter+line, loss curves, etc.

  tests/
    test_softmax_ce.py
    test_grad_check.py
    test_single_neuron_grads.py
```

## 1) Core math primitives (must be exact)

### 1.1 Stable softmax + cross-entropy
Implement in `maths/softmax.py`:

```text
softmax(logits: np.ndarray) -> np.ndarray
  Use stability: logits - logits.max().

cross_entropy(p: np.ndarray, y_index: int, eps=1e-12) -> float
  -log(p[y_index])

softmax_ce_backward(p: np.ndarray, y_index: int) -> np.ndarray
  Return dL/dh = p - y_onehot.
```

This is the chapter’s key simplification: log + exp “cancel out,” giving the clean gradient rule.

**Unit tests**
- `softmax(logits).sum() == 1`
- `dL/dh` matches finite-difference gradient on a tiny logits vector

### 1.2 Finite-difference gradient checker
Implement in `maths/finite_diff.py`:

```text
finite_diff_grad(f, theta, eps=1e-5) -> np.ndarray
```

Use this to validate:
- softmax + CE gradients
- city model parameter gradients
- single-neuron gradients

(Gradient checking is the fastest way to build trust in your backprop code.)

## 2) TinyGPS city classifier (Chapter 3’s main walkthrough)

### 2.1 Model definition (general K-class, support 2-class exercises)
Implement in `models/tinygps.py`:

**A) 1D input version (longitude only)**
Exercises explicitly say the simplified models use longitude `x` only and compute `h` and `y^` for each example.

Parameters:
- `m: np.ndarray` shape `(K,)`
- `b: np.ndarray` shape `(K,)`

Forward:
```text
h = m * x + b
p = softmax(h)
```

Loss: CE on correct class index.

**B) Optional extension: 2D input (lat+lon, “planes”)**
The chapter shows adding Barcelona forces using both longitude and latitude (each neuron becomes a plane).

Parameters:
- `M: np.ndarray` shape `(K, 2)`
- `b: np.ndarray` shape `(K,)`

Forward:
```text
h = M @ x_vec + b
```

### 2.2 Backward pass (the chain rule, explicit)
Implement `backward_1d(x, p, y_index)`:

```text
dL_dh = p - one_hot(y_index)

dL_dm = x * dL_dh
dL_db = dL_dh
```

The chapter motivates this via chain rule modularity.

**Tests**
- Compare analytic gradients (`dL_dm`, `dL_db`) to finite-difference estimates for random `m,b` and one `(x,y)` example.

### 2.3 Training loop + table reproduction
Implement in `training/sgd.py`:

```text
sgd_step(params, grads, lr)
train_steps(model, dataset, lr, steps, order="fixed") -> history
```

Important: “accuracy requires passing all four examples.”
Several exercises tell you to compute accuracy by running all 4 examples through the model.

Implement in `training/metrics.py`:
- `accuracy(model, dataset) -> float`
- `mean_ce_loss(model, dataset) -> float`

### 2.4 Experiment scripts (city exercises)
Implement in `experiments/ex_city_walkthrough.py`:
- Print a step-by-step table for:
  - Step 0: forward values `h`, probabilities `y^`
  - gradients `∂L/∂m`, `∂L/∂b`
  - loss and accuracy
- Then apply one GD step with `lr=0.1` and print Step 1.

Implement in `experiments/ex_city_train.py`:
- Run the full number of steps asked (e.g. step 9 / step 3 depending on the table).
- Compare final loss/accuracy with the book’s table (implement an “expected table” fixture once you copy the book’s numbers).

## 3) Single neuron regression — MSE (Exercises 3.18–3.23)
The exercises define a dataset (points) and ask you to fill forward pass, derive gradients, do GD updates with `lr=0.1`, plot lines at steps 0/1/7, and measure convergence steps.

Implement in `models/single_neuron.py` (MSE mode):

```text
y_hat = m*x + b
loss_mse = 0.5*(y_hat - y)**2

dL_dm = (y_hat - y)*x
dL_db = (y_hat - y)
```

Implement in `experiments/ex_regression_mse.py`:
- Step 0 forward pass for first example `(x=1, y=3)`
- Step 1 parameter update with `lr=0.1`
- Step 7 forward pass
- Plot lines for steps 0, 1, 7 on the scatter (matplotlib)
- “Steps until within ±0.001 of true `m,b`” and experiment with higher LR (with divergence guard)

## 4) Single neuron regression — L1 (Exercises 3.24–3.29)
Same structure but with L1 loss and explicit comparison to MSE convergence.

In `models/single_neuron.py` (L1 mode):

```text
loss_l1 = abs(y_hat - y)

subgradient:
  dL_dyhat = +1 if y_hat > y
             -1 if y_hat < y
              0 if equal

then:
  dL_dm = dL_dyhat * x
  dL_db = dL_dyhat
```

In `experiments/ex_regression_l1.py`:
- Same deliverables as MSE: step 0, step 1, step 7, plot lines
- “Steps to ±0.001” and compare to MSE convergence rate

## 4.5) UI layout split (TinyGPS vs Regression)

- Routes:
  - `/backprop/tinygps`: TinyGPS softmax-CE walkthrough.
  - `/backprop/regression`: single-neuron regression (MSE/L1).
- Components:
  - `frontend/src/components/backprop/tinygps/*`
  - `frontend/src/components/backprop/regression/*`
  - shared helpers in `frontend/src/components/backprop/common/*`
- CSS:
  - `frontend/src/styles/tinygps.css`
  - `frontend/src/styles/regression.css`
  - `frontend/src/styles/backprop-common.css`

Include a regression plot panel that overlays the fitted line at steps 0, 1, and 7 on top of the sample scatter.

## 5) Optional “supporting code” tasks (Chapter extras)

### 5.1 Llama forward pass (supporting code 3.1)
If you have the environment for it (GPU helps), implement `experiments/ex_llama_forwardpass.py` mirroring the chapter: load `meta-llama/Llama-3.2-1B`, tokenize prompt, compute next-token softmax probabilities, print top-k.

### 5.2 Map of Language (Exercise 3.30)
The exercise explicitly instructs you to clone and run the interactive Map of Language from the supporting repo and observe clusters across layers.

Deliverable: a short `experiments/ex_map_of_language.md` with:
- commands to run the demo
- what clusters you saw
- how they change for earlier vs later layers

## Exercise briefing for the AI agent (Chapter 3)

### 3.1–3.3 Conceptual questions
- **3.1**: Write a short argument about whether backprop is “as important as Newton’s laws” (Werbos claim).
- **3.2**: Explain how many simple GD steps can accumulate into complex behavior.
- **3.3**: Explain why a single-layer plane-fitting network can’t fit the Baarle-Hertog border complexity.

Deliverable: `answers/ch3_conceptual.md`

### 3.4–3.10 Two-city TinyGPS walkthrough (Paris vs Berlin)
- **3.4** Step 0 forward pass table: compute logits `h` and probabilities `y^` for the 4 examples (longitude only).
- **3.5–3.6** Derive and fill gradients `∂L/∂m` and `∂L/∂b` using softmax + CE derivative + chain rule.
- **3.7** Compute Step 0 cross-entropy loss and accuracy (accuracy requires all 4 examples).
- **3.8** Apply GD with `lr=0.1` to produce Step 1 params.
- **3.9** Compute final step (step 9) loss and accuracy.
- **3.10** Implement training and verify you match the table.

Deliverables:
- `experiments/ex_city_walkthrough.py` (prints Step 0 + Step 1 tables)
- `experiments/ex_city_train.py` (runs full steps and prints final loss/acc)

### 3.11–3.17 Two-city TinyGPS walkthrough (Paris vs Madrid)
Same tasks as 3.4–3.10, but for the Paris/Madrid dataset and the second table.

Deliverables:
- Add dataset fixture + a second run mode in the same scripts

### 3.18–3.23 Single neuron with squared error (MSE)
- **3.18** Step 0 forward pass for `x=1, y=3` (compute `y^` and Loss).
- **3.19** Solve `∂L/∂m`, `∂L/∂b` via chain rule; fill table values.
- **3.20** Step 1 update with `lr=0.1`.
- **3.21** Step 7 forward pass.
- **3.22** Plot fitted line at steps 0, 1, 7 and assess improvement; estimate “correct” `m,b`.
- **3.23** Implement GD; count steps to get within ±0.001 of true `m,b`; see if higher LR reduces steps.

Deliverables:
- `experiments/ex_regression_mse.py`
- plot images saved under `out/`

### 3.24–3.29 Single neuron with L1 loss
Same as 3.18–3.23 but with L1 and an explicit comparison to MSE convergence.

Deliverables:
- `experiments/ex_regression_l1.py`
- a printed comparison summary: steps(MSE) vs steps(L1)

### 3.30 Map of Language interactive
Clone and run the interactive Map of Language; describe clusters and how they change across layers.

Deliverable:
- `experiments/ex_map_of_language.md`

## “Done” criteria
- `tinygps.py` gradients pass finite-difference checks.
- Your scripts can print a step-by-step table (Step 0 → Step 1) that matches the chapter’s arithmetic once you enter the same initial params.
- You can reproduce MSE vs L1 training behaviors and quantify convergence steps.
- Optional: you can run the Llama forward-pass demo and inspect the Map of Language.

If you want, I can also generate a single Codex prompt that contains the entire plan as “tasks in order,” with exact function signatures and pytest assertions, so the agent can execute it mechanically.
