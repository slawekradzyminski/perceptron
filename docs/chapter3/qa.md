# Chapter 3 Q&A

## 3.1 Question
Paul Werbos’ claim that backpropagation is as important to intelligence as Newton’s laws were to physics may seem grandiose. If I were in Minsky’s shoes in the early 1970s I would have been skeptical as well. However, time has only proven Werbos more right—backpropagation has scaled astoundingly well—do you think Werbos was right? Is backpropagation as important to the study of intelligence as F=ma was to the study of physics?

## 3.1 Answer
Werbos’ claim is directionally right: backpropagation is the practical mechanism that makes learning in large, multi-layer models feasible, just as Newton’s laws provide a compact, general rule for predicting physical motion. The analogy isn’t perfect—backprop isn’t a fundamental law of nature, and intelligence isn’t governed by a single equation—but in practice, backprop has played a similarly foundational role by enabling scalable optimization across many domains. So: not identical in philosophical status, but comparable in enabling power.

## 3.2 Question
Backpropagation and gradient descent, when viewed a single step at a time, as shown in Figure 3.19, can look chaotic. It never ceases to amaze me that these little parameter update steps, driven by gradient descent, are able to learn such complicated things—each step is really quite simple. How do you think it is that all these little steps are able to add up to such impressive results?

## 3.2 Answer
Each gradient step is local, but it is consistently aligned with the direction that reduces loss. Over many steps, those small, coherent improvements accumulate into large changes in the model’s behavior. The key is that the loss surface encodes global structure; even local gradients point toward useful basin-wide trends, and repeated updates integrate those signals into a strong overall solution.

## 3.3 Question
Why is it impossible for our single layer plane-fitting neural network to fit the borders of the town of Baarle-Hertog as shown in Figure 3.26?

## 3.3 Answer
A single-layer linear classifier creates one plane (or line in 2D) per class, which can only separate space into convex regions. Baarle-Hertog’s borders are fragmented and disconnected, requiring multiple disjoint regions. A single plane per class can’t carve those shapes, so you need deeper or more complex models to represent such intricate boundaries.

## 3.4 Question (how to do it in the UI)
Using the neural network and equations above, complete the “forward pass” for step zero of the table on the next page, computing the h and ŷ values. Following the same approach as the model introduced in Figure 3.4, this model only uses the longitude (x). The top row of the table indicates the order the examples are iterated through, starting with x=2.351. The y₁ and y₂ rows indicate the city/label of the current example, where y₁=1 corresponds to Paris, and y₂=1 corresponds to Berlin. Initial model parameters are shown in the first column of the table.

## 3.4 Answer (how to complete via UI)
Use the TinyGPS UI to reproduce the table’s Step 0 forward pass:

1) Open **/backprop/tinygps** and select the preset **Paris–Berlin (4)**.  
   This loads the 4‑sample exercise subset for the two cities.

2) In **Exercise Setup**, paste the book’s initial parameters:  
   - `m` = `-1, 1`  
   - `b` = `0, 0`  
   Click **Apply Exercise Params**.

3) (Optional) If the table specifies an explicit sample order, paste it into **Sample order** (e.g., `0,1,2,3`).  
   If the order is the default sequence, you can leave it blank.

4) Press **Step TinyGPS** (or hit **S**) to compute Step 0.  
   The **TinyGPS Step** card shows:
   - **h / logits** (`h₁`, `h₂`): listed under **Logits**  
   - **ŷ / probabilities**: listed under **Probabilities**

5) Record those values into the table’s Step 0 column.  
   Repeat **Step TinyGPS** to advance through the order if you need the full row of step‑by‑step entries.

Notes:
- This UI is longitude‑only for the 2‑city exercise (matches the book’s simplified model).  
- If you want exact book‑style rounding, use the displayed decimals directly; all computations are done in full precision and only rounded for display.

**Loss and Accuracy (how to interpret the table rows):**
- **Loss** is the **cross‑entropy** for the current example only:  
  `Loss = -log(ŷ_correct)` where ŷ_correct is the predicted probability for the true city (y₁ or y₂).  
  Lower is better; 0 means the model assigned probability 1 to the correct city.
- **Accuracy** in the book’s tables is **dataset accuracy** after each step, not just the current sample.  
  The UI computes accuracy by running **all 4 examples** through the current parameters and reporting the fraction correct.  
  So you should interpret the Accuracy row as “model performance over the whole exercise dataset” at that step.

## 3.18 Question (Regression MSE, how to do it in the UI)
Use the regression model `ŷ = m x + b` with MSE loss to complete the step table and line plots.

## 3.18 Answer (UI steps)
1) Open **/backprop/regression**.  
2) Click **Load MSE (Book)** to load the dataset, order, and initial parameters (m=1, b=0, lr=0.1).  
3) Press **Step Regression** (or **S**) to advance step-by-step.  
4) Use **Book Table Mode** to read the exact step values for the table.  
5) The plot overlays the lines at steps 0/1/7 over the data points.

## 3.24 Question (Regression L1, how to do it in the UI)
Use L1 loss instead of MSE and compare the gradient updates and loss.

## 3.24 Answer (UI steps)
1) On **/backprop/regression**, click **Load L1 (Book)**.  
2) Step with **S** to fill the table.  
3) If you want the printed-table match, keep **Loss display = Book‑literal**.  
4) Compare the step history and plotted lines to the MSE case.
