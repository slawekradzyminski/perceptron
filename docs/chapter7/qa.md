# Chapter 7 - Mechanistic Interpretability | Exercises Q&A

These answers correspond to the exercises from "The Welch Labs Illustrated Guide to AI" (Chapter 7, pages 264-266).

### 7.1 Neural networks are often called "black box" models. Why is it difficult to understand how neural networks, such as modern large language models, work?
Neural networks are composed of billions of parameters (weights and biases) arranged in complex, layered architectures. While we understand the mathematical operations (matrix multiplication, activation functions) perfectly, the *meaning* of the individual parameters is opaque.
* **Distributed Representations:** Concepts are not stored in single neurons but are often spread across many neurons (superposition).
* **Polysemanticity:** A single neuron often responds to multiple, unrelated concepts (e.g., a neuron might fire for "cats" and "finance").
* **Scale:** The sheer number of interactions makes manual inspection impossible.

### 7.2 What is the residual stream?
The **residual stream** is the primary vector (or matrix of vectors) that passes through the entire model, layer by layer.
* **Data Highway:** Instead of layers strictly transforming data sequentially (Layer 1 → Layer 2), modern Transformers use residual connections where the output of a layer is *added* to its input.
* **Accumulation:** The stream accumulates information as it travels deeper. Each attention and MLP block reads from the stream and writes (adds) its result back into it.
* **Transformation:** It starts as the token embedding and eventually becomes the final vector used to predict the next token.

### 7.3 Why do language models typically not just always pick the most probable next token when generating text?
Always picking the most probable token (greedy decoding, temperature = 0) often leads to repetitive, generic, or "boring" text.
* **Sampling:** Models sample from the probability distribution to introduce variety and creativity.
* **Nuance:** Lower probability options might represent valid but less common phrasings, or different "takes" (e.g., skeptical vs. positive views).
* **Temperature:** Parameters like temperature control this randomness.

### 7.4 Smaller or poorly trained language models sometimes run into issues of just repeating tokens (e.g. "The reliability of Wikipedia is very very"). Why might this happen?
This often happens when the model gets stuck in a loop where the most probable next token is the one it just generated.
* **Local Minima:** In the absence of a strong, broader context or "understanding," the statistical pattern of repeating a word might momentarily outweigh other options in a weak model.
* **Visualization:** Looking at the residual stream early in the model (e.g., Layer 1), the prediction might not have changed enough from the input token, leading to repetition.

### 7.5 Why is understanding and controlling language models using individual neurons problematic?
* **Polysemanticity:** Most neurons do not correspond to a single, clean concept. A neuron might fire for both "academic citations" and "pictures of birds" (or capital letters and acronyms).
* **Superposition:** Concepts are likely stored as linear combinations of neurons rather than in single neurons. Controlling one neuron affects all the unrelated concepts it participates in.

### 7.6 How do sparse autoencoders help us deal with polysemanticity?
Sparse Autoencoders (SAEs) act as a "dictionary learning" tool to disentangle the messy activations of neurons.
* **Expansion:** They map the smaller number of model neurons (e.g., 2,304) to a much larger number of "features" (e.g., 16,384).
* **Sparsity:** They are trained to use as few features as possible to reconstruct the original signal. This forces the SAE to find specific, monosemantic directions (features) that represent single concepts.
* **Result:** A single "feature" in the SAE often maps to a clear human concept (like "skepticism"), unlike the original neurons.

### 7.7 After training a sparse autoencoder, how can we figure out what concept a given feature corresponds to?
We identify a feature's meaning by **maximal activation search**:
1. Run a large dataset (e.g., millions of text snippets) through the model.
2. Record which text examples cause that specific feature to fire most strongly.
3. Analyze the common theme among those examples (e.g., if they all involve questioning or doubt, the feature likely represents "skepticism").

### 7.8 As language models become a larger part of our world, how important do you think explainability will be?
(Open-ended thought question, but based on the text:)
It is likely crucial for **safety, trust, and control**.
* **Lie Detection:** We can't tell if a model is lying just by asking it; we need to see its internal state (e.g., the "internal conflict" feature).
* **Steering:** We want to control behaviors (like reducing bias or increasing honesty) directly, rather than just hoping fine-tuning works.
* **Debugging:** Understanding *why* a model failed requires looking inside the black box.

### 7.9 The general approach of Mechanistic Interpretability has made some impressive progress in recent years, but is arguably still quite limited. Do you think approaches like the sparse autoencoder we saw in this chapter will be able to give us the level of control and understanding of LLMs we would want?
(Open-ended analysis:)
* **Promise:** SAEs have successfully extracted interpretable features (like "Golden Gate Bridge" or "Skepticism") and enabled steering.
* **Limitations:** We have only extracted a tiny fraction (<1%) of the estimated concepts ("dark matter"). Scaling up extraction is computationally expensive, and handling "cross-layer" superposition is unsolved.
* **Conclusion:** It's a powerful "telescope" for the brightest stars, but we may need fundamental breakthroughs to map the entire universe of LLM concepts.
