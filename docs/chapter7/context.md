# Chapter 7 & 8 Context (Glass Box Lab)

## Overview
Phase 7 combines **Chapter 7 (Mechanistic Interpretability)** and **Chapter 8 (Attention)** into a unified "Glass Box" experience. While previous phases treated the model largely as a function mapping $X \to Y$, this phase allows users to peer inside the "black box" to understand *how* the mapping happens.

## Chapter 7: Mechanistic Interpretability
**"The Dark Matter of AI"**

This chapter focuses on reverse-engineering the internal computations of LLMs.

### The Residual Stream
The "main highway" of information in the Transformer. Token embeddings enter at the bottom, and layers read/write information to this stream.

*Lab Feature*: **Logit Lens**. By applying the final output decoder to intermediate layers, we can see what the model "believes" at Layer 6 vs Layer 20.

### Steering
Modifying the model's behavior by directly intervening in the residual stream (adding "feature vectors").

*Future Feature*: **Steering Sliders**. Users could inject a "sentiment" vector to make the model output more positive/negative without changing the prompt.

### Sparse Autoencoders (SAEs)
(Conceptual) A technique to disentangle "polysemantic" neurons (neurons that do 5 different things) into interpretable features.

**Key Exercises covered:**
* **7.2 - 7.6**: Visualizing residual stream evolution.
* **7.9**: Logit Lens — seeing how predictions refine through layers.

## Chapter 8: Attention
**"How DeepSeek Rewrote the Transformer"**

This chapter details the Attention mechanism and optimizations for massive context.

### Attention Patterns
The $N \times N$ heatmap showing how tokens "vote" for information from other tokens.

*Lab Feature*: **Attention Viewer**. Interactive visualization of attention heads across all layers. Hover to see which tokens attend to which.

### Head Types
Different heads learn different roles:
- **Previous Token Heads**: Copy info from the prior position.
- **Induction Heads**: Pattern-match and copy (e.g., "Harry... Harry Potter").
- **Modifier Heads**: Connect adjectives to nouns.

### DeepSeek MLA (Multi-Head Latent Attention)
A specialized architecture to drastically reduce the **Key-Value (KV) Cache** size.

- *The Problem*: Standard Attention requires storing massive K/V matrices for every token in history.
- *The Solution*: MLA compresses K/V into a low-rank latent vector.
- *Lab Feature*: **KV Cache Calculator**. Interactive comparison of MHA, MQA, GQA, and MLA memory usage.

**Key Exercises covered:**
* **8.8 - 8.9**: Interpreting specific attention patterns (looking back, modifying).
* **8.11 - 8.12**: Matrix math efficiency (visualized via the Calculator).

## Technical Implementation Strategy
We use the **HuggingFace GPT-2** model already loaded in `backend/services/transformer_service.py`.

1. **Model**: GPT-2 Small (124M params, 12 layers, 12 heads). Sufficient for demonstrating attention heads and residual streams while remaining responsive.

2. **PyTorch Hooks**: Configure the model with:
   - `output_attentions=True`: Captures attention weights at each layer.
   - `output_hidden_states=True`: Captures the residual stream at each layer.

3. **Logit Lens Implementation**:
   - Apply the final `lm_head` (unembedding) to hidden states at each layer.
   - Show top-k token predictions per layer for the final token position.

4. **Visualization**:
   - **Attention Heatmaps**: Canvas-based rendering for efficient display of N×N attention matrices.
   - **Logit Lens Table**: Layer-by-layer view of top predicted tokens.
   - **KV Cache Calculator**: Interactive form with bar chart comparison.

## Educational Value
This phase answers fundamental questions:
- "How do transformers work internally?" → Attention patterns show information flow.
- "Where does the model decide?" → Logit Lens shows prediction refinement.
- "Why is DeepSeek efficient?" → KV Cache comparison shows memory savings.
