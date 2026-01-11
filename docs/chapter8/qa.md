# Chapter 8 - Attention | Exercises Q&A

These answers correspond to the exercises from "The Welch Labs Illustrated Guide to AI" (Chapter 8, pages 290-293).

### 8.1 Conceptually, why do attention layers use multiple attention heads?
Multiple heads allow the model to focus on different types of relationships simultaneously.
* **Specialization:** One head might learn to attend to the *previous token*, another to *adjectives modifying nouns*, and another to *related entities* later in the sentence.
* **Parallelism:** The model can "vote" on the next token's meaning from multiple perspectives at once.

### 8.2 How do attention heads capture/measure the similarity between key and query vectors?
They use the **dot product**.
* Mathematically: $Score = Q \cdot K^T$
* If a Query vector and a Key vector align (point in similar directions in high-dimensional space), their dot product is large.
* This large score (after softmax) leads to a high attention weight, meaning the model "pays attention" to that token.

### 8.3 How are keys and queries in attention layers different than keys and queries in information retrieval systems like databases? Do you think "key" and "query" are helpful names for these vectors?
* **Database (Exact):** A query matches a key exactly (e.g., `SELECT * WHERE ID=123`). It's a discrete, binary lookup.
* **Attention (Fuzzy):** Keys and Queries are dense vectors learned from data. "Matching" is a continuous measure of similarity (dot product), not an exact equality.
* **Names:** The names are a useful *analogy* for retrieval (looking up relevant info), but can be misleading because the "matching" is soft, probabilistic, and learned, rather than hard-coded.

### 8.4 As we saw in Chapter 3, the transformer architecture is composed of alternating fully connected/multi-layer perceptron layers and attention layers. Compare and contrast these different types of neural network layers.
* **Attention Layers:**
    * **Function:** Move information *between* tokens. They allow tokens to "talk" to each other (e.g., "blue" looking back at "sky").
    * **Scope:** Global view of the context window.
* **MLP Layers (Feed-Forward):**
    * **Function:** Process information *within* a single token. They think about the meaning of the current vector in isolation (e.g., transforming the concept "King" + "Woman" to "Queen").
    * **Scope:** Local view (token-wise).

### 8.5 Without caching, attention block compute grows quadratically with the number of input tokens—why?
Because for *every* new token generated, the model must re-calculate the interactions (dot products) between *all* previous tokens.
* If you have generated 100 tokens, the 101st generation requires re-processing the 100 previous positions to get their Keys and Values again.
* This scales as $O(N^2)$ because you process $N$ tokens for $N$ steps.

### 8.6 In KV caching, why don't we cache our query matrices? Why do we only store keys and values?
* **Queries change:** The query vector ($Q$) represents the *current* token asking for information. It is unique to the current step and doesn't need to be reused for future steps.
* **Keys/Values are static:** The Keys ($K$) and Values ($V$) for *past* tokens do not change once computed. We store them so the current Query can check against them without re-computing them from scratch.

### 8.7 Why would Multi-Query Attention (MQA) reduce algorithmic performance (e.g. worse error rate)?
MQA forces **all** attention heads to share the *same* Key and Value matrices.
* **Constraint:** This significantly limits the model's capacity. A head cannot "look" for different specific key features than another head; they all have to query the same database of keys.
* **Trade-off:** You gain massive memory savings (smaller cache) but lose the nuance of having 128 different "views" of the keys.

### 8.8 Here's a closer look at one of the attention patterns from Figure 8.1. What might this attention pattern be doing? (Figure A)

* **Observation:** The token "American" (input) is strongly attending to "flag" (output).
* **Interpretation:** This is likely a **modifier head**. It connects adjectives ("American") to the nouns they modify ("flag") to build a complex representation of the object "American Flag".

### 8.9 Here's a closer look at one of the attention patterns from Figure 8.1. What might this attention pattern be doing? (Figure B)

* **Observation:** The last token position (output) is attending to "flag", "red", and "white".
* **Interpretation:** This is likely a **context head** or **copying head** gathering relevant information to predict the next token. It's collecting the features of the current subject ("flag is red, white") to probably predict "and blue".

### 8.11 In Figures 8.14 and 8.15, we saw how Multihead Latent Attention (MLA) could be simplified... Show how we can combine $W_Q$ and $W_{UK}$ into a single weight matrix...
The simplification relies on matrix associativity: $(AB)C = A(BC)$.
1. **Original Score:** $Score = Q \cdot K^T$
2. **Definitions:** $Q = X W_Q$ and $K = L_{KV} W_{UK}$ (where $L_{KV}$ is the compressed latent).
3. **Substitution:** $Score = (X W_Q) \cdot (L_{KV} W_{UK})^T$
4. **Transpose Rule:** $(AB)^T = B^T A^T$. So, $(L_{KV} W_{UK})^T = W_{UK}^T L_{KV}^T$.
5. **Expand:** $Score = X W_Q W_{UK}^T L_{KV}^T$
6. **Combine:** We can pre-multiply $W_Q$ and $W_{UK}^T$ into a single matrix $W_{absorbed} = W_Q W_{UK}^T$.
7. **Result:** $Score = X \cdot W_{absorbed} \cdot L_{KV}^T$

We now perform one multiply instead of two during inference!

### 8.12 Show that we can combine $W_{UV}$ and $W_O$ into a single weight matrix...
Similar logic applies to the output projection.
1. **Layer Output:** $U_1 = O W_O$
2. **Head Output:** $O = A V$ (Attention matrix × Values)
3. **Value Definition:** $V = L_{KV} W_{UV}$
4. **Substitution:** $U_1 = (A L_{KV} W_{UV}) W_O$
5. **Associativity:** $U_1 = (A L_{KV}) (W_{UV} W_O)$
6. **Combine:** Pre-compute $W_{merged} = W_{UV} W_O$
7. **Result:** $U_1 = (A L_{KV}) W_{merged}$

Again, reducing computational steps during inference.
