<!-- Generated from description.html; run course/scripts/sync-handouts.py after editing. -->

# Assignment 14: Share attention across word positions

**Your starting point and handoff**

| Step | What to use |
| --- | --- |
| Start with | A copy of your saved **context.json**; keep the cross-attention assignment’s attention.json nearby as a reference |
| Modify | Keep the Dataset, token embeddings, loss and output Softmax. Replace the flattened context MLP with self-attention, add learned positions and a causal mask, and resize the vocabulary projection. |
| Save and keep | **self-attention.json**; use a copy in the next assignment. |

The context MLP allocates separate input weights to every slot. We will instead apply the same learned projections at each position and let tokens select earlier context. In the cross-attention assignment, the separate question supplied one query for the story. Here, every token supplies a query, and queries, keys and values all come from the same sequence.

Use the narrative configuration selected below in your Dataset block. It keeps the same football narratives, **25-token vocabulary**, **32-token input windows** and training/held-out split. There is one target after the final input token. Keep the supplied data and split unchanged. Held-out data is validation during development.

Select your Dataset block and open **Details → Browse datasets…**. Choose **Text and sequences** → **Football stories**. Choose the **Narrative generation** preset. Set **Training targets** to **Final position only**. Click **Use this dataset**. Keep the supplied data and split unchanged.

**Prepared dataset properties**

| Dataset property | Supplied value |
| --- | --- |
| Vocabulary entries | 25, including unknown and padding |
| Input token IDs | 32 |
| Position IDs | 32 |
| Next-token target IDs | 1 |
| Training windows | 1534 |
| Held-out windows | 765 |

## Build shared self-attention

Add **16-dimensional token embeddings** and **16-dimensional learned position embeddings**, using the Dataset’s token and position IDs. As in the story/question model, use a separate **Embedding** block for each lookup and add their outputs. Work out the two table shapes. Call their sum X. Create independent **16 × 16** matrices for Q, K and V and compute each projection from X.

Compute **QK<sup>T</sup>/4**. Add **Build → Sequences and attention → Causal mask** between the scaled scores and their Softmax. It allows each position to use itself and earlier positions. The word masks in the cross-attention assignment excluded padding; this causal mask excludes future positions. Apply Softmax across each row, then multiply those attention weights by V.

## Finish independently

Keep the resulting 16-feature row at every token position. Apply one shared vocabulary matrix and bias to all rows. Slice only the **final row of logits** for the loss and the separate output Softmax. Determine the Slice bounds from the context length. Do not add a hidden MLP yet.

Save **self-attention.json**. The benefits here are shared parameters and contextual representations at every position; lower loss than the context MLP is not required.

## Parameter-count hint

Your completed models should have the following counts. Count individual learned numbers, including embedding tables and biases, and count each shared Param only once.

- **self-attention.json:** **2,105** trainable parameters.

## Train and save

First set **Training execution** to **Tensor engine · minibatches** to show the Tensor controls. Then follow the Train tab from top to bottom:

**Training settings — Train tab**

| Control | Value |
| --- | --- |
| Epochs per run | 5 |
| Report loss every | 1 |
| Examples per update | 32 |
| Reshuffle training examples each epoch | Checked |
| Exact learning rate | 0.003 |
| Training execution | Tensor engine · minibatches |
| Backend | Tensor CPU |
| Optimizer | Adam |
| Gradient norm limit (0 = off) | 1 |
| Early stopping patience (0 = off) | 0 |

**Loss block → Details → Parameter regularization**

| Control | Value |
| --- | --- |
| Penalty | None |

Tensor CPU is the validated backend. Auto or WebGL may be used; speed depends on the computer and browser. Use the same backend for timed comparisons.

**Param → Initialize choices**

| Param block | Initialize |
| --- | --- |
| Word and position tables | uniform |
| Other matrices | Xavier · matrices |
| Bias vectors | zeros |

Apply changed shapes, check the saved Initialize choices, then use **Train → Randomize parameters**. Remove unused Params after an architecture change. Each changed architecture starts fresh; evaluation copies retain their trained values.

## Optional: generate a continuation

In **Dataset → Details → Continue the text**, paste this prompt and try **Most likely** or **Sample → Generate 40 tokens**:

> nora entered bedroom . liam entered kitchen . nora took football . nora entered hall . nora dropped football . now the football is in the hall . end nora entered bedroom . liam entered kitchen . nora took football . nora dropped football . nora entered hall . now

The prompt is longer than 32 tokens, so the app uses its last 32. Short prompts need padding and may behave differently. The temperature and sampling controls introduced earlier apply to every generator in this sequence. Use the same prompt to compare Most likely with Sample at temperatures 0.5, 1 and 2, keeping Top-k at 25. Sampling changes token selection without training. A fluent continuation can still misstate a person’s location, the football’s location or who holds it. This inspection is optional; the submitted model itself is the graded artifact.

## Submit and grading

Points below are dataset / model form / training loss / held-out loss. Loss limits are upper bounds. Performance credit requires the assigned architecture. The grader recomputes results from saved parameters; cached scores and declared epoch counts are ignored.

| File | Points | Training loss ≤ | Held-out loss ≤ |
| --- | --- | --- | --- |
| self-attention.json | 10 / 50 / 20 / 20 | 1.35 | 1.4 |

Accuracy, parameter counts, common final-position loss and generated continuations are informational. No required written answers, screenshots or unsubmitted investigations.

Assignment 14-self-attention.
