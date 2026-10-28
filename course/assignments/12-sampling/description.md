<!-- Generated from description.html; run course/scripts/sync-handouts.py after editing. -->

# Assignment 12: Control how the model chooses words

**Your starting point and handoff**

| Step | What to use |
| --- | --- |
| Start with | Your trained **bigram.json** from the previous assignment |
| Modify | Save base.json, then construct temperature, top-k, and combined probability branches in a copy. Preserve every learned parameter; no new training. |
| Save and keep | **base.json** and **sampling.json**. Use base.json to build the context model in the next assignment. |

The highest-probability next token is not the only possible continuation. Temperature changes how concentrated the distribution is. Keep your trained bigram model as **base.json**. Do not retrain or randomize it.

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

Training learns the next-token probabilities. **Most likely** always chooses a token with the highest probability; **Sample** draws a token from the distribution. These are generation choices you can use with every generative model that follows.

## Part A — Temperature without training

In a copy of your trained bigram, keep the original logits, loss and output Softmax. Add two independent probability branches at **T = 0.5** and **T = 2**. Each computes **Softmax(logits / T)** from the original vocabulary logits. Work out the Arithmetic expressions and wiring yourself. Keep these full-vocabulary branches for comparison.

## Part B — Select candidates with top-k

Temperature changes relative probabilities. Top-k restricts which tokens can be chosen. Build the restriction yourself using **Argsort**, **Tensor transform → Slice**, and **Index mask**; changing the generation dialog alone does not satisfy this task.

- **Argsort** returns indices along the final axis, not the sorted values. In Details, use **Sort order → Descending**. Equal values keep the lower original index first; excluded entries sort last.
- **Slice** selects a range of those indices; its end is exclusive. Determine the correct axis and bounds from your tensor shapes.
- **Index mask** takes values at its first port and integer indices at its second. **Keep selected** retains those positions; **Exclude selected** removes them. The tensor keeps its original shape and token positions. A following Softmax assigns exactly zero probability to excluded entries and normalizes over the rest.

For example, Argsort of scores \[0.4, 2.0, 0.1\] returns indices \[1, 0, 2\]. Keeping only its first index should give a probability vector \[0, 1, 0\], still indexed by the original three tokens.

Padding is never a generated token. Use a fixed **Input** tensor of shape **\[1\]** containing the padding token ID **1**, and an **Index mask** in **Exclude selected** mode to remove padding **before ranking**. Use the resulting ranked indices to construct separate top-k distributions with **k = 1** and **k = 5** at **T = 1**. Preserve the full 25-token output shape; mask unwanted logits rather than shortening the vocabulary logits or setting them to zero. Keep the index construction connected to the model's logits, so it adapts to each input.

## Part C — Combine both controls

Independently extend your construction to produce **k = 5, T = 0.5** and **k = 5, T = 2** distributions. Decide which ranking and selection blocks can be shared. Every final selection must mask the correctly scaled original logits before Softmax. Keep all six required probability branches in **sampling.json**:

| Experiment | Temperature | Candidate restriction |
| --- | --- | --- |
| Sharper full-vocabulary probabilities | 0.5 | None |
| Flatter full-vocabulary probabilities | 2 | None |
| Single candidate | 1 | Top 1, excluding padding |
| Five candidates | 1 | Top 5, excluding padding |
| Sharper restricted probabilities | 0.5 | Top 5, excluding padding |
| Flatter restricted probabilities | 2 | Top 5, excluding padding |

Use different input examples to inspect which entries remain nonzero, how the probability mass changes, and what happens when k = 1. The saved constructions are the graded artifacts; no written answers or screenshots are required. The grader evaluates every vocabulary-token context, so a fixed list of favorite tokens is insufficient.

**Do not retrain or randomize.** The original Loss still receives unscaled, unmasked logits. These are parallel probability branches, not new training objectives. Save the original unchanged model as **base.json** and the six-branch construction as **sampling.json**; all learned parameters must match.

## Optional generation

In **Dataset → Details → Continue the text**, use the same prompt for each comparison:

**Optional generation controls**

| Control | Value |
| --- | --- |
| Prompt and generated text | Your chosen prompt; restore it before each comparison |
| Decoding | Sample |
| Temperature | The value being explored |
| Top-k | 25 |

Choose **Sample** under **Decoding** to show Temperature and Top-k. Compare separately with **Most likely**. These controls govern generation; the extra canvas branches let you inspect the mathematics and do not set the generation controls. The same generation controls work with every later generative model. Try Top-k 1 and 5 with temperatures 0.5, 1 and 2, resetting the same prompt each time. Compare variety and repetition; the bigram model still sees only the last token, so sampling cannot give it a longer memory. The two full-vocabulary temperature branches include padding for comparison. Your top-k branches and actual generation exclude padding before selecting candidates; compare those branches with matching generation settings. The required artifacts are base.json and sampling.json; no saved file proves which generation buttons you clicked.

## Parameter-count hint

Your completed models should have the following counts. Count individual learned numbers, including embedding tables and biases, and count each shared Param only once.

- **base.json:** **825** trainable parameters.
- **sampling.json:** **825** trainable parameters.

## Submit and grading

Points below are dataset / model form / training loss / held-out loss. Loss limits are upper bounds. Performance credit requires the assigned architecture. The grader recomputes results from saved parameters; cached scores and declared epoch counts are ignored.

| File | Points | Training loss ≤ | Held-out loss ≤ |
| --- | --- | --- | --- |
| base.json | 5 / 5 / 5 / 5 | 1.15 | 1.2 |
| sampling.json | 5 / 5 / 5 / 5 | 1.15 | 1.2 |

The two temperature branches earn **5 points each**. The k = 1 and k = 5 branches at T = 1 earn **10 points each**. The two combined k = 5 branches earn **10 points each**. These 50 construction points are graded independently. An additional **10 points** require identical learned parameters in base.json and sampling.json. Total: **100 points**.

Accuracy, parameter counts, common final-position loss and generated continuations are informational. No required written answers, screenshots or unsubmitted investigations.

Assignment 12-sampling.
