<!-- Generated from description.html; run course/scripts/sync-handouts.py after editing. -->

# Assignment 10: Let questions attend to story tokens

The previous assignment’s MLP connects every story-token position to a large classifier. Replace that flattened story input with **question-directed cross-attention**: the question selects a weighted combination of story-token values. Compare accuracy and parameter count; compact selection need not improve every answer.

## Modify your previous MLP solution

Keep **baseline.json** and its football evaluation separately. Open a copy of baseline.json and select the **Mixed-question classifier** dataset below. It matches the previous assignment.

For training and mixed evaluation: Select your Dataset block and open **Details → Browse datasets…**. Choose **Text and sequences** → **Football stories**. Choose the **Mixed-question classifier** preset. Click **Use this dataset**. Keep the supplied data and split unchanged.

For the required football evaluation copy: Select your Dataset block and open **Details → Browse datasets…**. Choose **Text and sequences** → **Football stories**. Choose the **Mixed training · football benchmark** preset. Click **Use this dataset**. Keep the supplied data and split unchanged.

**Prepared dataset properties**

| Dataset property | Supplied value |
| --- | --- |
| Vocabulary entries | 25, including unknown and padding |
| story token IDs | 32 |
| question token IDs | 8 |
| story word mask | 32 × 1 |
| question word mask | 8 × 1 |
| answer | 1 class index; 9 answer classes |
| Training examples in either QA preset | 3840 |
| Held-out examples — Mixed-question classifier | 1920 |
| Held-out examples — Mixed training · football benchmark | 240 |

Keep the shared **25 × 16 word embeddings**, **32 × 16 position table**, Dataset story/question IDs, word masks, and question encoder. The positioned, masked story has shape **32 × 16**; the **encoded masked question**, after its ReLU, has shape **1 × 16**.

Remove the reshape that flattened the story for the classifier. You will replace that input to Concatenate with an attended story row, and resize the first classifier matrix. Keep the 64-unit hidden layer, its bias, the nine-output layer, loss and output Softmax. Initialize the completed architecture afresh before training.

## Build question-directed attention

1. Compute **Q** from the encoded masked question with a learned **16 × 16** matrix. Compute **K** and **V** from the story’s token representations with two independent **16 × 16** matrices. Use no biases in these three projections. Their shapes are **1 × 16**, **32 × 16**, and **32 × 16**.
2. Transpose K with axes **1,0**. Use Matrix product for **QK<sup>T</sup>**, giving one score per story-token slot: **1 × 32**.
3. Transpose the story padding mask from **32 × 1** to **1 × 32**. Use Arithmetic `x1 / 4 - (1 - x2) * 1000000000`, with scores as x1 and this mask as x2. The divisor is √16, and the subtraction excludes padded tokens. The equivalent `x1 / 4 + (x2 - 1) * 1000000000` is also accepted.
4. Apply **Softmax** across the score row, then multiply those attention weights by V using Matrix product. This gives a **1 × 16** attended story row.

These weights select **tokens in the whole story**, rather than separately encoded sentences. Word positions come from the previous assignment’s learned position vectors. The attention weights are different from the answer probabilities.

## Reconnect the existing classifier

Concatenate the **attended story row first** and the **encoded masked question second**, on axis 1. Use the question encoder’s ReLU output before Q for that second input. Resize the first classifier matrix to accept the resulting **32 features**, keeping the 64-unit ReLU hidden layer and nine logits. Remove unused blocks/Params and retain the original answer target and separate output Softmax.

## Parameter-count hint

Both **attention.json** and its evaluation copy have **6,441 trainable parameters**, compared with the previous assignment’s 37,417. Count shared Params once.

## Train and save

First set **Training execution** to **Tensor engine · minibatches** to show the Tensor controls. Then follow the Train tab from top to bottom:

**Training settings — Train tab**

| Control | Value |
| --- | --- |
| Epochs per run | 5 |
| Report loss every | 5 |
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

Tensor CPU is the validated backend. You may try Auto or WebGL; speed depends on your computer and browser.

**Param → Initialize choices**

| Param block | Initialize |
| --- | --- |
| Word and position tables | uniform |
| Other matrices | Xavier · matrices |
| Bias vectors | zeros |

Apply changed shapes, then choose **Train → Randomize parameters**. Train from fresh parameters and save **attention.json**. The grading table below lists the targets. Longer training is optional; keep checkpoints because more training need not improve held-out accuracy. The grader checks saved quality, not claimed epochs.

## Required comparison: the same football questions

1. Keep attention.json on the **Mixed-question classifier** dataset. In a copy, open the dataset browser and choose **Text and sequences → Football stories → Mixed training · football benchmark**, then click **Use this dataset**, preserving every parameter and the architecture. Do not train or randomize the evaluation copy.
2. Run **Test → Held-out test set → Run inference**. Confirm **240 held-out examples** and read their accuracy. Compare it with the **240-question accuracy from the previous assignment**, together with the two parameter counts. The mixed 1,920-question accuracy measures a different question distribution.
3. Choose **File → Save as…** and save **football.json**. Submit attention.json and football.json. Keep both in a separate cross-attention folder so this football.json does not overwrite the previous assignment’s file.

The football evaluation retains all training examples and preprocessing, with 120 held-out questions for each moving/dropping order. There is **no minimum football accuracy**. The smaller model need not win every comparison. The whole story remains a token sequence, so selecting a word does not by itself guarantee that the model tracks every event correctly.

**Keep your cross-attention football.json for the final QA capstone.** You will submit it there as earlier.json and compare its score with your fine-tuned transformer on exactly these questions. Keep attention.json as the construction reference when attention returns in the self-attention assignment. You may also use the previous assignment’s optional whole-story/question experiment with this model; restore the original input before saving.

## Submit and grading

| File | Dataset | Model form | Training loss | Held-out loss | Accuracy |
| --- | --- | --- | --- | --- | --- |
| attention.json | 10 | 30 | 10: ≤ 0.80 | 10: ≤ 0.85 | 10: ≥ 60% |
| football.json | 5 | 15 | Reported without minimums |  |  |

Another **10 points** require identical learned parameters across the two files, for **100 points** total. The grader independently reconstructs canonical inputs and predictions and reports both football event orders. Performance credit requires the assigned model form. Higher accuracy is welcome; cached scores and claimed epochs are ignored. Held-out data is validation during development. No written comparison or screenshot is required.

Assignment 10-attention.
