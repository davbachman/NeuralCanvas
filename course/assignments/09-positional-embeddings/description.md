<!-- Generated from description.html; run course/scripts/sync-handouts.py after editing. -->

# Assignment 9: Positional embeddings and separate questions

The previous assignment represented an ordered story and its question with learned word embeddings. Now one story can lead to several questions: where either person is, where the football is, or who has it. Keep the story as **one continuous token sequence**, encode the question separately, and add **learned positional embeddings** to the story tokens. Your MLP will provide the baseline for question-directed attention in the next assignment.

## Start from your embedding MLP

Keep your saved **embedding.json** and work in a copy. Retain its embedding lookup, hidden-layer/output pattern, Cross entropy loss, and separate output Softmax. You will replace the data, resize the embeddings and classifier, and add a position table and question encoder. Initialize all parameters afresh after completing the changes.

## Data: a whole story and a separate question

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

The nine answer classes are **bathroom, bedroom, garden, hall, kitchen, liam, nobody, nora, office**.

The label **nobody** answers who has the football after it has been dropped. Whole room-layout groups and their actor/event-order variants stay in one split.

Each story has five events involving Liam, Nora and a football. Entering a room moves a person and any football they carry; taking picks up the football; dropping leaves it in that room. Sentence punctuation stays within the single story sequence. The question and answer are separate from that sequence.

The Dataset supplies these token IDs and masks in the shapes shown above. Connect the token IDs directly to their Embedding inputs; use the masks with the corresponding embeddings. A padding mask contains 1 for an observed token and 0 for padding. Padding reserves space; it is not an event.

## Add word positions

1. Resize the shared word-embedding table **E to 25 × 16**. Use that same Param for both story and question Embedding lookups. The story embeddings have shape **32 × 16**; the question embeddings have shape **8 × 16**.
2. Create a learned Param table **P of shape 32 × 16**. Row j is the positional embedding for story-token slot j. Add P directly to the story embeddings with Arithmetic addition. This gives every story token both word and position information.
3. Multiply the positioned story embeddings elementwise by the **32 × 1 story padding mask**. Broadcasting zeros every feature at a padded position. Apply the question’s **8 × 1 padding mask** to its embeddings as well. Use Arithmetic multiplication for these masks.

Flattening already keeps slots in order. The learned position vectors also put position information *inside* each token representation, which becomes useful when attention mixes tokens in the next assignment.

## Complete the question encoder and classifier independently

- Preserve all eight masked question embeddings in order as one row. Use a learned matrix, bias and ReLU to encode this row into **16 features**. Determine the matrix and bias shapes.
- Flatten the **32 positioned, masked story vectors** into one row. Append the encoded question using **Concatenate, axis 1**.
- Adapt the classifier to a **64-unit ReLU hidden layer** and **nine output logits**, with a matrix and bias in each layer. Determine the remaining shapes and remove unused blocks/Params.
- Connect the logits directly to **Cross entropy (logits)** with the original Dataset answer target. Keep a separate output **Softmax** branch.

<details>
<summary>Shape-check hint</summary>

Count the features in 32 story vectors of width 16, then include the question’s 16 features. The first classifier matrix must accept that total.

</details>

## Parameter-count hint

Count individual learned numbers, including embedding tables and biases, and count a shared Param only once. Both **baseline.json** and its evaluation copy have **37,417 trainable parameters**.

## Train from fresh parameters

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

Apply changed shapes, then choose **Train → Randomize parameters**. Train from fresh parameters and save **baseline.json**. Aim for training loss **≤ 0.75**. Five epochs is the intended starting budget; longer training is optional. The grader evaluates the saved model rather than claimed epochs. Held-out data is validation during development.

## Required: evaluate the common football benchmark (5 points)

**This 240-question accuracy is your baseline for comparison with the next cross-attention assignment and the final QA capstone.** The mixed 1,920-question score measures a different question distribution.

1. Save **baseline.json** with **Mixed-question classifier** still selected. Keep this original file for the next assignment.
2. In a copy, open the dataset browser and choose **Text and sequences → Football stories → Mixed training · football benchmark**, then click **Use this dataset**. Keep the architecture and every learned parameter fixed; do not train or randomize this evaluation copy.
3. Use **Test → Held-out test set → Run inference**. Confirm **240 held-out examples** and read their **accuracy**. These are the same football questions used in the word-count, embedding, cross-attention and final QA assignments, with 120 examples of each moving/dropping order.
4. Choose **File → Save as…** and save **football.json**. Submit both files. Reopen baseline.json to return to mixed questions or custom questions.

The evaluation dataset preserves all training questions and preprocessing. Its **5 points** require the correct evaluation data and model form, with parameters identical to baseline.json. There is no minimum football accuracy and no written response or screenshot.

## Optional: ask your model a question

Save baseline.json first. Select **Dataset → Details → Structured-story experiment**. Enter the complete paragraph in the **Story** field. Put the question in the separate **Question** field, without its answer:

```text
liam entered garden . nora entered kitchen . nora took football . nora dropped football . nora entered bedroom .
```

Try `where is football`, `where is nora`, `where is liam`, and `who has football`. After each edit, click **Apply experiment**, then **Train → Run forward**. Return to Dataset Details for the experimental prediction and nine answer probabilities. You can also swap the final moving and dropping events.

Use familiar wording, at most **32 story tokens** and **8 question tokens**, including punctuation. Unknown words map to `<unk>`. This changes the input for inspection, without training. The original answer remains only a loss reference; the app does not know the answer to your custom question. **Test → Run inference** evaluates the original dataset. Click **Restore original** before saving baseline.json again. This activity is optional and ungraded.

## Submission and grading

Submit **baseline.json** with the Mixed-question classifier dataset and **football.json** with the Mixed training · football benchmark dataset. The total is **100 points**:

| Requirement | Points |
| --- | --- |
| baseline.json: original dataset, vocabulary and split | 10 |
| baseline.json: assigned model form, including output Softmax | 60 |
| baseline.json: recomputed training loss ≤ 0.75 | 25 |
| football.json: valid evaluation copy with unchanged learned parameters | 5 |

The grader reconstructs inputs and predictions from saved operations and parameters. Performance credit requires the assigned model form. Held-out losses and accuracies are reported without a minimum. Use the **240-question football accuracy** for comparisons with the next assignment and the capstone.

Assignment 09-positional-embeddings.
