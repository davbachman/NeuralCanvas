<!-- Generated from description.html; run course/scripts/sync-handouts.py after editing. -->

# Assignment 19: Fine-tune your generator to answer questions

**Your starting point and handoff**

| Step | What to use |
| --- | --- |
| Start with | Your trained **multihead.json** generator and earlier cross-attention **football.json** |
| Modify | Preserve the generator’s weights, select the QA dataset in a copy, and change the output Slice to predict only the answer. Fine-tune a copy of that baseline. |
| Save and keep | **earlier.json**, **pretrained.json**, **before.json**, **after.json**. |

Complete the course sequence by adapting **your trained generator from the previous assignment**. Keep its learned parameters and continue training with an answer-prediction objective. Compare it with your earlier cross-attention model on the same football questions.

For the QA copy in Part A (after saving pretrained.json): Select your Dataset block and open **Details → Browse datasets…**. Choose **Text and sequences** → **Football stories**. Choose the **Answer-token fine-tuning** preset. Click **Use this dataset**. Keep the supplied data and split unchanged.

**Prepared dataset properties**

| Dataset property | Supplied value |
| --- | --- |
| Vocabulary entries | 25, including unknown and padding |
| Input token IDs | 23: story followed by question |
| Next-token target IDs | 1: the answer, excluded from the input |
| Training questions | 3840 |
| Held-out football questions | 240 |

## The new objective

Use the **Answer-token fine-tuning** preset in the dataset browser. It uses the same five-event stories, vocabulary and room-layout split as pretraining. Here the story and question form **one token sequence**, rather than the separate inputs used in the earlier MLP and cross-attention assignments. The **answer is the next token**.

> liam entered garden . nora entered kitchen . nora took football . nora dropped football . nora entered bedroom . where is football<br>**Next-token target:** kitchen

The prepared input has **23 tokens** and one answer target. The answer is excluded from the input. Training contains **3,840 questions** about people, football location and possession. The **240 held-out comparison questions** all ask where the football is: half dropping before moving, half moving before dropping. Room-layout groups and their actor/event-order variants stay in one split.

## Part A — Change the supervision

1. Choose your final pretraining checkpoint from the previous assignment, including any optional extra narrative training—your selected **multihead.json**—and save a copy as **pretrained.json** for this submission. In a separate copy, open **Dataset → Details → Browse datasets…**, choose **Text and sequences → Football stories → Answer-token fine-tuning**, and click **Use this dataset**.
2. Preserve every learned parameter, including the **32 × 16** position table. The shorter input uses its first 23 rows. No Param needs resizing.
3. Adapt the final vocabulary-logit output so only the **final input position** feeds Cross entropy and the separate probability Softmax. Determine the **Slice** axis, start and exclusive end from the 23-row output. The resulting logits and probabilities should each be **1 × 25**. Keep the Dataset's original target connected to the loss.
4. Run held-out inference and save **before.json** before training on questions. There is no answer-accuracy target for this baseline. The grader checks that its parameters match your submitted, trained pretraining model.

## Part B — Fine-tune on answers

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
| Backend | WebGL |
| Optimizer | Adam |
| Gradient norm limit (0 = off) | 1 |
| Early stopping patience (0 = off) | 0 |

**Loss block → Details → Parameter regularization**

| Control | Value |
| --- | --- |
| Penalty | None |

Reopen **before.json** and run five additional QA epochs using these settings, preserving the pretrained values. Fine-tuning updates all learned parameters. Save the result as **after.json**, keeping before.json unchanged. The grading table below lists the quality targets. Save a checkpoint before optional extra training.

The epoch counter may continue from pretraining; five *additional* epochs is the requested run. If interrupted, reopen before.json and repeat the five-epoch run. WebGL uses a supported GPU when available; Tensor CPU is a fallback and may take longer. Runtime varies and is not graded.

## Optional: things to try

Keep the five-epoch result. For each experiment, **reopen the unchanged before.json** to keep the starting weights and epoch counter comparable. Keep the remaining controls at the values in the default training and regularization tables. These experiments are optional; no extra submission or written comparison is required.

**Optional Adam comparison — changed Train controls**

| Control | Value |
| --- | --- |
| Epochs per run | 10 |
| Report loss every | 1 |
| Exact learning rate | 0.003 |
| Optimizer | Adam |

**Optional AdamW experiment — changed Train controls**

| Control | Value |
| --- | --- |
| Epochs per run | 10 |
| Report loss every | 1 |
| Exact learning rate | 0.003 |
| Optimizer | AdamW |
| Weight decay | 0.1 |

The AdamW **Weight decay** control appears when that optimizer is selected. Keep **Penalty** at **None** in the Loss block’s Details. Compare the two runs, then optionally change one setting from the AdamW experiment:

**Optional AdamW variations — changed Train controls**

| Control | Smaller decay | Smaller learning rate |
| --- | --- | --- |
| Exact learning rate | 0.003 | 0.001 |
| Weight decay | 0.01 | 0.1 |

Keep ten continuous QA epochs and the other settings fixed. Compare held-out loss and answer accuracy; training loss alone does not establish a better model.

Save each result as a separate checkpoint and choose your best **after.json** meeting the grading targets. Optimizer and learning-rate changes can help some initializations and hurt others. If you prefer to extend a saved QA result, additional **5-epoch** runs are also allowed; save a checkpoint first. Adam and AdamW start fresh running gradient averages each time you click Run, so two five-epoch runs can differ from one continuous ten-epoch run. More training can improve accuracy, leave it unchanged or make it worse.

Keep **pretrained.json** and **before.json** unchanged during QA training. Their learned parameters must match exactly. If you choose a different pretraining checkpoint, use that updated model as pretrained.json, recreate its unchanged-weight QA baseline as before.json, and fine-tune again. Gradescope evaluates the saved models on the same fixed examples; training time, epoch counts and optimizer choices are not graded, and extra training does not increase grading time.

## Part C — Compare with your earlier attention model

Submit a copy of **football.json** from your earlier cross-attention model as **earlier.json**, preserving its dataset and trained parameters. It uses the same training questions and the same 240 held-out cases as separate story and question inputs. The grader reports both models' results. There is no penalty for an unusually strong earlier model and no score based only on a difference between two submitted scores.

The transformer has also received narrative pretraining. This comparison assesses the complete course workflows; it does not establish that pretraining is necessary or beats training a transformer from scratch. Held-out data is validation during development.

## Parameter-count hint

**earlier.json:** 6,441 parameters using whole-story cross-attention and a concatenation head. **pretrained.json, before.json and after.json:** 5,689 each.

## Optional: ask a question directly

After fine-tuning, use **Dataset → Details → Continue the text**, enter the example without its answer, choose **Most likely**, and generate one token. Try `where is nora` or `who has football` with the same story. Keep the supplied names, rooms and wording. You can also compare with **Sample** using the temperature controls introduced in the sampling assignment. The grader evaluates answer probabilities directly; sampled answers do not determine accuracy. Direct generation is optional; passes, longer stories and arbitrary wording are outside this exercise.

## Submit — four files, 100 points

| File | Data | Model form | Training loss | Held-out loss | Accuracy | Total |
| --- | --- | --- | --- | --- | --- | --- |
| earlier.json | 5 | 10 | Informational | Informational | Informational | 15 |
| pretrained.json | 5 | 10 | 5 (≤ 0.60) | 5 (≤ 0.65) | Informational | 25 |
| before.json | 5 | 10 | Informational | Informational | Informational | 15 |
| after.json | 5 | 10 | 5 (≤ 0.75) | 5 (≤ 0.75) | 10 (≥ 70% overall) | 35 |

Another **10 points** verify identical learned parameters in pretrained.json and before.json. The grader verifies canonical data and model structure, then recomputes predictions. Cached metrics and declared epochs do not prove a training history. No written answers, screenshots or separate quiz are required.
