# Whole-story and question datasets

[Text curriculum](TEXT-CURRICULUM.md) · [Blocks](BLOCKS.md) · [Saving](FILES-AND-EXPORT.md)

Use `"representation": "story-question"` in a prepared text dataset JSON to supply a whole story and a separate question. Import it through the Dataset block's text importer. The file selects this representation, lengths, tokenizer, vocabulary, class order, and splits automatically; the import form needs no new controls.

## Prepared JSON schema

This is a complete, valid example:

```json
{
  "version": 1,
  "fileName": "stories.json",
  "task": "classification",
  "tokenizer": "word",
  "representation": "story-question",
  "lowercase": true,
  "vocabulary": ["<unk>", "<pad>", "mary", "john", "kitchen", "garden", "where", "?"],
  "classLabels": ["kitchen", "garden"],
  "maxLength": 32,
  "questionLength": 8,
  "stride": 32,
  "documents": [
    {"text": "Mary kitchen", "question": "where Mary ?", "label": "kitchen", "split": "train"},
    {"text": "John garden", "question": "where John ?", "label": "garden", "split": "train"},
    {"text": "Mary garden", "question": "where Mary ?", "label": "garden", "split": "test"}
  ]
}
```

The required fields and validation rules are:

| Field | Rule |
| --- | --- |
| `version` | `1`, using the existing prepared-text/project format. |
| `fileName` | String of at most 256 characters. |
| `task` | Must be `"classification"`. |
| `representation` | `"story-question"`. |
| `tokenizer` | `"word"` or `"character"`; word mode treats punctuation as tokens. |
| `lowercase` | Boolean, applied to both story and question. |
| `maxLength` | Integer 1–256: padded story length. |
| `questionLength` | Integer 1–256: padded question length, independent of `maxLength`. |
| `stride` | Existing recipe field, an integer from 1 to `maxLength`; classification uses one example per document rather than sliding windows. |
| `vocabulary` | 2–8,192 unique, nonempty strings, at most 1,024 characters each; ID 0 must be `<unk>` and ID 1 must be `<pad>`. Array order defines token IDs. |
| `classLabels` | At least two unique, nonblank strings. Array order defines answer IDs. Every document label must occur here. |
| `documents` | 2–60,000 documents with nonempty `text` and `question` strings, a `label`, and `split` equal to `"train"` or `"test"`. Both splits must be present. The existing 80-million-character story-text limit applies. |

Tokenization must produce at least one token in each story and question. Each is checked against its own limit. An overlong story or question is rejected; it is never silently truncated. Inputs are always padded in this representation, so `fixedLength` is not needed. There are no `facts`, `factWords`, or `maxFacts` requirements.

Prepared JSON preserves the supplied vocabulary, class labels, and their IDs without refitting. When creating data with `prepareTextDocuments`, use `representation: 'story-question'`, `maxLength`, and `questionLength`. Vocabulary fitting includes **both stories and questions from training documents only**. Class fitting also uses training documents only, and unseen held-out words use ID 0; held-out labels outside the fitted classes are rejected.

## Dataset outputs

For `maxLength = S` and `questionLength = Q`, the exact zero-based output order is:

| Slot | Label | Single-example shape | Tensor minibatch shape |
| --- | --- | --- | --- |
| 0 | story token IDs | `[S]` | `[batch, S]` |
| 1 | question token IDs | `[Q]` | `[batch, Q]` |
| 2 | story position IDs | `[S]` | `[batch, S]` |
| 3 | story word mask | `[S, 1]` | `[batch, S, 1]` |
| 4 | question word mask | `[Q, 1]` | `[batch, Q, 1]` |
| 5 | answer | `[1]` | `[batch, 1]` |

Padding uses token ID 1 and mask value 0. Real tokens, including unknown words encoded as ID 0, have mask value 1. The answer is the index into `classLabels`. Story position IDs are `0, 1, …, S−1` for every example, including padded slots. No fact-level mask is emitted. Story and question token positions keep their own indices in Data inspection.

Connect each token array to Embedding's token input. Multiply its embedding output by the corresponding word mask before pooling or combining inputs. Attention over story words must use the story mask to exclude padding. For learned positions, connect a position table of shape `[S, D]` and the story position IDs to a separate Embedding block. Add its `[S, D]` output to the story word embeddings, then apply the story mask so padded positions contribute nothing.

For the example above, the first story begins `[2, 4, 1, 1, ...]`, its mask begins `[1, 1, 0, 0, ...]`, the question begins `[6, 2, 7, 1, ...]`, and its mask begins `[1, 1, 1, 0, ...]`. The answer is `[0]`.

## Inspect and experiment

Dataset Details shows **Story** and **Question** with token lists padded to their respective lengths. The structured-story experiment editor lets students change either field. Unknown words are listed; invalid lengths disable **Apply experiment**. Applying recomputes both token arrays and both masks using the frozen vocabulary. **Run forward**, Softmax, attention inspection, and sidebar navigation retain the experiment and selected dataset row.

Edits are temporary inference inputs. Single-example parameter updates are disabled until restoring the original. **Run epochs** uses original training documents, and held-out inference uses original held-out documents. The original answer is identified as a reference: an edited story/question can invalidate it, so the app reports experimental predictions and probabilities without treating that answer as experimental correctness.

**Restore original**, selecting another row, switching splits, or replacing the dataset clears applied overrides. Save/import and local recovery can retain a current temporary experiment, without replacing canonical documents or changing learned parameters.

## Compatibility and converting a model

Existing `facts`, `tokens`, and `counts` datasets and saved projects keep their behavior. A facts model that previously stored a whole story as one fact may reshape `[1, S]` into `[S]` and slice/reshape its shared-length question down to `[Q]`. The new representation supplies those inputs directly. Use the story position IDs at slot 2 with a position-table Embedding lookup. Rewire the corresponding masks to slots 3 and 4 and the answer to slot 5; remove the obsolete input reshapes and question slices. Preserve the vocabulary array, class order, and parameter values during this conversion.

The regression fixture compares such models with story/question lengths 32/8. Their intermediate embeddings, attention, logits, Softmax probabilities, and held-out predictions are identical with unchanged parameters. Separate fixtures exercise other lengths, exact masks, malformed imports, editing/restoration, project round trips, and Tensor minibatch gradients and training.
