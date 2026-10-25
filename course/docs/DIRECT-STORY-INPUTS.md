# Direct story/question inputs and clearer dataset descriptions

Numbering note: this dated report retains its original assignment numbers. See
[SAMPLING-SEQUENCE.md](SAMPLING-SEQUENCE.md) for the current sequence after moving sampling to Assignment 12.

Assignments 9 and 10 now use `representation: "story-question"`, supported by
app commit `871a4a8a7dc733e6296e4c1d6c017abb18eef297`. The prepared files keep
their shared names, `stories.json` and `football-test.json`; their bytes match
between assignments. Each document's `text` is the whole story, with a separate
`question`. The vocabulary, class order, question/answer text, document order and
room-layout split are preserved from the validated single-paragraph pilots.

The Dataset supplies these outputs directly:

| Slot | Output | Shape |
| --- | --- | --- |
| 0 | story token IDs | 32 |
| 1 | question token IDs | 8 |
| 2 | story word mask | 32 × 1 |
| 3 | question word mask | 8 × 1 |
| 4 | answer | 1 |

Six nontrainable input Reshape/Slice blocks are removed. Embedding/classifier
reshapes remain part of the model. Learned parameters, epoch counts, training
recipes, numerical targets and point allocations are preserved: Assignment 9
has 37,417 parameters and Assignment 10 has 6,441. Assignment 19's earlier-model
dataset and grader match the new Assignment 10 evaluation file. Its transformer
still receives the story and question together as a 23-token sequence and learns
to predict the answer as the next token.

## Handout review across 7–19

Dataset import is described in short prose naming the file and **Import text
dataset** action. Filename-only settings tables are removed. **Prepared dataset
properties** tables identify vocabulary size, input lengths/shapes, answer space
and split sizes needed for students' independent Param-shape decisions. Training
and initialization controls retain their existing tables.

Assignment 9's optional experiment now uses the actual **Story** and **Question**
fields. Assignment 10 retains that input path. Assignment 11 explicitly introduces
the switch to one sequence with position IDs; Assignment 13 distinguishes padding
masks from the causal mask. Assignment 19 explains its combined sequence while
keeping the earlier-model comparison on the same questions. The rest of the
handout review found no wording that required a conceptual change. Existing
shape/wiring challenges and submission requirements remain.

## Validation and reproduction

The [native app check](direct-story-inputs-native.json) validates every prepared
example and compares all 5,760 mixed-example feature/target encodings for each
architecture with the original inputs. On the full-size models, 32-example Tensor
CPU predictions, gradients and one Adam update have zero numerical difference.
This is an input-representation conversion, so full-budget training is retained
from the original three-seed pilots rather than repeated.

Independent grader checks cover all three seeds, full-credit reference models,
partial/malformed submissions, mask equivalences, parameter preservation and
extracted upload ZIPs. Browser checks exercise both 1,920-question and 240-question
inference, model saving/reimporting, five custom story/question cases and restore,
prepared-data imports, dataset properties, and phone/desktop handout layouts.
Historical full-budget runs retain their original app commits. Current native
and browser checks use the clean app commit above. These are local checks;
hosted Gradescope and publishing to course sites are outside this update.

To convert committed legacy references in place, run
`.venv/bin/python scripts/assignments/football/direct_story_inputs.py`.
The paragraph release builder also uses these conversion functions. It rebuilds
private references, so rerun UI checks after a full rebuild. Repackage with
`scripts/assignments/package-advanced.py` and run the affected grader suites.

The independent evaluator now recognizes both representations and verifies the
canonical `questionLength`. Its self-contained release copies are refreshed
throughout 9–19. The current dataset-filename audit continues to require identical
bytes for any reused student filename.
