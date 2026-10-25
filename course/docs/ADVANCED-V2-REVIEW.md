# Assignments 9–20: revised trajectory and release review

This report preserves the sequence and evidence from before the stacking revision. Its release labels, archive paths and reproduction scripts describe that historical snapshot, available in Git history at `922605e`. Unpublished draft archives and obsolete scripts have since been removed; they are not alternate current assignments. For current numbering, releases and fresh two-block validation, see [Stacking revision](STACKING-REVISION.md). Historical measurements are not new two-block results.

Revised 2026-10-19. All twelve assignments now have v2 handouts and matching autograders. **Assignments 1–8 and all issued dataset bytes are unchanged.** Earlier releases are archived within each assignment. This review records the full-sequence check, not a request to change the earlier assignments.

## Learning sequence

| Day | Limitation or next need | New tool / lecture preparation | Independent work assessed in the saved artifact |
|---|---|---|---|
| 9 | The short single-person template must extend to multi-person, variable-length stories. | Structured inputs, padding masks, shared fact encoders, learned fact positions. Demonstrate the fact encoder. | Complete the question encoder, fact masking and classifier dimensions in baseline.json. Held-out accuracy is informational. |
| 10 | Fixed flattening does not explicitly let the question select facts. | Query/key/value cross-attention and score masking. Demonstrate the scoring rule. | Build retrieval and replace the readout in attention.json; reach the existing 90% accuracy target. |
| 11 | A plausible attention display alone does not establish what affects an answer. | Controlled interventions with fixed model parameters. Demonstrate the question-only change. | Design two constrained location changes; submit question.json, irrelevant.json and relevant.json with identical weights. |
| 12 | Classification must become repeated next-token prediction. | Character targets, Slice, and autoregressive feedback. Demonstrate generation in lecture. | Build and size the final-character embedding/linear predictor in bigram.json. |
| 13 | A single preceding character cannot distinguish longer contexts. | Reuse ordered embeddings from Assignment 8. | Independently extend the bigram to an eight-character, 64-hidden-unit MLP in context.json. |
| 14 | A fixed flattened MLP does not provide shared context selection at every position. | Transfer cross-attention to self-attention; introduce causal masking and token positions. | Derive matrix/score shapes and build attention.json with a final-row objective. |
| 15 | The model needs an identity path plus nonlinear feature updates. | Residual addition and a shared positionwise feed-forward network. Guide the first residual. | Size and wire the second update and shortcut in residual.json. |
| 16 | Intermediate feature scale changes across transformations. | Layer norm ports, feature-wise statistics and pre-normalization. | Place three independent normalizations while preserving shortcuts in transformer.json. |
| 17 | Only one of eight prediction positions supplies training supervision. | Shifted targets and comparable evaluation objectives. | Adapt the objective, then submit all_positions.json and an unchanged-weight last_position.json evaluation copy. |
| 18 | One head gives one attention pattern per token. | Divide a fixed total projection width between independent heads. | Derive each head’s dimensions, score scale and concatenation in multihead.json. |
| 19 | Greedy selection and one probability concentration limit decoding choices. | Temperature, sampling, and the distinction between logits and probabilities. | Add independently derived probability scaling to sampling.json, paired with unchanged base.json. |
| 20 | Models trained on different objectives need a common evaluation task. | No new concept; reuse the evaluation-copy method from 17. Reserve class for discussion. | Submit the four earlier models and a new, unchanged-weight multihead_last.json copy. |

Days 9–10 and 12–13 are paired problem/improvement sequences. The construction of a transformer unfolds over Days 14–18 rather than pretending each component must independently beat the preceding model’s validation loss. The existing pilots support large gains for question-directed attention and additional character context. Normalization, multiple heads and temperature have explicit structural or distributional objectives; no unsupported accuracy improvement is promised.

The new requirement before Day 10—structured masks and shared encoders—is taught in Day 9. Position IDs are introduced there before being reused in 14. Residuals precede normalization, causal masking precedes all-position supervision, and the evaluation-copy method precedes final synthesis. Each handout says which prior file to reuse and whether to initialize a fresh model or preserve weights. No preparation gap requires changing Assignments 1–8.

## Assessment changes

- Familiar embeddings, MLPs, logits, probability branches and shape arithmetic are increasingly specified as constraints instead of complete wiring recipes. New block controls remain explicit.
- Every substantial required task produces a saved model checked by the grader. Additional generation, inspection and discussion preparation are labeled optional. No quiz, prose grading, self-reported score or claimed button history is introduced.
- Day 11 accepts student-chosen replacement locations while enforcing a single controlled change. It derives the expected edited answer from the actual submission. Incorrect predictions on valid interventions are informative, not penalized separately.
- Day 19 accepts equivalent constant multiplication/division and independently checks the two required temperature distributions. It requires a base copy and verifies that learned weights did not change.
- Day 20 requires the fair-evaluation copy that was previously temporary and unsubmitted. Only the multihead original/evaluation pair must match parameters; the four distinct architectures must not be forced to share weights.
- Numerical recipes, canonical datasets and existing loss/accuracy targets are retained. There is no score based on weakening a comparison model. Epoch counts are an experiment protocol, not trusted evidence of training history.

## Release and migration

All twelve current ZIPs are versioned v2; each previous v1 release is retained in its assignment’s archive. Publish each handout with its matching ZIP. Dataset files are unchanged. The new file requirements are:

| Assignment | Change |
|---|---|
| 11 | Same three filenames; irrelevant/relevant edits now allow a student-chosen location under the stated constraints. |
| 19 | Add base.json alongside sampling.json. Forty points assess the base, fifty the transformed model, ten unchanged parameters. |
| 20 | Add multihead_last.json to the four original files. Three original final-position models earn 20 each, original multihead earns 10, its evaluation copy earns 20, and parameter preservation earns 10. |

No other submission filename changes. A Git push does not update Canvas or Gradescope.

## Verification and limits

`advanced-v2-audit-baseline.json` retains hashes for all protected Assignment 1–8 files and the previous 9–20 release. `advanced-v2-audit.json` records final preservation, packaging, submission and dependency checks. `advanced-v2-tests.json` records all twelve grader suites. Tests include valid reference ZIP execution, independently recomputed metrics, data/graph tampering, masks, invalid operations, equivalent lookup/scaling, partial submissions, flexible interventions and changed-weight comparisons.

The original three-seed training evidence remains labeled with its original app commits. The architecture/training recipes were not changed or retrained for this editorial and assessment revision. The newly derived evaluation and probability copies preserve learned values and are checked independently. Local production-browser checks use clean isolated app `813e34ad8876ef2b2233fb8e9f2b7f954435e9d8`; the shared app repository was not modified. See `advanced-v2-browser.json` and `advanced-v2-browser-grading.json` for the actual controls exercised and saved-file verification. Browser checks start from instructor graphs, not a complete drag-and-drop reconstruction. The public deployed app and hosted Gradescope were not tested.

Reproduce packaging and the release suites using the commands in each instructor guide. `scripts/assignments/advanced/browser-v2.cjs` checks new browser workflows using PLAYWRIGHT_ROOT, CHROMIUM_PATH, APP_URL and NEURALCANVAS_ROOT. It writes evidence only inside Assignments 9–20 and docs.

## Subsequent backend guidance update

The 2026-10-19 backend review is recorded in [BACKEND-VALIDATION.md](BACKEND-VALIDATION.md). It separately authorizes and checks guidance changes in Assignments 1–8, as requested by the instructor. The v2 preservation audit above describes the earlier revision snapshot; the backend validation audit checks the later update. Existing datasets, rubrics and graders remain unchanged.
