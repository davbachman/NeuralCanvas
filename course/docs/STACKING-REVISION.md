# Stacking revision: Days 10–20

Numbering note: this dated report retains its original assignment numbers. See
[SAMPLING-SEQUENCE.md](SAMPLING-SEQUENCE.md) for the current sequence after moving sampling to Assignment 12.

For the current synthetic-narrative Assignments 11–20, see [NARRATIVE-REVISION.md](NARRATIVE-REVISION.md). The measurements and earlier sequence below are historical evidence; they do not validate the replacement dataset.

Revised 2026-10-19. Assignments 1–9 and all issued student dataset bytes are unchanged. The current sequence has one active assignment per day and no standalone intervention day.

## Sequence and assessed work

| Day | Current assignment | Change |
|---|---|---|
| 10 | Question-directed attention | Train attention (80 points), then three short saved input interventions (15) and an unchanged-parameter check (5). |
| 11 | Next-character prediction | Former Day 12, same model and numerical thresholds. |
| 12 | Longer context MLP | Former Day 13, same model and thresholds. |
| 13 | Causal self-attention | Former Day 14, same model and thresholds. |
| 14 | Residual/feed-forward updates | Former Day 15, same model and thresholds. |
| 15 | Layer normalization | Former Day 16, same model and thresholds. |
| 16 | All-position supervision | Former Day 17, retaining one block and the fixed-weight final-position evaluation copy. |
| 17 | Stacking | New: submit the trained one-block all-position model (30 points), then independently build/train two blocks (70). |
| 18 | Multiple heads | Two width-8 heads in each of the two successive blocks. |
| 19 | Sampling | Reuse the two-block/two-head model; temperature branches preserve its learned parameters. |
| 20 | Synthesis | Reuse bigram, context, stacked and multihead models; submit a final-position multihead evaluation copy. |

The central new idea is that later attention can base its queries, keys and values on representations already enriched by earlier attention. This differs from parallel heads within a block. The depth lesson is reused in every subsequent model. Students can use Edit → Duplicate for the first complete block, then independently reconnect it as the next stage; they need not redraw every internal wire.

## Pilot decision

All current CPU pilots used clean app **2b6b96d8d07ee52e0beca3c3432f9f4e5819af79**, width 16, feed-forward width 32, context 8, the unchanged Alice split/vocabulary, Adam at 0.003, batch 32, reshuffling, clip norm 1, no early stopping or penalty, and five epochs in one run with reports every epoch. Initializers match the handout: Uniform embeddings, Xavier matrices, zero biases/offsets and unit normalization scales. Paired one-/two-block runs start with identical values for their shared first-block, embedding and output parameters; the second block receives separate seeded initialization. Students use the same initializer methods, not prescribed numeric seeds.

The initial experiment added depth to the final-position-only objective. Results were mixed:

| Seed | One-block final-position loss | Two-block final-position loss |
|---|---|---|
| 137 | 2.136672 | 2.156323 |
| 211 | 2.158861 | 2.151837 |
| 307 | 2.138003 | 2.113245 |

That did not justify promising a reliable improvement before teaching all-position supervision. The final sequence therefore teaches all-position training first, then depth. On that shared task:

| Seed | One block, one head | Two blocks, one head each | Two blocks, two heads each |
|---|---|---|---|
| 137 | 2.120314 | 2.087203 | 2.061065 |
| 211 | 2.128943 | 2.107851 | 2.046964 |
| 307 | 2.109196 | 2.089110 | 2.047451 |

All three depth comparisons improve, but the gains are modest. One block has **3,805 parameters**; two blocks have **5,965**. Splitting each block into two heads retains **5,965**. Equal epoch/update budgets do not imply equal work per update or equal runtime. Elapsed pilot times include concurrent cloud workloads and should not be advertised as laptop benchmarks. At this scale the context MLP remains a strong baseline; this is not a claim that each transformer component beats all earlier models.

Assignment 17 uses an absolute **2.3 training/validation cross-entropy target** for the stacked model, with the established **2.5** limits for its one-block comparison. It does not require a particular difference between student submissions or penalize an unusually good baseline. All other retained numerical thresholds are unchanged. The all-position average must not be compared directly with final-position loss; Assignment 16 supplies a graded evaluation copy, and the synthesis grader reports common final-position metrics.

## Grader and release changes

Assignments 7–20 are unpublished and each has one current handout and upload ZIP, with stable filenames. Earlier drafts, including the former standalone intervention assignment, remain in Git history. Private pilot evidence retains its original numbering and provenance. The unchanged Q @ K.T requirement in Assignment 10 is preserved; the equivalent K @ Q.T/reshape construction was not added to the grader.

The bounded independent evaluator now caches static arithmetic signatures and visits shared signature subgraphs once during target-leakage checks. It still recomputes every example numerically and checks the assigned architecture; the change avoids redundant work on the larger graph. Depth regression tests reject bypassing the second block, using original embeddings as its input, sharing first-block Params, or removing a second-block causal mask. Parameter counters also enforce separate normalization and head Params. Resource limits remain in place.

Every required independent task produces a saved model artifact. No written response, quiz or required unsubmitted investigation was added. New architecture comparisons start fresh; interventions, temperature branches and evaluation copies retain fixed learned parameters.

## Verification and limits

- CPU: 15 complete training runs (six exploratory final-position runs; nine all-position/depth/head runs), reimport checks, and independent NumPy comparisons for every final architecture and seed.
- Packaged graders: full-credit reference submissions, invalid structures, tampered data, target leakage, malformed/partial files and changed comparison weights. Results: `stacking-tests.json`, `stacking-seed-grading.json`, and per-assignment test logs/results.
- Production browser: actual applied/saved interventions; two-block import, held-out inference, File → Save, parameter initialization and one training epoch through the UI; Edit → Duplicate preserves internal connections and independent Params. The duplication selection is supplied in the fixture; checks do not claim to draw every wire. Current handouts fit 390- and 960-pixel widths.
- Auto/WebGL: fresh five-epoch runs of the two-block single-head and two-head architectures, plus independent grading of saved results. Unchanged one-block recipes retain their earlier full-budget WebGL evidence. Chromium used **SwiftShader software rendering**, which verifies compatibility and is not a physical GPU speed benchmark. Tensor CPU remains the optional speed/fallback choice.
- Browser and grader checks are local. They do not establish deployment to the student website or hosted Gradescope. A GitHub push does not publish course sites; upload the new handout and matching grader together.

## Reproduction

Run from the coursework root with `NEURALCANVAS_ROOT` pointing to an isolated checkout of the app commit above. Do not modify or switch the shared app checkout.

```sh
for pilot_seed in 137 211 307; do
  PILOT_SEEDS="$pilot_seed" node scripts/assignments/advanced/pilot.mjs assignments/11-next-character/Canvas/alice.json "/tmp/depth-final-$pilot_seed" transformer stacked
done
for pilot_kind in all-position stacked-all stacked-two-heads; do
  PILOT_SEEDS=137,211,307 node scripts/assignments/advanced/pilot.mjs assignments/11-next-character/Canvas/alice.json "/tmp/depth-$pilot_kind" "$pilot_kind"
done
.venv/bin/python scripts/assignments/advanced/install-stacking-pilots.py /tmp/depth-final-137 /tmp/depth-final-211 /tmp/depth-final-307 /tmp/depth-all-position /tmp/depth-stacked-all /tmp/depth-stacked-two-heads
.venv/bin/python scripts/assignments/advanced/grade-stacking-seeds.py
.venv/bin/python scripts/assignments/advanced/audit-stacking.py
```

The pilot kinds use ordinary editable app operations. `install-stacking-pilots.py` documents the installation arguments for separate seed/model output folders; `prepare-models.mjs` derives fixed-weight copies, and `configure-releases.py` rebuilds canonical configurations and handout rubrics. These commands overwrite reference artifacts and should be used intentionally. They do not regenerate issued raw datasets. `RELEASE_ASSIGNMENTS` can restrict configuration/packaging to specified comma-separated folder names. Current UI reproduction scripts are `browser-interventions.cjs`, `browser-stacking.cjs`, and `browser-duplicate-block.cjs`. Historical reproduction scripts are available in Git history.
