# Reward learning: implementation and validation

Assignments 25 and 26 continue the same two-block, two-head, 5,689-parameter
football transformer after instruction tuning. Their output is the exact frozen
transformer used by the final SAE assignment; a separately supplied steering
checkpoint contains a matching final transformer and trained SAE.

The app supports exact expected reward over the nine observed answer tokens,
with reference KL 0.1. This is a finite one-step policy objective, not PPO or
sampled exploration. RLVR uses the canonical answer checker. Preference mode
learns a bounded action reward from pairwise comparisons using a fixed reference
representation, then optimizes the transformer while holding the reward fixed.

## Preference provenance

Students enter human comparisons in the app under a specified factual-correctness
criterion. The reference pilots use explicitly simulated verifier-derived
comparisons (52 comparisons across 26 contexts, covering all nine answers); no precollected human-rating dataset or human-subject experiment is
claimed. Those fixtures validate mechanics and attainable saved-artifact targets,
not the reliability of every possible human judgment. Graders cannot establish
who entered comparison records.

## Independent initialization results

The native PyTorch primitive-graph interpreter was run from three retained
instruction-tuning initializations with Adam 0.0003, batch 32, clipping 1, two reward
epochs per stage. All validation accuracies below use the full output vocabulary
on the same 240 held-out football questions; they are not candidate-renormalized
accuracy. Validation is used for model selection, not an untouched final test.

| Seed | Instruction-tuned | RLVR | Learned-preference reward |
| --- | ---: | ---: | ---: |
|137|73.33%|72.50%|68.75%|
|211|77.08%|75.00%|74.58%|
|307|78.33%|76.25%|76.67%|

The final policy criterion is 65% full-vocabulary validation accuracy plus a
positive independently recomputed training reward-objective gain. Higher reward
is not presented as a guarantee of higher answer accuracy. Seed137 is the
canonical final checkpoint for the supplied steering model.

## Actual app checks

All three native chains pass the actual app's project parser after computing its
reference signature. On the canonical seed 137 policies, actual Tensor CPU and
independent PyTorch policy gradients were compared for both RLVR and learned
reward objectives on eight canonical contexts. Maximum absolute gradient errors
were 1.64e-7 and 1.25e-7; objective errors were 5.96e-8 and 0. These checks validate
calculation agreement, not identical training trajectories between backends.
Native shuffling differs from the app's deterministic order.

The private repository retains scripts, trained references, raw reports and
extracted-package tests under scripts/assignments/rewards/ and pilots/rewards/.
Both canonical releases earned 100/100 in local extracted-package execution; invalid submissions and forged cached reports were tested. The course packages independently evaluate operations and parameters, preserve
partial credit and reject invalid data, references, comparison records and
malformed submissions. Cached reward scores are ignored. Local checks do not
certify hosted Gradescope or the deployed website.

## Complete app-engine run

A separate seed 137 run completed both two-epoch stages in the actual Tensor CPU
engine on all 3,840 training examples. In the verifiable stage, validation
expected correctness rose from 0.6493 to 0.7146. In the learned-reward stage,
validation expected learned reward rose from 0.6658 to 0.7147 while expected
correctness fell from 0.7146 to 0.6851. Final candidate accuracy was 68.75%.
These are conditional candidate metrics; the native table above reports
full-vocabulary accuracy independently. The two backends need not produce the
same parameter arrays.

The native seed 137 checkpoint remains the canonical SAE base. The separate app
run does not replace it. Its pilot began before a serialization clone fix was
loaded; deterministic reward metadata was reattached afterward without changing
transformer parameters, then validated with the current app parser. Current
serialization is additionally covered by the runtime round-trip tests.

Validation used app parent commit `16a9b4d` with the reward-learning implementation
present as working-tree changes. Exact runtime and script hashes are retained in
the private `pilots/rewards/validation-environment.json` evidence.
