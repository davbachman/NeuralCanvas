# Public course materials

This directory is the canonical home for the 28 current assignments, Canvas
handouts, Gradescope ZIPs and grader sources, untrained
templates, instructor guides, packaging helpers, and curriculum/validation
documentation. Read README.md and the relevant Instructor/INSTRUCTOR.md before changes.

The instructor explicitly permits public wiring, model templates, shape details,
and autograder internals. Keep trained solution files, fitted answer coefficients,
answer keys, scripts/pilots that produce complete trained solutions, and exams in
the private NeuralCanvas-Coursework repository. Ask the user when uncertain.
Do not import private Git history or copy entire private directories here.

Instructor-authorized exception (2026-11-11): Assignment 28 supplies the frozen
trained checkpoint `student/feature-steering.json`, also deployed at
`public/models/feature-steering.json`. Keep its hash in the release manifest.
Completed discovery and intervention constructions remain private. Assignment
27 continues to use student-built models with no supplied trained model.

All current assignments use the app's dataset browser. Preserve saved-model
submissions, dataset bytes/splits, published grading thresholds, and stable current ZIP
filenames. Do not add dataset upload instructions, student starter models,
written responses, or manually graded questions. CSV bytes are protected by the
repository's .gitattributes. Keep one current version per assignment.

Each assignment's `description.html` is the canonical handout. Run
`python course/scripts/sync-handouts.py` from the app root after editing it;
this generates the adjacent `description.md` for GitHub, refreshes handout hashes,
and updates the student website pages in `public/course/`. Do not edit the
generated Markdown independently or recreate `Canvas/` folders. Datasets live
in the app's `public/datasets/`, with hashes in `src/learning/dataset-library.json`.
Only student handouts are copied into the deployed app; instructor materials
remain in Git.

Run course commands from this directory. Use Python 3.12 and requirements.txt.
Run `python scripts/validate-course.py` after changes; it checks package/source
parity, manifests, app dataset hashes, public artifact boundaries, and all 28
packaged runners with invalid submissions. Rebuild changed autograders with the
appropriate `scripts/assignments/package-*.py` command. Full-credit and semantic
regression tests use private fixtures and run from NeuralCanvas-Coursework.

For design changes, follow docs/ASSIGNMENT-DESIGN.md. Validate the exact student
recipe before setting numerical thresholds. Keep independent model-building
work and automatically graded artifacts. Do not regenerate issued source data
or claim a runtime/accuracy improvement without evidence. Public summaries
retain historical app revisions; private pilots retain trained artifacts.

A Git push does not upload materials to Canvas or Gradescope. Follow the root
AGENTS.md commit/push workflow for public changes. Private compatibility links
require both repositories as siblings; the public course must work on its own.

## Approved full-course revision (2026-11-11)

The instructor authorized the 28-assignment build-out and renumbering. Follow
NeuralCanvas/course/sequence.json for current folders and order. Current ZIP names
use gradescope-<assignment-folder>.zip. Preserve historical issued archives and
recipe evidence. The final sequence is instruction tuning, RLVR, RLHF, SAE feature
discovery, and steering. Interpretability uses the final reward-trained model;
checkpoint compatibility and feature-specific targets require fresh validation.
