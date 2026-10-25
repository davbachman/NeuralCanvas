# Public course materials

This directory is the canonical home for the 19 current assignments, Canvas
handouts, dataset fixtures, Gradescope ZIPs and grader sources, untrained
templates, instructor guides, packaging helpers, and curriculum/validation
documentation. Read README.md and the relevant INSTRUCTOR.md before changes.

The instructor explicitly permits public wiring, model templates, shape details,
and autograder internals. Keep trained solution files, fitted answer coefficients,
answer keys, scripts/pilots that produce complete trained solutions, and exams in
the private NeuralCanvas-Coursework repository. Ask the user when uncertain.
Do not import private Git history or copy entire private directories here.

All current assignments use the app's dataset browser. Preserve saved-model
submissions, dataset bytes/splits, published grading thresholds, and stable ZIP
filenames. Do not add dataset upload instructions, student starter models,
written responses, or manually graded questions. CSV bytes are protected by the
repository's .gitattributes. Keep one current version per assignment.

Canvas/description.html is the canonical handout. Run
`python course/scripts/sync-handouts.py` from the app root after editing it;
`public/course/` contains the generated student website pages. Only student
handouts are copied into the deployed app; instructor materials remain in Git.

Run course commands from this directory. Use Python 3.12 and requirements.txt.
Run `python scripts/validate-course.py` after changes; it checks package/source
parity, manifests, dataset copies, public artifact boundaries, and all 19
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
