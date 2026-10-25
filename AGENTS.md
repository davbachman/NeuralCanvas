# Project workflow

Work directly on `main` and stay on it. Do not create a separate branch unless
the user requests one.

After completing changes, run the relevant checks, then automatically commit
and push all project changes to the current branch on `origin`. Do not wait for
a separate request or confirmation to commit or push.

Use a descriptive commit message. If checks, committing, or pushing fail,
report the failure and preserve the work. Never force-push to resolve a failure.

# Course materials

Public teaching materials live in `course/`; read `course/AGENTS.md` before
changing them. Assignments, autograders, wiring/templates, instructor guides,
rubrics, and validation summaries belong here. Trained solutions, answer keys,
solution-producing pilots, and exams belong in the private sibling
`NeuralCanvas-Coursework` repository. Ask the user when the boundary is unclear.
