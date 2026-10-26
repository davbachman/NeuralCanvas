# Automatic updates of a Mac checkout

Run these commands in Terminal inside your local NeuralCanvas checkout:

```sh
git pull --ff-only
bash scripts/auto-pull.sh install
bash scripts/auto-pull.sh status
```

The installer creates a macOS LaunchAgent for your user. It runs at login and
every five minutes while you are logged in and the Mac is awake, including when
Terminal and Codex are closed. It fetches `origin/main` and fast-forwards the
local `main` branch. It skips updates when you have local edits, untracked files,
local commits, another Git operation in progress, or a different branch checked
out. It does not push, discard work, reset branches, or create merge commits.
Network failures are logged; the next scheduled run tries again.

This updates source files and course materials. It does not install new Node
dependencies or rebuild `dist/`. Use `npm ci` and `npm run build` when running or
building the updated app locally. Normal use of the hosted app requires neither.

For a checkout in iCloud Drive, select **Keep Downloaded** in Finder so its
working files and `.git` metadata remain available. If you move the checkout,
run the installer again from its new location. The copied runner lives outside
iCloud in `~/Library/Application Support/NeuralCanvas/`. Rerun the installer to
refresh that copy after changes to the updater script.

Inspect the updater and its latest results with:

```sh
bash scripts/auto-pull.sh status
tail -n 20 "$HOME/Library/Logs/NeuralCanvas-auto-pull.log"
```

The status should say the updater is loaded. A successful run logs either
`Already current` or `Updated main`. A loaded agent with a nonzero last exit code
or an error in the log needs attention; consult the actual error before changing
authentication or permissions. Installation must run on your Mac; a cloud Codex
workspace cannot enable your Mac's LaunchAgent.

To stop automatic updates:

```sh
bash scripts/auto-pull.sh uninstall
```

Uninstalling removes this updater's LaunchAgent and copied runner, retaining
your repository and log. The job label is `org.neuralcanvas.auto-pull`; its
configuration is `~/Library/LaunchAgents/org.neuralcanvas.auto-pull.plist`.
