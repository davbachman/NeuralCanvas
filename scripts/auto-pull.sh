#!/bin/bash
# Install a per-user macOS job, or run one guarded update of this checkout.
set -euo pipefail

nc_label="org.neuralcanvas.auto-pull"
nc_support="$HOME/Library/Application Support/NeuralCanvas"
nc_runner="$nc_support/auto-pull.sh"
nc_plist="$HOME/Library/LaunchAgents/$nc_label.plist"
nc_log="$HOME/Library/Logs/NeuralCanvas-auto-pull.log"

message() { printf '[%s] %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*"; }
fail() { message "$*" >&2; exit 1; }
git_repo() { "$nc_git" -C "$nc_repo" "$@"; }
require_mac() { [ "$(uname -s)" = Darwin ] || fail 'Installation requires your Mac; run this in its Terminal.'; }

validate_repo() {
  [ -d "$nc_repo" ] || fail "Checkout unavailable: $nc_repo"
  [ -x "$nc_git" ] || fail "Git unavailable: $nc_git"
  [ "$(git_repo rev-parse --is-inside-work-tree)" = true ] || fail 'Choose a Git checkout.'
  git_repo remote get-url origin >/dev/null || fail 'The checkout needs an origin remote.'
}

check_worktree() {
  if [ "$(git_repo symbolic-ref --quiet --short HEAD || true)" != main ]; then
    message 'Skipped: the checkout is not on main.'
    return 1
  fi
  if [ -n "$(git_repo status --porcelain --untracked-files=normal)" ]; then
    message 'Skipped: local edits or untracked files need attention.'
    return 1
  fi
  nc_git_dir="$(git_repo rev-parse --absolute-git-dir)"
  for nc_operation in MERGE_HEAD CHERRY_PICK_HEAD REVERT_HEAD rebase-merge rebase-apply sequencer; do
    if [ -e "$nc_git_dir/$nc_operation" ]; then
      message 'Skipped: another Git operation is in progress.'
      return 1
    fi
  done
}

run_update() {
  validate_repo
  check_worktree || return 0
  export GIT_TERMINAL_PROMPT=0
  git_repo -c http.lowSpeedLimit=1 -c http.lowSpeedTime=30 fetch --quiet --no-tags origin \
    refs/heads/main:refs/remotes/origin/main
  # Check again after the network operation, before changing any working files.
  check_worktree || return 0
  if [ "$(git_repo rev-list --count refs/remotes/origin/main..HEAD)" != 0 ]; then
    message 'Skipped: local commits need attention.'
    return 0
  fi
  nc_before="$(git_repo rev-parse HEAD)"
  git_repo merge --ff-only --no-edit refs/remotes/origin/main
  nc_after="$(git_repo rev-parse HEAD)"
  if [ "$nc_before" = "$nc_after" ]; then
    message "Already current: $nc_after"
  else
    message "Updated main: $nc_before -> $nc_after"
  fi
}

write_plist() {
  # plutil escapes spaces, ampersands, and other special path characters.
  /usr/bin/plutil -create xml1 "$nc_temp_plist"
  /usr/bin/plutil -insert Label -string "$nc_label" "$nc_temp_plist"
  /usr/bin/plutil -insert ProgramArguments -json '[]' "$nc_temp_plist"
  /usr/bin/plutil -insert ProgramArguments.0 -string /bin/bash "$nc_temp_plist"
  /usr/bin/plutil -insert ProgramArguments.1 -string "$nc_runner" "$nc_temp_plist"
  /usr/bin/plutil -insert ProgramArguments.2 -string run "$nc_temp_plist"
  /usr/bin/plutil -insert ProgramArguments.3 -string "$nc_repo" "$nc_temp_plist"
  /usr/bin/plutil -insert ProgramArguments.4 -string "$nc_git" "$nc_temp_plist"
  /usr/bin/plutil -insert StartInterval -integer 300 "$nc_temp_plist"
  /usr/bin/plutil -insert RunAtLoad -bool YES "$nc_temp_plist"
  /usr/bin/plutil -insert ProcessType -string Background "$nc_temp_plist"
  /usr/bin/plutil -insert StandardOutPath -string "$nc_log" "$nc_temp_plist"
  /usr/bin/plutil -insert StandardErrorPath -string "$nc_log" "$nc_temp_plist"
  /usr/bin/plutil -lint "$nc_temp_plist" >/dev/null
}

main() {
case "${1:-help}" in
  run)
    [ "$#" -ge 2 ] || fail 'Usage: auto-pull.sh run /absolute/path/to/checkout [git-binary]'
    nc_repo="$2"
    nc_git="${3:-$(command -v git)}"
    run_update
    ;;
  install)
    require_mac
    nc_source_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
    nc_source="$nc_source_dir/$(basename "${BASH_SOURCE[0]}")"
    nc_git="$(command -v git)"
    nc_repo="$("$nc_git" -C "$nc_source_dir/.." rev-parse --show-toplevel)"
    validate_repo
    mkdir -p "$nc_support" "$(dirname "$nc_plist")" "$(dirname "$nc_log")"
    if [ "$nc_source" != "$nc_runner" ]; then cp "$nc_source" "$nc_runner"; fi
    chmod 700 "$nc_runner"
    nc_temp_dir="$(mktemp -d "$nc_support/agent.XXXXXX")"
    trap 'rm -f "$nc_temp_dir/agent.plist"; rmdir "$nc_temp_dir"' EXIT
    nc_temp_plist="$nc_temp_dir/agent.plist"
    write_plist
    chmod 600 "$nc_temp_plist"
    mv "$nc_temp_plist" "$nc_plist"
    nc_service="gui/$(id -u)/$nc_label"
    /bin/launchctl bootout "$nc_service" >/dev/null 2>&1 || true
    /bin/launchctl enable "$nc_service"
    /bin/launchctl bootstrap "gui/$(id -u)" "$nc_plist"
    message "Enabled automatic updates of $nc_repo every five minutes while logged in."
    printf 'Check status: bash scripts/auto-pull.sh status\nLog: %s\n' "$nc_log"
    ;;
  status)
    require_mac
    if nc_state="$(/bin/launchctl print "gui/$(id -u)/$nc_label" 2>/dev/null)"; then
      printf 'NeuralCanvas automatic updater is loaded.\n'
      printf '%s\n' "$nc_state" | sed -nE '/^[[:space:]]*(state|runs|last exit code) =/p'
      if [ -f "$nc_log" ]; then tail -n 8 "$nc_log"; fi
    else
      fail 'Automatic updater is not loaded. Run: bash scripts/auto-pull.sh install'
    fi
    ;;
  uninstall)
    require_mac
    /bin/launchctl bootout "gui/$(id -u)/$nc_label" >/dev/null 2>&1 || true
    rm -f "$nc_plist" "$nc_runner"
    message 'Automatic updater removed. Checkout and log retained.'
    ;;
  *)
    printf 'Usage: bash scripts/auto-pull.sh {install|status|uninstall}\n'
    ;;
esac
}

if [ "${BASH_SOURCE[0]}" = "$0" ]; then main "$@"; fi
