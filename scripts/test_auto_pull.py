"""Exercise the updater against real local Git repositories without launchd."""
from pathlib import Path
import plistlib
import shutil
import subprocess
import sys
import tempfile
import unittest

SCRIPT = Path(__file__).resolve().with_name('auto-pull.sh')


class AutoPullTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.remote = self.root / 'remote.git'
        self.writer = self.root / 'writer'
        self.reader = self.root / 'local checkout & spaces'
        self.git('init', '--bare', '--initial-branch=main', str(self.remote))
        self.git('clone', str(self.remote), str(self.writer))
        self.configure(self.writer)
        self.write_commit(self.writer, 'lesson.txt', 'original\n')
        self.git('-C', str(self.writer), 'push', 'origin', 'main')
        self.git('clone', str(self.remote), str(self.reader))
        self.configure(self.reader)

    def git(self, *args):
        return subprocess.check_output(['git', *args], text=True, stderr=subprocess.STDOUT).strip()

    def configure(self, repo):
        self.git('-C', str(repo), 'config', 'user.name', 'Updater Test')
        self.git('-C', str(repo), 'config', 'user.email', 'updater@example.invalid')

    def write_commit(self, repo, name, contents):
        (repo / name).write_text(contents)
        self.git('-C', str(repo), 'add', name)
        self.git('-C', str(repo), 'commit', '-m', name)

    def advance_remote(self):
        self.write_commit(self.writer, 'lesson.txt', 'updated\n')
        self.git('-C', str(self.writer), 'push', 'origin', 'main')

    def update(self, expected=0):
        result = subprocess.run(['bash', str(SCRIPT), 'run', str(self.reader), shutil.which('git')],
                                text=True, capture_output=True, timeout=30)
        self.assertEqual(result.returncode, expected, result.stdout + result.stderr)
        return result.stdout + result.stderr

    def head(self):
        return self.git('-C', str(self.reader), 'rev-parse', 'HEAD')

    def test_clean_main_updates_and_repeat_is_current(self):
        self.advance_remote()
        self.assertIn('Updated main:', self.update())
        self.assertEqual((self.reader / 'lesson.txt').read_text(), 'updated\n')
        self.assertEqual(self.head(), self.git('-C', str(self.writer), 'rev-parse', 'HEAD'))
        self.assertIn('Already current:', self.update())

    def test_local_edits_and_untracked_files_are_preserved(self):
        self.advance_remote()
        before = self.head()
        (self.reader / 'lesson.txt').write_text('local edits\n')
        self.assertIn('local edits', self.update())
        self.assertEqual(self.head(), before)
        self.assertEqual((self.reader / 'lesson.txt').read_text(), 'local edits\n')
        self.git('-C', str(self.reader), 'restore', 'lesson.txt')
        (self.reader / 'notes.txt').write_text('private notes\n')
        self.assertIn('untracked files', self.update())
        self.assertEqual(self.head(), before)
        self.assertEqual((self.reader / 'notes.txt').read_text(), 'private notes\n')

    def test_other_branch_and_detached_head_are_unchanged(self):
        self.advance_remote()
        before = self.head()
        self.git('-C', str(self.reader), 'checkout', '-b', 'experiment')
        self.assertIn('not on main', self.update())
        self.assertEqual(self.head(), before)
        self.git('-C', str(self.reader), 'checkout', '--detach')
        self.assertIn('not on main', self.update())
        self.assertEqual(self.head(), before)

    def test_local_commits_are_preserved_without_merging(self):
        self.write_commit(self.reader, 'local.txt', 'local work\n')
        before = self.head()
        self.assertIn('local commits', self.update())
        self.advance_remote()
        self.assertIn('local commits', self.update())
        self.assertEqual(self.head(), before)
        self.assertEqual((self.reader / 'lesson.txt').read_text(), 'original\n')

    def test_in_progress_operation_is_unchanged(self):
        self.advance_remote()
        before = self.head()
        (self.reader / '.git/rebase-merge').mkdir()
        self.assertIn('another Git operation', self.update())
        self.assertEqual(self.head(), before)

    def test_unreachable_remote_does_not_change_checkout(self):
        before = self.head()
        self.git('-C', str(self.reader), 'remote', 'set-url', 'origin', str(self.root / 'missing.git'))
        self.update(expected=128)
        self.assertEqual(self.head(), before)

    @unittest.skipUnless(sys.platform == 'darwin', 'Native macOS plutil check')
    def test_launch_agent_plist_preserves_paths_and_schedule(self):
        target = self.root / 'agent.plist'
        runner = self.root / 'Application Support & tools/auto-pull.sh'
        log = self.root / 'logs & updates.log'
        git = shutil.which('git')
        subprocess.run(['bash', '-c', 'source "$1"; nc_temp_plist="$2"; nc_runner="$3"; '
                        'nc_repo="$4"; nc_git="$5"; nc_log="$6"; write_plist',
                        'plist-test', str(SCRIPT), str(target), str(runner), str(self.reader), git, str(log)],
                       check=True, capture_output=True, text=True)
        with target.open('rb') as stream:
            agent = plistlib.load(stream)
        self.assertEqual(agent['ProgramArguments'], ['/bin/bash', str(runner), 'run', str(self.reader), git])
        self.assertEqual(agent['StartInterval'], 300)
        self.assertTrue(agent['RunAtLoad'])
        self.assertEqual(agent['Label'], 'org.neuralcanvas.auto-pull')
        self.assertEqual(agent['StandardOutPath'], str(log))
        self.assertEqual(agent['StandardErrorPath'], str(log))


if __name__ == '__main__':
    unittest.main()
