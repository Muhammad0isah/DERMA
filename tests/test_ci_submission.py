"""Mocked GitHub API tests for the trusted submission boundary."""
from __future__ import annotations
import copy
import hashlib
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from scripts.ci_submission import (CIError, MAX_ENVELOPE_BYTES, fetch_files, parse_event,
                                   publish_result, score_submission, select_files, validate_bundle)

HEAD = "a" * 40
REPOSITORY = "organizer/competition"
HEAD_REPOSITORY = "participant/competition"
TEAM = "team-one"


def blob_sha(contents: bytes) -> str:
    return hashlib.sha1(b"blob " + str(len(contents)).encode() + b"\0" + contents).hexdigest()


class FakeAPI:
    def __init__(self):
        self.calls = []
        self.identity = {"repository": REPOSITORY, "head_repository": HEAD_REPOSITORY,
                         "head_sha": HEAD, "actor": "alice", "pr_number": 17}
        self.pull = {"state": "open", "changed_files": 2, "head": {"sha": HEAD, "repo": {"full_name": HEAD_REPOSITORY}},
                     "base": {"repo": {"full_name": REPOSITORY}}, "user": {"login": "alice"}}
        self.contents = {"submission.enc": b"encrypted predictions", "meta.json": json.dumps({"team": TEAM, "type": "human", "model": "none"}).encode()}
        self.files = [{"filename": f"submissions/{TEAM}/{name}", "sha": blob_sha(value), "status": "added",
                       "raw_url": "https://attacker.invalid/never-follow"} for name, value in self.contents.items()]
        self.final_tree = {"truncated": False, "tree": [{"path": name, "mode": "100644", "type": "blob",
                                                        "sha": blob_sha(value), "size": len(value)} for name, value in self.contents.items()]}
        self.override_metadata = None
        self.override_raw = None

    def request(self, route, *, raw=False, limit=None, method="GET", payload=None):
        self.calls.append((route, raw, limit, method))
        if route == f"/repos/{REPOSITORY}/pulls/17":
            return copy.deepcopy(self.pull)
        if route == f"/repos/{REPOSITORY}/pulls/17/files?per_page=100&page=1":
            return copy.deepcopy(self.files)
        if route == f"/repos/{HEAD_REPOSITORY}/git/commits/{HEAD}":
            return {"tree": {"sha": "1" * 40}}
        for index, component in enumerate(("submissions", TEAM), 1):
            if route == f"/repos/{HEAD_REPOSITORY}/git/trees/{str(index) * 40}":
                return {"truncated": False, "tree": [{"path": component, "mode": "040000", "type": "tree", "sha": str(index + 1) * 40}]}
        if route == f"/repos/{HEAD_REPOSITORY}/git/trees/{'3' * 40}":
            return copy.deepcopy(self.final_tree)
        for name, value in self.contents.items():
            if route == f"/repos/{HEAD_REPOSITORY}/contents/submissions/{TEAM}/{name}?ref={HEAD}":
                if raw:
                    return self.override_raw if self.override_raw is not None else value
                return self.override_metadata or {"type": "file", "sha": blob_sha(value), "size": len(value)}
        if method == "POST" and route == f"/repos/{REPOSITORY}/issues/17/comments":
            return {"id": 1}
        raise AssertionError(f"Unexpected API route: {route}")


class SubmissionBoundaryTests(unittest.TestCase):
    def test_event_uses_author_and_checks_base_repository(self):
        api = FakeAPI()
        event = {"number": 17, "repository": {"full_name": REPOSITORY}, "pull_request": api.pull,
                 "sender": {"login": "not-the-author"}}
        identity = parse_event(event, REPOSITORY)
        self.assertEqual(identity["actor"], "alice")
        event["pull_request"]["head"]["sha"] = "$(arbitrary shell)"
        with self.assertRaises(CIError):
            parse_event(event, REPOSITORY)

    def test_fetch_only_two_changed_regular_files_at_immutable_sha(self):
        api = FakeAPI()
        team, envelope, metadata = fetch_files(api, api.identity)
        self.assertEqual(team, TEAM)
        self.assertEqual(envelope, api.contents["submission.enc"])
        self.assertEqual(metadata["type"], "human")
        for route, raw, limit, method in api.calls:
            self.assertTrue(route.startswith("/repos/"))
            self.assertNotIn("attacker.invalid", route)
            if "/contents/" in route:
                self.assertTrue(route.endswith("?ref=" + HEAD))
                if raw and "submission.enc" in route:
                    self.assertEqual(limit, MAX_ENVELOPE_BYTES)

    def test_changed_file_attack_paths_and_statuses_are_rejected(self):
        api = FakeAPI()
        attacks = ["../../scripts/ci_submission.py", "submissions/../submission.enc",
                   "submissions/other/submission.enc", "submissions/team-one/extra.json",
                   "submissions/team-one/submission.enc\n"]
        for path in attacks:
            files = copy.deepcopy(api.files)
            files[0]["filename"] = path
            with self.subTest(path=path), self.assertRaises(CIError):
                select_files(files)
        for status in ("removed", "renamed", "copied"):
            files = copy.deepcopy(api.files)
            files[0]["status"] = status
            with self.assertRaises(CIError):
                select_files(files)
        with self.assertRaises(CIError):
            select_files(api.files + [api.files[0]])

    def test_symlink_oversize_and_blob_mismatch_are_rejected(self):
        for modification in ("symlink", "oversize", "sha", "truncated", "raw"):
            api = FakeAPI()
            if modification == "symlink":
                api.final_tree["tree"][0]["mode"] = "120000"
            elif modification == "oversize":
                api.final_tree["tree"][0]["size"] = MAX_ENVELOPE_BYTES + 1
            elif modification == "sha":
                api.override_metadata = {"type": "file", "sha": "b" * 40, "size": len(api.contents["submission.enc"])}
            elif modification == "truncated":
                api.final_tree["truncated"] = True
            else:
                api.override_raw = b"forged content"
            with self.subTest(modification=modification), self.assertRaises(CIError):
                fetch_files(api, api.identity)

    def test_stale_head_and_extra_changed_files_are_rejected(self):
        api = FakeAPI()
        api.pull["head"]["sha"] = "b" * 40
        with self.assertRaises(CIError):
            fetch_files(api, api.identity)
        api = FakeAPI()
        api.pull["changed_files"] = 3
        with self.assertRaises(CIError):
            fetch_files(api, api.identity)


class ScoreArtifactTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        (self.root / "competition").mkdir()
        (self.root / "evaluation").mkdir()
        (self.root / "competition/config.json").write_text(json.dumps({"benchmark_id": "test-v1"}))
        (self.root / "evaluation/test_manifest.json").write_text(json.dumps({"benchmark_id": "test-v1", "images": []}))
        (self.root / "evaluation/ground_truth.enc").write_bytes(b"encrypted ground truth")
        self.api = FakeAPI()
        self.event = self.root / "event.json"
        self.event.write_text(json.dumps({"number": 17, "repository": {"full_name": REPOSITORY}, "pull_request": self.api.pull}))

    def score(self):
        output = self.root / "result.json"
        with mock.patch("scripts.ci_submission.decrypt_bytes", side_effect=[b"plaintext predictions", b"secret labels"]) as decrypt:
            with mock.patch("competition.task.validate_submission") as validate:
                with mock.patch("competition.task.evaluate", return_value={"score": 0.8, "metrics": {"iou": 0.8}, "primary_metric": "iou"}) as evaluate:
                    bundle = score_submission(self.root, self.event, output, self.api, REPOSITORY, b"private key")
        self.assertEqual(decrypt.call_args_list[0].kwargs["purpose"], "submission")
        self.assertEqual(decrypt.call_args_list[1].kwargs["purpose"], "ground_truth")
        self.assertEqual(validate.call_args.args[1], self.root / "evaluation/test_manifest.json")
        self.assertEqual(evaluate.call_args.args[2], self.root / "evaluation/test_manifest.json")
        return bundle, output

    def test_artifact_contains_only_public_results_and_release_binding(self):
        bundle, output = self.score()
        text = output.read_text()
        for secret in ("plaintext predictions", "secret labels", "private key"):
            self.assertNotIn(secret, text)
        self.assertEqual(bundle["actor"], "alice")
        self.assertEqual(bundle["result"]["score"], 0.8)
        self.assertEqual(validate_bundle(bundle, self.root, REPOSITORY), bundle)
        (self.root / "evaluation/test_manifest.json").write_text('{"benchmark_id":"changed"}')
        with self.assertRaises(CIError):
            validate_bundle(bundle, self.root, REPOSITORY)

    def test_missing_release_or_secret_has_clear_setup_error(self):
        with self.assertRaisesRegex(CIError, "SUBMISSION_PRIVATE_KEY"):
            score_submission(self.root, self.event, self.root / "result.json", self.api, REPOSITORY, b"")
        (self.root / "evaluation/ground_truth.enc").unlink()
        with self.assertRaisesRegex(CIError, "setup is incomplete"):
            score_submission(self.root, self.event, self.root / "result.json", self.api, REPOSITORY, b"key")

    def test_publishing_refuses_to_reset_local_worktrees(self):
        with mock.patch.dict(os.environ, {"GITHUB_ACTIONS": "false"}):
            with self.assertRaisesRegex(CIError, "isolated GitHub Actions"):
                publish_result(self.root, self.root / "result.json", self.api, REPOSITORY, "token", "main")

    def test_publication_retries_after_non_fast_forward(self):
        bundle, output = self.score()
        calls, pushes = [], []
        def fake_git(root, arguments, token, check=True):
            calls.append(arguments)
            status = 0
            if arguments[0] == "diff":
                status = 1
            if arguments[0] == "push":
                pushes.append(arguments)
                status = 1 if len(pushes) == 1 else 0
            return mock.Mock(returncode=status)
        with mock.patch.dict(os.environ, {"GITHUB_ACTIONS": "true", "GITHUB_WORKSPACE": str(self.root)}):
            with mock.patch("scripts.ci_submission.git", side_effect=fake_git):
                entry = publish_result(self.root, output, self.api, REPOSITORY, "token", "main")
        self.assertEqual(entry["score"], 0.8)
        self.assertEqual(len(pushes), 2)
        self.assertEqual(sum(arguments[0] == "fetch" for arguments in calls), 2)
        self.assertEqual(len(json.loads((self.root / "leaderboard/records.json").read_text())), 1)


if __name__ == "__main__":
    unittest.main()
