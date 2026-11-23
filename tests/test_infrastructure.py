"""Security and first-valid-submission checks using temporary organizer state."""
from __future__ import annotations
import csv
import io
import json
from pathlib import Path
import tempfile
import unittest

from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import rsa

from competition.render_leaderboard import csv_safe, ranked_records
from competition.submission import SubmissionError, parse_metadata, record_result
from encryption.crypto import EnvelopeError, atomic_write, decrypt_bytes, encrypt_bytes


def key_pair(bits: int = 2048) -> tuple[bytes, bytes]:
    private = rsa.generate_private_key(public_exponent=65537, key_size=bits)
    return (private.private_bytes(serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8,
                                  serialization.NoEncryption()),
            private.public_key().public_bytes(serialization.Encoding.PEM, serialization.PublicFormat.SubjectPublicKeyInfo))


class EncryptionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.private, cls.public = key_pair()

    def test_round_trip_and_randomized_encryption(self) -> None:
        payload = bytes(range(256)) * 100
        first = encrypt_bytes(payload, self.public, "test-v1")
        second = encrypt_bytes(payload, self.public, "test-v1")
        self.assertNotEqual(first, second)
        self.assertEqual(decrypt_bytes(first, self.private, "test-v1"), payload)

    def test_dynamic_rsa_key_length(self) -> None:
        private, public = key_pair(3072)
        encrypted = encrypt_bytes(b"predictions", public, "test-v1")
        self.assertEqual(int.from_bytes(encrypted[4:6], "big"), 384)
        self.assertEqual(decrypt_bytes(encrypted, private, "test-v1"), b"predictions")

    def test_tampering_truncation_and_wrong_context(self) -> None:
        encrypted = encrypt_bytes(b"predictions", self.public, "test-v1")
        for offset in (0, 4, 8, len(encrypted) - 5):
            mutated = bytearray(encrypted)
            mutated[offset] ^= 1
            with self.subTest(offset=offset), self.assertRaises(EnvelopeError):
                decrypt_bytes(bytes(mutated), self.private, "test-v1")
        for invalid in (b"", encrypted[:8], encrypted[:-20]):
            with self.assertRaises(EnvelopeError):
                decrypt_bytes(invalid, self.private, "test-v1")
        with self.assertRaises(EnvelopeError):
            decrypt_bytes(encrypted, self.private, "other-benchmark")
        with self.assertRaises(EnvelopeError):
            decrypt_bytes(encrypted, self.private, "test-v1", "ground_truth")
        other_private, _ = key_pair()
        with self.assertRaises(EnvelopeError):
            decrypt_bytes(encrypted, other_private, "test-v1")

    def test_atomic_writer_refuses_overwrite(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "prediction.enc"
            atomic_write(path, b"first")
            with self.assertRaises(FileExistsError):
                atomic_write(path, b"second")
            self.assertEqual(path.read_bytes(), b"first")
            atomic_write(path, b"third", overwrite=True)
            self.assertEqual(path.read_bytes(), b"third")


class SubmissionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        (self.root / "competition").mkdir()
        (self.root / "competition/config.json").write_text(json.dumps({"benchmark_id": "test-v1"}))
        self.metadata = {"team": "team-one", "type": "human", "model": "none"}
        self.result = {"score": 0.75, "primary_metric": "metric", "metrics": {"metric": 0.75}}

    def record(self, **overrides) -> dict:
        options = dict(result=self.result, metadata=self.metadata, actor="alice", team="team-one", pr_number=1,
                       submission_digest="a" * 64, benchmark_id="test-v1", root=self.root)
        options.update(overrides)
        if "metadata" not in overrides:
            options["metadata"] = {**self.metadata, "team": options["team"]}
        return record_result(**options)

    def test_first_valid_rule_is_per_team_not_github_user(self) -> None:
        original = self.record()
        self.assertEqual(self.record(), original)
        self.assertEqual(self.record(actor="bob"), original)
        for fields in ({"pr_number": 2}, {"actor": "ALICE", "pr_number": 2},
                       {"metadata": {"team": "team-one", "type": "llm", "model": "a-model"}, "pr_number": 3}):
            with self.subTest(fields=fields), self.assertRaises(SubmissionError):
                self.record(**fields)
        self.record(team="team-two", actor="bob", pr_number=2)
        records = json.loads((self.root / "leaderboard/records.json").read_text())
        self.assertEqual(len(records), 2)

    def test_invalid_results_do_not_reserve_an_entry(self) -> None:
        for result in ({"score": float("nan")}, {"score": -0.1}, {"score": 0.5, "secret_labels": [1, 2]},
                       {"score": 0.5, "benchmark_id": "wrong"}, {"score": 0.5, "metrics": {"bad": True}}):
            with self.subTest(result=result), self.assertRaises(SubmissionError):
                self.record(result=result)
        self.assertFalse((self.root / "leaderboard/records.json").exists())
        self.record()

    def test_numeric_precision_and_ties(self) -> None:
        self.record(result={"score": 0.750000001})
        self.record(actor="bob", team="team-two", pr_number=2, result={"score": 0.750000002})
        self.record(actor="carol", team="team-three", pr_number=3, result={"score": 0.750000002})
        rows = list(csv.DictReader(io.StringIO((self.root / "leaderboard/leaderboard.csv").read_text())))
        self.assertEqual([row["rank"] for row in rows], ["1", "1", "3"])
        self.assertEqual(rows[2]["f1_score"], "0.750000001")

    def test_untrusted_metadata_is_plain_text(self) -> None:
        self.record(metadata={"team": "team-one", "type": "llm", "model": "=HYPERLINK(\"bad\")", "notes": "<script>alert(1)</script>|\nhi"})
        rows = list(csv.DictReader(io.StringIO((self.root / "docs/leaderboard.csv").read_text())))
        self.assertTrue(rows[0]["model"].startswith("'="))
        markdown = (self.root / "leaderboard/leaderboard.md").read_text()
        self.assertNotIn("<script>", markdown)
        page = (self.root / "docs/index.html").read_text()
        self.assertIn("textContent", page)
        self.assertNotIn("innerHTML", page)
        with self.assertRaises(SubmissionError):
            parse_metadata({"team": "team-one", "type": "human", "model": "none", "untrusted": "field"})
        for value in ("=cmd", "  +cmd", "@SUM(1)", "\tbad", "\rnew"):
            self.assertTrue(csv_safe(value).startswith("'"))


if __name__ == "__main__":
    unittest.main()
