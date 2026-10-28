"""CPU controls for the actual schedule, admission, cleanup and preservation gates."""

import copy
import importlib.util
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

HERE = Path(__file__).resolve().parent


def module(name):
    spec = importlib.util.spec_from_file_location(name, HERE / (name + ".py"))
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


launcher = module("launch")
preservation = module("preserve")


class Controls(unittest.TestCase):
    def test_actual_frozen_schedule(self):
        plan = launcher.read(HERE / "plan.json")
        source = HERE.parents[5]
        inherited = launcher.read(source / plan["inherited"]["batch"] / "schedule.json")
        schedule = launcher.read(HERE / "schedule.json")
        self.assertEqual(len(launcher.timing_rows(plan, schedule, inherited)), 16)
        for key, value in (
            ("sequence", 9),
            ("arm", "patched"),
            ("position", 1),
            ("block", 1),
            ("capture_order", "winner-first"),
            ("cell", "other"),
        ):
            changed = copy.deepcopy(schedule)
            changed["entries"][0][key] = value
            with self.subTest(key=key), self.assertRaises(AssertionError):
                launcher.timing_rows(plan, changed, inherited)

    def test_deadline_reserves_preservation_time(self):
        self.assertEqual(launcher.worker_budget(1000, 0), 300)
        self.assertEqual(launcher.worker_budget(1000, 850), 60)
        for now in (910, 999, 1001):
            with self.assertRaises(TimeoutError):
                launcher.worker_budget(1000, now)

    def test_source_inventory_rejects_added_or_modified_files(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "input/current"
            source.mkdir(parents=True)
            file = source / "source.py"
            file.write_bytes(b"exact\n")
            launcher.write_new(
                root / "native-reuse-baseline.json",
                {"input_files": {"source.py": launcher.sha(file)}, "input_links": {}},
            )
            self.assertEqual(
                launcher.check_input(root, "baseline")["input_files_verified"], 1
            )
            file.write_bytes(b"changed\n")
            with self.assertRaises(AssertionError):
                launcher.check_input(root, "baseline")
            file.write_bytes(b"exact\n")
            (source / "extra.py").write_bytes(b"extra")
            with self.assertRaises(AssertionError):
                launcher.check_input(root, "baseline")

    def test_cleanup_only_removes_matching_owned_container(self):
        with tempfile.TemporaryDirectory() as directory:
            cid = Path(directory) / "container-id"
            cid.write_text("a" * 64)

            def command(args, **_kwargs):
                return SimpleNamespace(returncode=0, stdout="/ours\n")

            with patch.object(launcher.subprocess, "run", side_effect=command) as run:
                launcher.remove_own_container(cid, "ours")
                self.assertEqual(
                    run.call_args_list[-1].args[0], ["docker", "rm", "-f", "a" * 64]
                )
            with patch.object(
                launcher.subprocess,
                "run",
                return_value=SimpleNamespace(returncode=0, stdout="/someone-else\n"),
            ) as run:
                with self.assertRaises(RuntimeError):
                    launcher.remove_own_container(cid, "ours")
                self.assertEqual(run.call_count, 1)
            cid.write_text("malformed")
            with patch.object(launcher.subprocess, "run") as run:
                with self.assertRaises(RuntimeError):
                    launcher.remove_own_container(cid, "ours")
                run.assert_not_called()

    def test_gate_rejects_changed_result_or_failed_gate(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            entries = []
            for arm in ("baseline", "patched"):
                path = root / (arm + ".json")
                launcher.write_new(path, {"status": "passed"})
                bootstrap = path.with_suffix(".bootstrap.json")
                launcher.write_new(
                    bootstrap,
                    {
                        "status": "passed",
                        "result_sha256": launcher.sha(path),
                        "binding_before": {},
                        "binding_after": {},
                    },
                )
                entries.append(
                    {
                        "phase": "ordering",
                        "returncode": 0,
                        "result_relative_path": path.name,
                        "result_sha256": launcher.sha(path),
                        "bootstrap_sha256": launcher.sha(bootstrap),
                    }
                )
            launcher.write_new(root / "launch-ledger.json", {"entries": entries})
            launcher.write_new(root / "ordering-gate.json", {"status": "passed"})
            launcher.gate(root, "ordering-gate.json", ("ordering",))
            (root / "baseline.json").write_text('{"status":"passed"}\n')
            with self.assertRaises(AssertionError):
                launcher.gate(root, "ordering-gate.json", ("ordering",))
            (root / "ordering-gate.json").write_text('{"status":"failed"}')
            with self.assertRaises(AssertionError):
                launcher.gate(root, "ordering-gate.json", ("ordering",))

    def test_preservation_is_exact_and_never_overwrites(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            evidence = root / "results/delay-binding/attempt01"
            evidence.mkdir(parents=True)
            (evidence / "raw.log").write_bytes(b"exact\x00\xff\r\n")
            destination = root / "archives"
            destination.mkdir()
            result = preservation.preserve(root, "ordering", destination)
            self.assertEqual(result["status"], "verified")
            self.assertEqual(result["files"], 1)
            with self.assertRaises(RuntimeError):
                preservation.preserve(root, "ordering", destination)
            (evidence / "link").symlink_to(evidence / "raw.log")
            with self.assertRaises(RuntimeError):
                preservation.preserve(root, "admission", destination)

    def test_actual_winner_cache_is_preserved_exactly(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            runtime, raw = root / "runtime", root / "raw"
            (runtime / "tuner").mkdir(parents=True)
            (raw / "baseline").mkdir(parents=True)
            cache = runtime / "tuner/cache.json"
            cache.write_bytes(b'{"winner": -1}\n')
            (runtime / "tuner/extra.log").write_bytes(b"keep error evidence\n")
            record = launcher.preserve_tuner(runtime, raw, "baseline/block0")
            self.assertEqual(record["cache_sha256"], launcher.sha(cache))
            self.assertEqual(
                (raw / record["cache_relative_path"]).read_bytes(), cache.read_bytes()
            )
            self.assertEqual(
                (raw / "baseline/block0.tuner/extra.log").read_bytes(),
                b"keep error evidence\n",
            )
            with self.assertRaises(FileExistsError):
                launcher.preserve_tuner(runtime, raw, "baseline/block0")


if __name__ == "__main__":
    unittest.main(verbosity=2)
