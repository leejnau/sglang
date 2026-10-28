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
        schedule = launcher.read(HERE / "schedule.json")
        self.assertEqual(len(launcher.timing_rows(plan, schedule)), 16)
        for key, value in (
            ("sequence", 9),
            ("arm", "v2"),
            ("position", 1),
            ("block", 1),
            ("trial", 1),
            ("phase", "warm"),
            ("capture_order", "invalid"),
            ("mode_order", "invalid"),
        ):
            changed = copy.deepcopy(schedule)
            changed["entries"][0][key] = value
            with self.subTest(key=key), self.assertRaises(AssertionError):
                launcher.timing_rows(plan, changed)

    def test_ordering_uses_its_initialized_result_directory(self):
        """The fixed-only probe must not resolve under a missing treatment arm."""
        with tempfile.TemporaryDirectory() as directory:
            raw = Path(directory)
            for name in ("ordering", "v1", "v2"):
                (raw / name).mkdir()
            for phase, arm, block in (
                ("ordering", "patched", None),
                ("admission", "v1", None),
                ("warm", "v2", None),
                ("timing", "v1", 7),
            ):
                target = raw / (launcher.result_stem(phase, arm, block) + ".json")
                with target.open("x") as output:
                    output.write("exact\n")
                self.assertEqual(target.read_text(), "exact\n")
            self.assertTrue((raw / "ordering/patched.json").is_file())
            self.assertFalse((raw / "patched").exists())

    def test_deadline_reserves_preservation_time(self):
        self.assertEqual(launcher.worker_budget(1000, 0), 300)
        self.assertEqual(launcher.worker_budget(1000, 850), 60)
        for now in (910, 999, 1001):
            with self.assertRaises(TimeoutError):
                launcher.worker_budget(1000, now)

    def test_source_inventory_rejects_added_or_modified_files(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "input/patched"
            source.mkdir(parents=True)
            file = source / "source.py"
            file.write_bytes(b"exact\n")
            launcher.write_new(
                root / "native-reuse-patched.json",
                {"input_files": {"source.py": launcher.sha(file)}, "input_links": {}},
            )
            self.assertEqual(
                launcher.check_input(root, "v1")["input_files_verified"], 1
            )
            file.write_bytes(b"changed\n")
            with self.assertRaises(AssertionError):
                launcher.check_input(root, "v1")
            file.write_bytes(b"exact\n")
            (source / "extra.py").write_bytes(b"extra")
            with self.assertRaises(AssertionError):
                launcher.check_input(root, "v1")

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
            for arm in ("v1", "v2"):
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
                        "phase": "admission",
                        "returncode": 0,
                        "result_relative_path": path.name,
                        "result_sha256": launcher.sha(path),
                        "bootstrap_sha256": launcher.sha(bootstrap),
                    }
                )
            launcher.write_new(root / "launch-ledger.json", {"entries": entries})
            launcher.write_new(root / "admission-analysis.json", {"status": "passed"})
            launcher.gate(root, "admission-analysis.json", ("admission",))
            (root / "v1.json").write_text('{"status":"passed"}\n')
            with self.assertRaises(AssertionError):
                launcher.gate(root, "admission-analysis.json", ("admission",))
            (root / "admission-analysis.json").write_text('{"status":"failed"}')
            with self.assertRaises(AssertionError):
                launcher.gate(root, "admission-analysis.json", ("admission",))

    def test_preservation_is_exact_and_never_overwrites(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            evidence = root / "results/v1-v2-corrected/attempt01"
            evidence.mkdir(parents=True)
            (evidence / "raw.log").write_bytes(b"exact\x00\xff\r\n")
            destination = root / "archives"
            destination.mkdir()
            result = preservation.preserve(root, "admission", destination)
            self.assertEqual(result["status"], "verified")
            self.assertEqual(result["files"], 1)
            with self.assertRaises(RuntimeError):
                preservation.preserve(root, "admission", destination)
            (evidence / "link").symlink_to(evidence / "raw.log")
            with self.assertRaises(RuntimeError):
                preservation.preserve(root, "warm", destination)

    def test_actual_winner_cache_is_preserved_exactly(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            runtime, raw = root / "runtime", root / "raw"
            (runtime / "tuner").mkdir(parents=True)
            (raw / "v1").mkdir(parents=True)
            cache = runtime / "tuner/cache.json"
            cache.write_bytes(b'{"winner": -1}\n')
            (runtime / "tuner/extra.log").write_bytes(b"keep error evidence\n")
            record = launcher.preserve_tuner(runtime, raw, "v1/block0")
            self.assertEqual(record["cache_sha256"], launcher.sha(cache))
            self.assertEqual(
                (raw / record["cache_relative_path"]).read_bytes(), cache.read_bytes()
            )
            self.assertEqual(
                (raw / "v1/block0.tuner/extra.log").read_bytes(),
                b"keep error evidence\n",
            )
            with self.assertRaises(FileExistsError):
                launcher.preserve_tuner(runtime, raw, "v1/block0")

    def test_warm_process_consumes_only_exact_admission_cache(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            arm = root / "v1"
            arm.mkdir()
            seed = arm / "admission-auto_synthetic.cache.json"
            seed.write_bytes(b'{"winner": 12}\n')
            launcher.write_new(
                arm / "admission-auto_synthetic.json",
                {"cache": {"final_sha256": launcher.sha(seed)}},
            )
            target = root / "fresh-cache.json"
            record = launcher.seed_warm_cache(root, "v1", target)
            self.assertEqual(target.read_bytes(), seed.read_bytes())
            self.assertEqual(record["cache_seed_sha256"], launcher.sha(seed))
            with self.assertRaises(FileExistsError):
                launcher.seed_warm_cache(root, "v1", target)
            seed.write_bytes(b'{"winner": -1}\n')
            with self.assertRaises(AssertionError):
                launcher.seed_warm_cache(root, "v1", root / "other.json")
            self.assertFalse((root / "other.json").exists())


if __name__ == "__main__":
    unittest.main(verbosity=2)
