"""Replay recorded timings through upstream selection code without CUDA."""

import argparse
import ast
import hashlib
import json
import statistics
from pathlib import Path
from types import SimpleNamespace


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--release", type=Path, required=True)
    parser.add_argument("--upstream", type=Path, required=True)
    parser.add_argument("--cold-result", type=Path, required=True)
    parser.add_argument("--warm-result", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    recorded = json.loads(args.warm_result.read_text())
    for key, relative in (
        ("flashinfer_tuner", "flashinfer/autotuner/autotuner.py"),
        ("flashinfer_cache", "flashinfer/autotune_cache.py"),
    ):
        assert (
            sha256(args.release / relative) == recorded["source_files"][key]["sha256"]
        )

    path = args.upstream / "flashinfer/autotuner/autotuner.py"
    tree = ast.parse(path.read_text())
    cls = next(
        n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "AutoTuner"
    )
    method = next(
        n
        for n in cls.body
        if isinstance(n, ast.FunctionDef) and n.name == "_settle_near_ties"
    )
    module = ast.Module(
        body=[
            ast.ImportFrom(
                module="__future__", names=[ast.alias(name="annotations")], level=0
            ),
            method,
        ],
        type_ignores=[],
    )
    namespace = {"statistics": statistics}
    exec(compile(ast.fix_missing_locations(module), str(path), "exec"), namespace)

    records = json.loads(args.cold_result.read_text())["profiles"]
    groups = []
    for row in records:
        if row["shapes"][:2] != [[1, 1, 14336], [1, 14336, 4096]]:
            continue
        if row["tactic"] == "-1":
            groups.append([])
        groups[-1].append(row)
    assert len(groups) == 2 and all(len(group) == 9 for group in groups)

    def no_retime(*_):
        raise AssertionError("Zero tolerance must not add measurements")

    results = []
    for phase, rows in zip(("legacy startup", "scoped startup"), groups):
        measured = [(row["ms"], 0, int(row["tactic"])) for row in rows]
        winner = namespace["_settle_near_ties"](
            SimpleNamespace(_TIE_MAX_CONTENDERS=4), measured, no_retime, 0.0
        )
        assert winner == min(range(len(measured)), key=lambda i: measured[i][0])
        best, _, tactic = measured[winner]
        runner_up = min(t for i, (t, _, _) in enumerate(measured) if i != winner)
        results.append(
            {
                "phase": phase,
                "chosen_tactic": tactic,
                "best_us": 1000 * best,
                "runner_up_us": 1000 * runner_up,
                "margin_ns": 1e6 * (runner_up - best),
                "margin_percent": 100 * (runner_up / best - 1),
                "times_us": {str(tac): t * 1000 for t, _, tac in measured},
                "upstream_default_matches_release_argmin": True,
            }
        )
    assert [row["chosen_tactic"] for row in results] == [-1, 1]
    report = {
        "cold_result_sha256": sha256(args.cold_result),
        "warm_result_sha256": sha256(args.warm_result),
        "upstream_tuner_sha256": sha256(path),
        "recorded_release_source_hashes_match": True,
        "results": results,
        "limits": "Offline replay through an extracted production method; no new profiling, CUDA execution, or evidence that an optional positive tie tolerance improves serving.",
    }
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
