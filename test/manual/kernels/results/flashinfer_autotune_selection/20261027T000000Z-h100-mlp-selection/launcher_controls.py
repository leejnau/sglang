"""CPU controls for the actual launcher's schedule and resume guards."""

import ast
import hashlib
import json
import subprocess
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

root = Path(__file__).resolve().parent
launcher = (root / "launch.sh").read_text()
batch = Path(sys.argv[1])


def heredoc(marker):
    return launcher.split("<<'" + marker + "'\n", 1)[1].split("\n" + marker, 1)[0]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


checks = {}
for mode, count in (("admission", 6), ("timing", 36)):
    schedule = subprocess.run(
        [sys.executable, "-", str(batch / "schedule.json"), mode],
        input=heredoc("PY_SCHEDULE"),
        text=True,
        capture_output=True,
        check=True,
    )
    rows = [line.split("\t") for line in schedule.stdout.splitlines()]
    assert len(rows) == count
    assert [row[-1] for row in rows] == [
        ("admission-" if mode == "admission" else "") + entry["id"]
        for entry in json.loads((batch / "schedule.json").read_text())["entries"]
        if mode == "timing" or entry["trial"] == 0
    ]
    checks[mode + "_actual_schedule"] = "passed"
tree = ast.parse((root / "run_current.py").read_text())
nodes = [
    node
    for node in tree.body
    if isinstance(node, ast.FunctionDef)
    and node.name in ("verify_native_files", "require_ninja_no_work")
]
namespace = {
    "provider": SimpleNamespace(digest=digest),
    "REFERENCE_SHA": "7c4727aca3d99adcb3fe2f35d06f4408b512baa4fe9a91b975bdf77358fe205d",
}
exec(
    compile(
        ast.Module(body=nodes, type_ignores=[]), str(root / "run_current.py"), "exec"
    ),
    namespace,
)
reference_sha = namespace["REFERENCE_SHA"]
native_hashes = json.loads((batch / "plan.json").read_text())["native_reuse"][
    "required_library_sha256"
]
with tempfile.TemporaryDirectory(prefix="fi-mlp-selection-launcher-") as temporary:
    folder = Path(temporary)
    names = [
        f"admission-trial0-{arm}-{phase}"
        for arm in ("default", "v1", "v2")
        for phase in ("cold", "warm")
    ]
    analysis = folder / "admission-analysis.json"
    outer = folder / "admission-outer-audit.json"
    reuse_manifest = folder / "native-build-cache/reuse-manifest.json"
    reuse_manifest.parent.mkdir()
    reuse_manifest.write_text("{}")
    for case in (
        "valid",
        "missing_analysis",
        "missing_outer",
        "failed_analysis",
        "failed_outer",
        "omitted_worker",
        "omitted_outer_worker",
        "changed_result",
        "changed_bootstrap",
        "bad_bootstrap_result_binding",
        "failed_worker",
        "changed_native_reference",
        "changed_reuse_manifest",
        "failed_post_worker_native_guard",
        "wrong_native_elf",
    ):
        rows, audited = [], []
        for index, name in enumerate(names):
            result = folder / (name + ".json")
            bootstrap = result.with_suffix(".bootstrap.json")
            result.write_text(
                json.dumps(
                    {
                        "status": "failed"
                        if case == "failed_worker" and index == 0
                        else "success"
                    }
                )
                + "\n"
            )
            data = {
                "status": "passed",
                "result_sha256": digest(result),
                "native_reuse": {
                    "reference_sha256": reference_sha,
                    "manifest_sha256": digest(reuse_manifest),
                    **{
                        point: {"status": "passed", "elf_sha256": native_hashes}
                        for point in ("before", "after_build", "after_worker")
                    },
                },
            }
            if index == 0:
                if case == "changed_native_reference":
                    data["native_reuse"]["reference_sha256"] = "0" * 64
                elif case == "changed_reuse_manifest":
                    data["native_reuse"]["manifest_sha256"] = "0" * 64
                elif case == "failed_post_worker_native_guard":
                    data["native_reuse"]["after_worker"]["status"] = "failed"
                elif case == "wrong_native_elf":
                    data["native_reuse"]["after_build"]["elf_sha256"] = {}
                elif case == "bad_bootstrap_result_binding":
                    data["result_sha256"] = "0" * 64
            bootstrap.write_text(json.dumps(data) + "\n")
            rows.append(
                {
                    "name": name,
                    "result_sha256": digest(result),
                    "bootstrap_sha256": digest(bootstrap),
                }
            )
            audited.append(
                {
                    "name": name,
                    "raw_sha256": digest(result),
                    "bootstrap_sha256": digest(bootstrap),
                }
            )
            if index == 0 and case == "changed_result":
                result.write_text(result.read_text() + " ")
            if index == 0 and case == "changed_bootstrap":
                bootstrap.write_text(bootstrap.read_text() + " ")
        analysis.write_text(
            json.dumps(
                {
                    "status": "failed" if case == "failed_analysis" else "passed",
                    "results": rows[1:] if case == "omitted_worker" else rows,
                }
            )
        )
        outer.write_text(
            json.dumps(
                {
                    "status": "failed" if case == "failed_outer" else "passed",
                    "phase": "admission",
                    "workers": audited[1:]
                    if case == "omitted_outer_worker"
                    else audited,
                }
            )
        )
        if case == "missing_analysis":
            analysis.unlink()
        if case == "missing_outer":
            outer.unlink()
        completed = subprocess.run(
            [sys.executable, "-", temporary],
            input=heredoc("ADMISSION"),
            text=True,
            capture_output=True,
        )
        assert (completed.returncode == 0) == (case == "valid"), case + completed.stderr
        checks[case] = "passed"

    prefix = "current/.cache/flashinfer/0.0.0+unknown/90a/cached_ops/"
    cache = folder / "cache"
    files = {}
    for name in ("gemm", "trtllm_utils"):
        path = cache / prefix / name / (name + ".so")
        path.parent.mkdir(parents=True)
        path.write_bytes(name.encode())
        files[str(path.relative_to(cache))] = digest(path)
    namespace["NATIVE_HASHES"] = {
        name: files[prefix + name + "/" + name + ".so"]
        for name in ("gemm", "trtllm_utils")
    }
    manifest = {
        "status": "passed",
        "native_reference_sha256": reference_sha,
        "native_files": files,
    }
    guard = namespace["verify_native_files"]
    assert guard(cache, manifest)["status"] == "passed"
    checks["native_valid_files"] = "passed"
    target = cache / prefix / "gemm/gemm.so"
    for case in (
        "native_changed_bytes",
        "native_extra_file",
        "native_missing_file",
        "native_wrong_reference",
    ):
        original = target.read_bytes()
        extra = target.parent / "unexpected.o"
        if case == "native_changed_bytes":
            target.write_bytes(b"changed")
        elif case == "native_extra_file":
            extra.write_bytes(b"unexpected")
        elif case == "native_missing_file":
            target.unlink()
        else:
            manifest["native_reference_sha256"] = "0" * 64
        try:
            guard(cache, manifest)
        except (RuntimeError, FileNotFoundError):
            checks[case] = "passed"
        else:
            raise AssertionError(case)
        target.write_bytes(original)
        extra.unlink(missing_ok=True)
        manifest["native_reference_sha256"] = reference_sha

    spec = SimpleNamespace(
        name="gemm",
        build_dir=Path("/native/gemm"),
        ninja_path=Path("/native/gemm/build.ninja"),
    )
    for case, stdout, stderr, accepted in (
        (
            "ninja_no_work",
            "ninja: Entering directory `/native/gemm'\nninja: no work to do.\n",
            "",
            True,
        ),
        (
            "ninja_rebuild_rejected",
            "ninja: Entering directory `/native/gemm'\n[1/1] compile\n",
            "",
            False,
        ),
        (
            "ninja_warning_rejected",
            "ninja: Entering directory `/native/gemm'\nninja: no work to do.\n",
            "warning",
            False,
        ),
    ):
        namespace["subprocess"] = SimpleNamespace(
            run=lambda *args, **kwargs: SimpleNamespace(
                stdout=stdout, stderr=stderr, returncode=0
            )
        )
        try:
            namespace["require_ninja_no_work"](spec)
        except RuntimeError:
            assert not accepted, case
        else:
            assert accepted, case
        checks[case] = "passed"

report = {
    "status": "passed",
    "checks": checks,
    "launcher_sha256": digest(root / "launch.sh"),
    "bootstrap_sha256": digest(root / "run_current.py"),
    "scope": "Actual embedded launcher code with controlled JSON and the frozen schedule; no Docker, GPU, allocation or worker execution.",
}
(root / "launcher-cpu-controls.json").write_text(json.dumps(report, indent=2) + "\n")
print(json.dumps(report))
