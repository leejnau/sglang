"""Descriptive, existing-data-only review of one saved ranking reversal."""

import argparse
import hashlib
import json
import math
import statistics
from pathlib import Path


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def extent(values):
    return {
        "minimum": min(values),
        "median": statistics.median(values),
        "maximum": max(values),
    }


def ratio(pairs, first=None):
    values = []
    for pair in pairs:
        if first is not None and pair["order"][0] != first:
            continue
        rows = {x["role"]: x for x in pair["measurements"]}
        values.append(
            math.log(rows["winner"]["elapsed_ms"] / rows["default"]["elapsed_ms"])
        )
    return math.exp(statistics.median(values))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--raw-root",
        type=Path,
        required=True,
        help="Directory containing the eight saved worker JSONs and bootstrap records.",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    workers, hashes, envelopes, comparison = [], {}, [], []
    for block in range(8):
        name = f"block{block}-auto_synthetic-timing"
        path = args.raw_root / (name + ".json")
        w = json.loads(path.read_text())
        workers.append(w)
        hashes[path.name] = sha(path)
        bootstrap = args.raw_root / (name + ".bootstrap.json")
        hashes[bootstrap.name] = sha(bootstrap)
        boot = json.loads(bootstrap.read_text())
        assert w["status"] == "success" and w["block"] == block
        assert (
            len(w["startup"]["profiles"]) == 81
            and len(w["confirmation"]["rounds"]) == 12
        )
        assert len(w["diagnostic_profile_attempts"]) == 24
        assert boot["result_sha256"] == hashes[path.name]
        correct = {x["check"]: x for x in w["profile_correctness"]}
        values = {"winner": [], "default": []}
        for attempt in w["diagnostic_profile_attempts"]:
            check = correct[attempt["stage"]]
            role = attempt["stage"].rsplit("_", 1)[1]
            assert check["checked_ns"] > attempt["ended_ns"] > attempt["started_ns"]
            assert attempt["metadata"] == w["retained_profile"]["metadata"]
            assert (
                attempt["tensor_object_ids"]
                == w["retained_profile"]["tensor_object_ids"]
            )
            assert attempt["settings"] == w["confirmation"]["settings"]
            assert attempt["error"] is None and check["passed"]
            envelopes.append(
                dict(
                    block=block,
                    attempt_index=attempt["index"],
                    stage=attempt["stage"],
                    role=role,
                    tactic=attempt["tactic"],
                    returned_profiler_average_us=attempt["elapsed_ms"] * 1000,
                    host_profile_envelope_us=(
                        attempt["ended_ns"] - attempt["started_ns"]
                    )
                    / 1000,
                    post_return_to_correctness_us=(
                        check["checked_ns"] - attempt["ended_ns"]
                    )
                    / 1000,
                    started_ns=attempt["started_ns"],
                    returned_ns=attempt["ended_ns"],
                    correctness_checked_ns=check["checked_ns"],
                )
            )
            values[role].append(attempt["elapsed_ms"] * 1000)
        p = w["startup"]["profiles"]
        finalist = p[w["finalist"]["profile_index"]]
        assert all(x["metadata"] == w["retained_profile"]["metadata"] for x in p[72:81])
        assert all(x["settings"] == w["confirmation"]["settings"] for x in p[72:81])
        pairlogs = []
        for pair in w["confirmation"]["rounds"]:
            rows = {x["role"]: x for x in pair["measurements"]}
            assert (
                rows["winner"]["tactic"] == w["finalist"]["tactic"]
                and rows["default"]["tactic"] == -1
            )
            pairlogs.append(
                math.log(rows["winner"]["elapsed_ms"] / rows["default"]["elapsed_ms"])
            )
        comparison.append(
            dict(
                block=block,
                capture_order=w["capture_order"],
                finalist_profile_index=finalist["index"],
                finalist_tactic=finalist["tactic"],
                first_pass_default_us=p[72]["elapsed_ms"] * 1000,
                first_pass_finalist_us=finalist["elapsed_ms"] * 1000,
                first_pass_finalist_over_default=finalist["elapsed_ms"]
                / p[72]["elapsed_ms"],
                original_choice="default"
                if w["natural"]["tactic"] == -1
                else "explicit",
                confirmation_finalist_over_default=ratio(w["confirmation"]["rounds"]),
                confirmation_default_first_ratio=ratio(
                    w["confirmation"]["rounds"], "default"
                ),
                confirmation_explicit_first_ratio=ratio(
                    w["confirmation"]["rounds"], "winner"
                ),
                confirmation_explicit_faster_pairs=sum(x < 0 for x in pairlogs),
                confirmation_average_us_by_role={
                    role: extent(data) for role, data in values.items()
                },
                retained_layer_graph_finalist_over_default=w[
                    "selected_over_default_graph_event_median"
                ],
                native_elf_sha256={
                    key: value["library_sha256"]
                    for key, value in boot["native"]["providers"].items()
                },
            )
        )
    assert len(envelopes) == 192
    w = workers[5]
    p = w["startup"]["profiles"]
    c = w["confirmation"]
    metadata = w["retained_profile"]["metadata"]
    pairs = []
    for pair in c["rounds"]:
        measurements = {x["role"]: x for x in pair["measurements"]}
        pairs.append(
            dict(
                index=pair["index"],
                order=pair["order"],
                default_us=measurements["default"]["elapsed_ms"] * 1000,
                explicit_us=measurements["winner"]["elapsed_ms"] * 1000,
                explicit_over_default=measurements["winner"]["elapsed_ms"]
                / measurements["default"]["elapsed_ms"],
            )
        )
    retained_end = w["retained_profile"]["retained_after_ns"]
    fixed = {
        "geometry": [144, 1024, 3072],
        "actual_profile_m": 144,
        "canonical_cache_m": 256,
        "settings": c["settings"],
        "same_settings_for_nine_original_target_calls_and_24_confirmation_calls": True,
        "same_six_shapes_strides_dtypes_and_pointers_for_original_target_and_confirmation": True,
        "retained_tensor_object_ids_unchanged_during_confirmation": True,
        "original_default_index": 72,
        "original_default_position_zero_based": 0,
        "original_finalist_index": 75,
        "original_finalist_position_zero_based": 3,
        "finalist_is_third_explicit": True,
        "same_finalist_descriptor_initial_and_confirmation": w["finalist"]["tactic"],
        "earlier_identical_descriptor_records": [
            dict(index=x["index"], m=x["m"], elapsed_us=x["elapsed_ms"] * 1000)
            for x in p
            if x["tactic"] == w["finalist"]["tactic"]
        ],
        "original_cached_choice_preserved": w["natural"][
            "cache_preserved_after_confirmation"
        ]
        and w["final"]["cache_unchanged"],
        "synthetic_read_operand_hashes_before_confirmation": w["retained_profile"][
            "read_operand_hashes"
        ],
        "synthetic_read_operands_unchanged_at_end": w["final"]["operands_unchanged"],
        "same_synthetic_hashes_across_eight_processes": len(
            {tuple(x["retained_profile"]["read_operand_hashes"]) for x in workers}
        )
        == 1,
        "synthetic_pointer_metadata": metadata,
        "native_explicit_acceptance_after_all_timings": {
            "serving_inputs": w["post_timing_acceptance"]["used_descriptor"],
            "retained_synthetic_inputs": w["synthetic_acceptance"][0][
                "used_descriptor"
            ],
        },
        "qualified_source": w["source"],
        "fixture": w["fixture"],
    }
    changed = [
        "Initial target default follows72 earlier candidate profiles and is always first within its bucket; finalist follows two other explicit candidates. Initial order was not counterbalanced.",
        "Confirmation follows completion of all81 original profiles, original cache publication/return, retention of final operands, GPU-to-CPU read hashes and an FP32 reference bmm.",
        "Confirmation uses12 balanced two-candidate pairs, with output NaN fill and device-wide synchronization before each profiler call and device-wide synchronization/numerical checks after return.",
        "The initial race has one returned average per candidate. Confirmation has12 returned averages per candidate; it is a changed-history repeated comparison, not12 untouched reproductions of the initial race.",
        "Serving graphs are captured only after confirmation. They include real captured-activation quantization plus GEMM, new input/output/workspace allocations and a separate queued16-replay measurement objective.",
    ]
    unobserved = [
        "No individual warmup or measured-kernel durations are logged inside any returned profiler average; each value summarizes ten calls after three warmups.",
        "Original81 rows have no per-candidate host start/end timestamp. Index establishes order but not precise timing gaps.",
        "No actual profiler-window delay-kernel duration/stream, start-event queue-completion query or per-window device clock/power/temperature is recorded.",
        "Actual submitted algorithms and CUDA kernel identities were not traced during original or confirmation profiling. Public default tactic -1 alone is not a native algorithm identity.",
        "Synthetic values were not byte-hashed during original profiling. Pointer/layout equality and the source read-only contract are observed; direct byte attestation starts only after the initial race.",
        "The post-timing trace identifies the two retained serving graphs only; it cannot retrospectively identify every original/confirmation profiler launch.",
    ]
    sequence = [
        dict(
            index=x["index"],
            m=x["m"],
            position_in_bucket=x["index"] % 9,
            tactic=x["tactic"],
            elapsed_us=x["elapsed_ms"] * 1000,
            error=x["error"],
        )
        for x in p
    ]
    host = extent([x["host_profile_envelope_us"] for x in envelopes])
    post = extent([x["post_return_to_correctness_us"] for x in envelopes])
    report = dict(
        schema_version=1,
        status="complete_existing_data_only",
        target_block=5,
        input_sha256=hashes,
        scope="Descriptive audit of the prespecified eight-process cohort; no new tests, exclusion, trimming, hypothesis test, protocol change or experiment.",
        fixed_observations=fixed,
        changed_execution_context=changed,
        unobserved=unobserved,
        block5=dict(
            original_full_sequence=sequence,
            confirmation_pairs=pairs,
            original_ratio=w["original_history"]["explicit_over_default"],
            confirmation_ratio=ratio(c["rounds"]),
            serving_graph_ratio=w["selected_over_default_graph_event_median"],
            trace=w["trace"],
            chronology_ns={
                "startup_started": w["startup"]["started_ns"],
                "retained_after_last_profile": retained_end,
                "startup_ended": w["startup"]["ended_ns"],
                "reference_preparation_started": w["retained_profile"][
                    "reference_preparation_started_ns"
                ],
                "reference_preparation_ended": w["retained_profile"][
                    "reference_preparation_ended_ns"
                ],
                "confirmation_started": c["started_ns"],
                "confirmation_ended": c["ended_ns"],
                "decision_finalized": c["decided_ns"],
            },
            default_first_ratio=ratio(c["rounds"], "default"),
            explicit_first_ratio=ratio(c["rounds"], "winner"),
        ),
        all_eight_process_comparison=comparison,
        all_192_profiler_host_envelopes=envelopes,
        host_envelope_us=host,
        post_return_to_correctness_us=post,
        envelope_interpretation="Host profiler envelopes omit most of the subsequent roughly5ms synchronization/check interval despite a configured5000us delay. This exposes a stream/measurement-boundary observation gap. The post-return interval includes GPU-wide synchronization AND numerical work; it is not a direct measurement of the delay kernel or proof of a wrong stream.",
        strongest_descriptive_pattern="Block5 initial default10.1984us is higher than the other seven initial default averages9.2480–9.6160us. Its finalist9.5648us is within the other seven finalist range9.5072–9.8464us. Every process later favors default by its median confirmation ratio; block5 both pair-order strata also favor default. These comparisons do not identify the cause.",
        smallest_missing_observation="Observe the actual configured delay-kernel stream and completion relative to the timed-event stream in the original/confirmation profiler, including the first measured window. Bind that to the actual native algorithm/kernel identities rather than adding more candidate-average samples. Saved data alone cannot supply it; no new run is proposed or authorized here.",
        retained_decision="Preserve the inconclusive-stop outcome and unchanged production selection. Do not diagnose history, noise, input distribution or stream behavior as the cause from this audit alone.",
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as handle:
        json.dump(report, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")


if __name__ == "__main__":
    main()
