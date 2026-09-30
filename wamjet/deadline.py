"""Report inference latency relative to the action-chunk duration.

Chunk duration is horizon / control_hz. Keep both workload settings fixed when
comparing optimizations. Deadline headroom is a deployment metric; it does not
decide whether an explicitly requested latency optimization should end.
"""
from __future__ import annotations

import argparse
import math


def analyse(latency_ms: float, horizon: int, control_hz: float,
            pipeline_at: int | None = None) -> dict:
    """Compare a measured latency against a robot's control deadline."""
    chunk_s = horizon / control_hz
    deadline_ms = chunk_s * 1e3
    # with pipelining you must finish before the last `pipeline_at` actions run out
    budget_ms = (pipeline_at / control_hz * 1e3) if pipeline_at else deadline_ms

    achievable_hz = horizon / (latency_ms / 1e3)
    ticks_during_inference = latency_ms / 1e3 * control_hz

    return {
        "latency_ms": latency_ms,
        "horizon": horizon,
        "control_hz": control_hz,
        "pipeline_at": pipeline_at,
        "chunk_duration_ms": deadline_ms,
        "budget_ms": budget_ms,
        "meets_deadline": latency_ms <= budget_ms,
        "slack_ms": budget_ms - latency_ms,
        "utilisation": latency_ms / budget_ms,
        "achievable_control_hz": achievable_hz,
        "headroom_x": achievable_hz / control_hz,
        # levers
        "min_horizon": math.ceil(latency_ms / 1e3 * control_hz),
        "max_latency_ms": budget_ms,
        "speedup_needed": max(1.0, latency_ms / budget_ms),
        # cost of the cheap levers
        "staleness_ms_blocking": latency_ms,
        "staleness_ms_full_chunk": latency_ms + deadline_ms,
        "ticks_stale": ticks_during_inference,
    }


def verdict(r: dict) -> list[str]:
    out = []
    if r["meets_deadline"]:
        out.append(
            f"MEETS DEADLINE at the given horizon, with {r['slack_ms']:.0f} ms to spare "
            f"({r['utilisation']*100:.0f}% of the budget used). The policy produces "
            f"{r['achievable_control_hz']:.1f} actions/s against a {r['control_hz']:.0f} Hz "
            f"loop -- {r['headroom_x']:.1f}x headroom.")
    else:
        out.append(
            f"MISSES DEADLINE by {-r['slack_ms']:.0f} ms. The robot exhausts its "
            f"{r['horizon']}-action chunk in {r['chunk_duration_ms']:.0f} ms and the next "
            f"one is not ready.")
        out.append(
            f"You need {r['speedup_needed']:.2f}x less latency "
            f"({r['latency_ms']:.0f} -> {r['max_latency_ms']:.0f} ms at this horizon).")
    out.append(
        f"STALENESS: {r['ticks_stale']:.1f} control ticks elapse between the observation "
        f"and the first action derived from it ({r['staleness_ms_blocking']:.0f} ms); the "
        f"last action of a chunk is {r['staleness_ms_full_chunk']:.0f} ms old. This is the real "
        f"cost of a long chunk, and the reason the horizon is a behavioural setting rather "
        f"than a tuning knob.")
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--latency-ms", type=float, required=True,
                   help="measured steady-state latency per inference (gate G0)")
    p.add_argument("--horizon", type=int, required=True,
                   help="action steps CONSUMED per inference, as the deployment defines it")
    p.add_argument("--control-hz", type=float, required=True,
                   help="the policy replan rate the robot needs (not the servo rate)")
    p.add_argument("--pipeline-at", type=int,
                   help="if replanning asynchronously, actions still queued when you start")
    a = p.parse_args()

    r = analyse(a.latency_ms, a.horizon, a.control_hz, a.pipeline_at)
    print(f"  latency            {r['latency_ms']:>9.1f} ms")
    print(f"  horizon consumed   {r['horizon']:>9d} actions")
    print(f"  control rate       {r['control_hz']:>9.1f} Hz  "
          f"(chunk lasts {r['chunk_duration_ms']:.0f} ms on the robot)")
    if r["pipeline_at"]:
        print(f"  pipelined at       {r['pipeline_at']:>9d} actions remaining "
              f"-> budget {r['budget_ms']:.0f} ms")
    print(f"  budget used        {r['utilisation']*100:>9.0f}%")
    print(f"  achievable rate    {r['achievable_control_hz']:>9.1f} Hz  "
          f"({r['headroom_x']:.2f}x the requirement)")
    print("\n  VERDICT")
    for line in verdict(r):
        print(f"    - {line}")


if __name__ == "__main__":
    main()
