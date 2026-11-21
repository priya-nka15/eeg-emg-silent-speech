#!/usr/bin/env python3
"""
URF/A-SSR Hardware Validation Script
PRD §9, §54 Gate 3 — Runs all hardware validation gates H1–H8.

Usage:
    python scripts/validate_hardware.py [--samples-csv PATH]
                                        [--expected-rate HZ]
                                        [--adc-max ADC_MAX]
                                        [--output JSON_PATH]

This script CANNOT manufacture hardware results.
Gates that require physical hardware return NOT_STARTED or BLOCKED.
Human data collection must not begin until all gates return PASS.

Outputs:
    PASS         — gate requirement met
    FAIL         — gate requirement not met
    NOT_STARTED  — requires physical hardware (cannot be evaluated in software)
    BLOCKED      — blocked on TBD hardware value (e.g. adc_max not known)
    INFORMATIONAL — gate provides data but has no software pass/fail criterion

Structural sync check (H3) always returns NOT_STARTED because physical
simultaneous-acquisition verification requires hardware measurement, not
just inspecting data fields.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

# Add repo root to path so we can import backend
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backend.acquisition.hardware_validator import HardwareValidator, GateResult
from backend.acquisition.frame_parser import Sample


_GATE_DESCRIPTIONS = {
    "H1_channel_mapping":     "CH mapping — verifies each channel maps to the correct anatomical location",
    "H2_sampling_rate":       "Sampling rate — measures actual SPS from timestamps and counter",
    "H3_synchronization":     "Synchronisation — structural check (physical simultaneity requires hardware)",
    "H4_dropped_samples":     "Dropped samples — detects counter gaps",
    "H5_saturation":          "Saturation — detects repeated ADC limit hits (adc_max required)",
    "H6_flatline":            "Flatline — detects suspiciously low-variance channels",
    "H7_noise":               "Noise — characterises electronics-only / contact / signal noise",
    "H8_sustained_streaming": "Sustained streaming — stress test for buffer overflow / firmware stability",
}

_RESULT_MARKER = {
    GateResult.PASS:        "  PASS        ",
    GateResult.FAIL:        "  FAIL !!!    ",
    GateResult.NOT_STARTED: "  NOT_STARTED ",
    GateResult.BLOCKED:     "  BLOCKED     ",
}


def load_samples_from_csv(path: Path) -> list[Sample]:
    """Load samples from a raw recording CSV."""
    samples = []
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            samples.append(Sample.from_dict(row))
    return samples


def run(
    samples_csv: Path | None,
    expected_rate_hz: float | None,
    adc_max: int | None,
    output_path: Path | None,
    mock: bool = False,
) -> int:
    """Run all gates and return exit code (0 = all PASS or NOT_STARTED, 1 = any FAIL)."""
    hv = HardwareValidator()
    samples = None

    if mock:
        print("Running in MOCK mode — generating synthetic samples from MockSource.")
        from backend.realtime.mock_source import MockSource
        rate = expected_rate_hz or 2000.0
        src = MockSource(sampling_rate_hz=rate)
        samples = []
        for i in range(2000):
            s = src.generate_sample()
            s.sample_id = i
            s.timestamp_us = int(i * (1_000_000 / rate))
            samples.append(s)
        print(f"  {len(samples)} mock samples generated.")
        if expected_rate_hz is None:
            expected_rate_hz = rate
    elif samples_csv is not None:
        print(f"Loading samples from: {samples_csv}")
        samples = load_samples_from_csv(samples_csv)
        print(f"  {len(samples)} samples loaded.")
    else:
        print("No samples CSV provided — hardware-dependent gates will return NOT_STARTED.")

    if expected_rate_hz is None:
        print(
            "NOTE: --expected-rate not provided. H8 will return NOT_STARTED because "
            "the expected sample rate must come from the hardware specification, not "
            "from a default assumption (PRD rule 1)."
        )

    results = hv.run_all(
        samples=samples,
        adc_max=adc_max,
        expected_sampling_rate_hz=expected_rate_hz,
    )

    print("\n" + "=" * 70)
    print("URF/A-SSR Hardware Validation Report")
    print("=" * 70)

    any_fail = False
    report_gates = []

    for r in results:
        marker = _RESULT_MARKER.get(r.result, f"  {r.result!s:<13}")
        desc = _GATE_DESCRIPTIONS.get(r.gate, r.gate)
        print(f"{marker} {desc}")
        if r.result == GateResult.FAIL:
            any_fail = True
            print(f"           Reason: {r.description}")
            if r.details:
                print(f"           Details: {json.dumps(r.details, indent=12)}")
        elif r.result in (GateResult.NOT_STARTED, GateResult.BLOCKED):
            print(f"           {r.description}")
            # Print structural check results from H3 even though gate is NOT_STARTED
            if r.gate == "H3_synchronization" and r.details.get("structural_issues"):
                print(f"           Structural issues: {r.details['structural_issues']}")
            elif r.gate == "H3_synchronization" and r.details:
                sc = r.details.get("structural_check_passed")
                if sc is not None:
                    print(f"           Structural check passed: {sc}")

        report_gates.append(r.to_dict())

    print("=" * 70)
    all_pass = all(r.result == GateResult.PASS for r in results)
    n_pass = sum(1 for r in results if r.result == GateResult.PASS)
    n_not_started = sum(1 for r in results if r.result == GateResult.NOT_STARTED)
    n_blocked = sum(1 for r in results if r.result == GateResult.BLOCKED)
    n_fail = sum(1 for r in results if r.result == GateResult.FAIL)

    print(f"\nSummary:  PASS={n_pass}  NOT_STARTED={n_not_started}  "
          f"BLOCKED={n_blocked}  FAIL={n_fail}")

    if all_pass:
        print("\nAll gates PASSED. Human data collection may proceed (subject to")
        print("human_use_ready=true and institutional safety review).")
    elif n_fail > 0:
        print(f"\n{n_fail} gate(s) FAILED. Do not proceed with human data collection.")
    else:
        print(
            "\nNo gates FAILED. Remaining NOT_STARTED/BLOCKED gates require "
            "physical hardware or hardware specification values."
        )
        print("Human data collection must not begin until all gates return PASS.")

    if output_path:
        report = {
            "report_version": "1",
            "human_data_collection_approved": all_pass,
            "summary": {
                "pass": n_pass,
                "not_started": n_not_started,
                "blocked": n_blocked,
                "fail": n_fail,
            },
            "gates": report_gates,
        }
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(report, f, indent=2)
        print(f"\nReport saved to: {output_path}")

    return 1 if any_fail else 0


def main() -> None:
    parser = argparse.ArgumentParser(
        description="URF/A-SSR Hardware Validation (PRD §9, Gates H1–H8)",
    )
    parser.add_argument(
        "--samples-csv",
        type=Path,
        default=None,
        help="Path to a raw recording CSV. If not provided, hardware-dependent "
             "gates return NOT_STARTED.",
    )
    parser.add_argument(
        "--expected-rate",
        type=float,
        default=None,
        help="Expected sampling rate in Hz from hardware specification. "
             "Required for H8. Do not use the 4 kSPS target as a substitute "
             "for the measured rate (PRD rule 1).",
    )
    parser.add_argument(
        "--adc-max",
        type=int,
        default=None,
        help="ADC saturation value (HARDWARE-TBD). Required for H5.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Path to save JSON validation report.",
    )
    parser.add_argument(
        "--mock",
        action="store_true",
        help="Run validation in mock mode using synthetic samples.",
    )
    args = parser.parse_args()
    sys.exit(run(args.samples_csv, args.expected_rate, args.adc_max, args.output, mock=args.mock))


if __name__ == "__main__":
    main()
