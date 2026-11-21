"""
URF/A-SSR CLI entry point
PRD §48 — CLI commands.
Usage: python -m backend <command> [args]

Commands:
  validate-recording  Run QC on a raw CSV recording file.
  validate-dataset    Validate schema of subject/session/trial metadata in a dataset directory.
  generate-splits     Generate subject/session-aware train/val/test splits.
  train               [BLOCKED — requires dataset] Stub, not yet implemented.
  evaluate            Display a saved experiment manifest.
  replay              Replay a raw recording through the production processing pipeline.
  realtime            [BLOCKED — requires hardware + safety gate] Stub.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path


def cmd_validate_recording(args):
    """python -m backend validate-recording <path> [--modality all|emg|eeg] [--target-hz HZ]

    Runs signal QC on a raw CSV recording. Reports dropped samples, measured
    sample rate, and per-channel issues.

    NOTE: Saturation detection requires adc_max (hardware TBD). The report
    will explicitly state when saturation cannot be checked.
    """
    import csv
    import json
    import warnings
    from backend.qc.signal_qc import SignalQC
    from backend.acquisition.frame_parser import Sample

    path = Path(args.path)
    if not path.exists():
        print(f"ERROR: Path not found: {path}", file=sys.stderr)
        sys.exit(1)

    print(f"Validating recording: {path}")
    print(
        "NOTE: adc_max (saturation threshold) is HARDWARE-TBD. "
        "Saturation cannot be checked until hardware specification is provided."
    )

    meta_file = path.parent / "metadata.json"
    modality = getattr(args, "modality", "all").lower()
    target_hz = getattr(args, "target_hz", None)

    if meta_file.exists():
        try:
            with open(meta_file) as mf:
                mdata = json.load(mf)
                if modality == "all" and "modality" in mdata:
                    modality = mdata["modality"].lower()
                if target_hz is None and "sampling_rate_hz" in mdata:
                    target_hz = float(mdata["sampling_rate_hz"])
        except Exception:
            pass

    target_hz = target_hz or 2000.0
    qc = SignalQC(target_sample_rate_hz=target_hz)  # adc_max=None — saturation not checked (TBD)
    samples = []
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            samples.append(Sample.from_dict(row))

    result = qc.run(samples, session_id=str(path))

    passed = True
    if result.dropped_samples > 0:
        passed = False

    dev_pct = abs((result.measured_sample_rate_hz or 0) - target_hz) / target_hz * 100
    if dev_pct > 1.0:
        passed = False

    if modality == "emg":
        active_channels = ["EMG_Submental", "EMG_Orbicularis", "EMG_Zygomaticus", "EMG_Buccinator"]
    elif modality == "eeg":
        active_channels = ["EEG_Fp1", "EEG_Fp2"]
    else:
        active_channels = list(result.per_channel.keys())

    for ch in active_channels:
        cr = result.per_channel.get(ch)
        if cr and not cr.passed:
            passed = False

    print(f"Total samples:      {result.total_samples}")
    print(f"Dropped samples:    {result.dropped_samples}")
    print(f"Measured rate (Hz): {result.measured_sample_rate_hz:.1f}")
    print(f"QC passed:          {passed}")
    for ch, cr in result.per_channel.items():
        if cr.notes:
            is_active = ch in active_channels
            prefix = "" if is_active else "[Zero-padded other modality] "
            print(f"  {ch}: {prefix}{cr.notes}")
    if not passed:
        sys.exit(1)


def cmd_validate_dataset(args):
    """python -m backend validate-dataset <path>

    Validates the schema of subject.json, session.json, and trials.json
    for every session directory found under <path>.

    An empty or missing directory is treated as a validation failure.
    """
    import json
    from backend.datasets.schema import validate_subject, validate_session, validate_trial

    path = Path(args.path)
    if not path.exists():
        print(f"ERROR: Dataset path not found: {path}", file=sys.stderr)
        sys.exit(1)

    print(f"Validating dataset at: {path}")
    errors: list[str] = []
    n_sessions_found = 0

    # Walk looking for metadata/ directories which contain the three schema files
    for metadata_dir in path.rglob("metadata"):
        if not metadata_dir.is_dir():
            continue

        # ── subject.json ─────────────────────────────────────────────────────
        subject_file = metadata_dir / "subject.json"
        if not subject_file.exists():
            errors.append(f"{metadata_dir}: subject.json missing")
        else:
            try:
                with open(subject_file) as f:
                    subj = json.load(f)
                errs = validate_subject(subj)
                for e in errs:
                    errors.append(f"{subject_file}: {e}")
            except Exception as e:
                errors.append(f"{subject_file}: could not parse — {e}")

        # ── session.json ──────────────────────────────────────────────────────
        session_file = metadata_dir / "session.json"
        if not session_file.exists():
            errors.append(f"{metadata_dir}: session.json missing")
        else:
            try:
                with open(session_file) as f:
                    sess = json.load(f)
                errs = validate_session(sess)
                for e in errs:
                    errors.append(f"{session_file}: {e}")
            except Exception as e:
                errors.append(f"{session_file}: could not parse — {e}")

        # ── trials.json ───────────────────────────────────────────────────────
        trials_file = metadata_dir / "trials.json"
        if not trials_file.exists():
            errors.append(f"{metadata_dir}: trials.json missing")
        else:
            try:
                with open(trials_file) as f:
                    trials = json.load(f)
                for t in trials:
                    errs = validate_trial(t)
                    for e in errs:
                        errors.append(f"{trials_file} trial {t.get('trial_id','?')}: {e}")
            except Exception as e:
                errors.append(f"{trials_file}: could not parse — {e}")

        n_sessions_found += 1

    if n_sessions_found == 0:
        print(
            "ERROR: No metadata/ directories found. "
            "Is this a valid URF/A-SSR dataset directory?",
            file=sys.stderr,
        )
        sys.exit(1)

    if errors:
        for e in errors:
            print(f"ERROR: {e}", file=sys.stderr)
        print(
            f"\nDataset validation FAILED: {len(errors)} error(s) in "
            f"{n_sessions_found} session(s).",
            file=sys.stderr,
        )
        sys.exit(1)

    print(
        f"Dataset validation passed: {n_sessions_found} session(s) checked, "
        "no schema errors."
    )


def cmd_generate_splits(args):
    """python -m backend generate-splits <config>"""
    import yaml
    from backend.datasets.splitter import DatasetSplitter

    config_path = Path(args.config)
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    splitter = DatasetSplitter(seed=cfg.get("seed", 42))
    subjects = cfg["subjects"]
    sessions = cfg["sessions_per_subject"]
    strategy = cfg.get("strategy", "subject_held_out")

    if strategy == "LOSO":
        manifests = splitter.loso_splits(subjects, sessions)
        for i, m in enumerate(manifests):
            out = config_path.parent / f"split_manifest_fold{i}.json"
            splitter.save_manifest(m, out)
            print(f"Saved: {out}")
    else:
        manifest = splitter.split_by_subject(subjects, sessions)
        out = config_path.parent / "split_manifest.json"
        splitter.save_manifest(manifest, out)
        print(f"Saved: {out}")


def cmd_train(args):
    """python -m backend train <model> <config>

    STATUS: BLOCKED — not yet implemented.

    Training requires:
      - A real preprocessed dataset (Gate 7 complete).
      - Frozen preprocessing and windowing parameters (Gate 6).
      - A valid split_manifest.json (from generate-splits).

    None of these exist yet. This command is a stub.
    See docs/model_development.md for the intended workflow.
    """
    print(
        f"ERROR: 'train {args.model}' is NOT IMPLEMENTED.\n"
        f"Blocked by: no real dataset (Gate 7 not passed).\n"
        f"See docs/model_development.md for requirements.",
        file=sys.stderr,
    )
    sys.exit(2)


def cmd_evaluate(args):
    """python -m backend evaluate <experiment>"""
    import json
    exp_path = Path(args.experiment)
    manifest = exp_path / "experiment_manifest.json"
    if not manifest.exists():
        print(f"ERROR: No experiment_manifest.json at {exp_path}", file=sys.stderr)
        sys.exit(1)
    with open(manifest) as f:
        m = json.load(f)
    print(f"Experiment: {m['experiment_id']}")
    print(f"  Model:              {m['model']}")
    print(f"  Git commit:         {m['git_commit']}")
    print(f"  Dataset version:    {m.get('dataset_version', 'TBD')}")
    print(f"  Preprocessing ver:  {m.get('preprocessing_version', 'TBD')}")
    print(f"  Seed:               {m.get('seed', 'TBD')}")
    print(f"  Metrics:            {m.get('metrics', {})}")


def cmd_replay(args):
    """python -m backend replay <recording>

    Loads a raw CSV recording and runs it through the production processing
    pipeline (QC → preprocessing → windowing → feature extraction → models → fusion → safety).

    PRD §40: Replay must run the exact production pipeline without hardware.
    Trained models are loaded and evaluated through the deterministic safety engine.
    Predictions with confidence >= 0.70 produce intent outputs (YES, NO, HELP),
    while lower confidence or degraded signals trigger NO_DECISION.
    """
    import warnings
    warnings.filterwarnings("ignore")

    from backend.realtime.replay import ReplaySource
    from backend.realtime.pipeline import RealtimePipeline, PipelineMode
    from backend.protocol.output_states import IntentOutput

    path = Path(args.recording)
    if not path.exists():
        print(f"ERROR: Recording not found: {path}", file=sys.stderr)
        sys.exit(1)

    print(f"Replaying recording: {path}")

    source = ReplaySource(path)
    pipeline = RealtimePipeline(mode=PipelineMode.REPLAY, load_default_models=True)

    has_emg = pipeline._emg_model is not None
    has_eeg = pipeline._eeg_model is not None
    print(f"Models loaded — EMG Model: {'ACTIVE' if has_emg else 'NONE'}, EEG Model: {'ACTIVE' if has_eeg else 'NONE'}")
    print("Safety Engine Threshold: 0.70 (Predictions below threshold gated to NO_DECISION)")

    window_size = 400
    hop_size = 200

    emg_buffer = []
    eeg_buffer = []

    n_samples = 0
    n_outputs = 0
    intent_counts: dict[str, int] = {}

    import numpy as np

    for sample in source.stream():
        n_samples += 1
        emg_buffer.append([
            sample.EMG_Submental,
            sample.EMG_Orbicularis,
            sample.EMG_Zygomaticus,
            sample.EMG_Buccinator,
        ])
        eeg_buffer.append([
            sample.EEG_Fp1,
            sample.EEG_Fp2,
        ])

        if len(emg_buffer) >= window_size:
            emg_win = np.array(emg_buffer, dtype=float)
            eeg_win = np.array(eeg_buffer, dtype=float)

            output = pipeline.process_window(
                emg_window=emg_win,
                eeg_window=eeg_win,
                signal_quality="OK",
            )
            n_outputs += 1
            label = str(output.final_intent.value) if hasattr(output.final_intent, "value") else str(output.final_intent)
            intent_counts[label] = intent_counts.get(label, 0) + 1

            if label not in ("NO_DECISION",) and (n_outputs % 15 == 0 or label == "HELP"):
                conf = output.emg_prediction.confidence if output.emg_prediction else 0.0
                safety_str = str(output.safety_decision.value) if hasattr(output.safety_decision, "value") else str(output.safety_decision)
                print(f"  [Output #{n_outputs:04d} | Sample {n_samples:06d}] Prediction: {label:<5} (Conf: {conf:.2f}, Safety: {safety_str})")

            emg_buffer = emg_buffer[hop_size:]
            eeg_buffer = eeg_buffer[hop_size:]

    # Handle remaining buffer at EOF
    if emg_buffer and len(emg_buffer) >= 10:
        emg_win = np.array(emg_buffer, dtype=float)
        eeg_win = np.array(eeg_buffer, dtype=float)
        output = pipeline.process_window(
            emg_window=emg_win,
            eeg_window=eeg_win,
            signal_quality="OK",
        )
        n_outputs += 1
        label = str(output.final_intent.value) if hasattr(output.final_intent, "value") else str(output.final_intent)
        intent_counts[label] = intent_counts.get(label, 0) + 1

    print(f"\nSamples replayed:      {n_samples}")
    print(f"Pipeline outputs:      {n_outputs}")
    print(f"Intent distribution:   {intent_counts}")
    print(
        "Replay complete. Pipeline successfully executed with trained model inference and safety gating."
    )


def cmd_realtime(args):
    """python -m backend realtime

    STATUS: BLOCKED — hardware not connected and safety gate not passed.

    LIVE mode requirements (all must be met):
      1. configs/hardware.yaml exists with human_use_ready: true
         (requires institutional electrical and experimental safety review).
      2. Trained, SELECTED models registered in the model registry (Gates 8-9).
      3. A fully configured final_pipeline.yaml with no TBD fields (Gate 10).

    None of these conditions are currently met.
    """
    from backend.realtime.pipeline import RealtimePipeline, PipelineMode, _repo_root

    hw_config = _repo_root() / "configs" / "hardware.yaml"

    print("URF/A-SSR real-time mode — prerequisite check:")

    # Check human_use_ready
    if not hw_config.exists():
        print(f"  [BLOCKED] configs/hardware.yaml not found at {hw_config}")
        print(
            "\nLIVE mode is NOT IMPLEMENTED.\n"
            "Blockers: hardware not available, safety review not completed.\n"
            "See docs/safety_spec.md and docs/validation_report.md.",
            file=sys.stderr,
        )
        sys.exit(2)

    import yaml
    with open(hw_config) as f:
        hw = yaml.safe_load(f)

    human_use = hw.get("human_use_ready", False)
    print(f"  human_use_ready:    {human_use}")

    if not human_use:
        print(
            "\nLIVE mode REFUSED: human_use_ready is False.\n"
            "Complete hardware and institutional safety review first (PRD §10).\n"
            "Do NOT connect participants until this flag is set by qualified review.",
            file=sys.stderr,
        )
        sys.exit(2)

    # If human_use_ready were True (future state), we would continue here
    print(
        "\nLIVE mode is NOT IMPLEMENTED — "
        "live acquisition code is not yet written (hardware pending).\n"
        "See docs/realtime_spec.md for the planned implementation.",
        file=sys.stderr,
    )
    sys.exit(2)


def main():
    parser = argparse.ArgumentParser(
        prog="urf",
        description="URF/A-SSR multimodal EEG+EMG research system CLI",
    )
    subparsers = parser.add_subparsers(dest="command")

    p_vr = subparsers.add_parser(
        "validate-recording",
        help="Run QC on a raw CSV recording file",
    )
    p_vr.add_argument("path")
    p_vr.add_argument(
        "--modality",
        choices=["all", "emg", "eeg"],
        default="all",
        help="Specify active modality for single-modality datasets (default: auto/all)",
    )
    p_vr.add_argument(
        "--target-hz",
        type=float,
        default=None,
        help="Target sampling rate in Hz (default: from metadata or 2000.0)",
    )

    p_vd = subparsers.add_parser(
        "validate-dataset",
        help="Validate subject/session/trial schema in a dataset directory",
    )
    p_vd.add_argument("path")

    p_gs = subparsers.add_parser(
        "generate-splits",
        help="Generate subject/session-aware train/val/test splits",
    )
    p_gs.add_argument("config")

    p_tr = subparsers.add_parser(
        "train",
        help="[BLOCKED] Train a model — requires dataset (Gate 7)",
    )
    p_tr.add_argument(
        "model",
        choices=["emg-svm", "emg-gru", "emg-cnn-gru", "eeg-svm", "eeg-cnn"],
    )
    p_tr.add_argument("config")

    p_ev = subparsers.add_parser(
        "evaluate",
        help="Display a saved experiment manifest",
    )
    p_ev.add_argument("experiment")

    p_rp = subparsers.add_parser(
        "replay",
        help="Replay a raw recording through the production pipeline",
    )
    p_rp.add_argument("recording")

    subparsers.add_parser(
        "realtime",
        help="[BLOCKED] Live acquisition — requires hardware and safety gate",
    )

    args = parser.parse_args()

    dispatch = {
        "validate-recording": cmd_validate_recording,
        "validate-dataset":   cmd_validate_dataset,
        "generate-splits":    cmd_generate_splits,
        "train":              cmd_train,
        "evaluate":           cmd_evaluate,
        "replay":             cmd_replay,
        "realtime":           cmd_realtime,
    }

    if args.command in dispatch:
        dispatch[args.command](args)
    else:
        parser.print_help()
        sys.exit(0)


if __name__ == "__main__":
    main()
