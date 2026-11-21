"""
URF/A-SSR Real-Time Pipeline
PRD §39, §40, §47, §54 — Real-time pipeline with mock/replay/live modes.
Final mode refuses to start if final_pipeline.yaml contains TBD fields.

SAFETY GATE POLICY (PRD §10, §47):
  - hardware.yaml is resolved relative to the repository root, NOT the CWD.
  - Missing hardware.yaml → LIVE mode refused (fail-closed).
  - human_use_ready=false → LIVE mode refused.
  - Missing final_pipeline.yaml → LIVE mode refused.
  - Any TBD field in final_pipeline.yaml → LIVE mode refused.
"""
from __future__ import annotations

import os
import warnings
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Optional

import yaml

from backend.fusion.decision_fusion import DecisionFusion, FusionResult
from backend.fusion.confidence import ConfidenceGate, ConfidenceConfig
from backend.fusion.temporal_consistency import (
    TemporalConsistencyFilter,
    TemporalConsistencyConfig,
)
from backend.models.base import Prediction
from backend.protocol.output_states import IntentOutput
from backend.realtime.latency_monitor import LatencyMonitor
from backend.safety.safety_engine import SafetyEngine, SafetyConfig


def _repo_root() -> Path:
    """
    Resolve the repository root as the directory containing this package.
    backend/ is one level below the repo root, so go up two levels from this file.
    """
    return Path(__file__).resolve().parent.parent.parent


class PipelineMode(str, Enum):
    MOCK   = "MOCK"
    REPLAY = "REPLAY"
    LIVE   = "LIVE"


@dataclass
class PipelineOutput:
    """One complete pipeline output for a processed window."""
    mode: PipelineMode
    emg_prediction: Optional[Prediction]
    eeg_prediction: Optional[Prediction]
    fusion_result: FusionResult
    safety_decision: str              # "ACTION" | "NO_DECISION"
    safety_rule: str
    final_intent: str                 # YES | NO | HELP | NO_DECISION
    is_mock: bool = False
    notes: str = ""


class RealtimePipeline:
    """
    Orchestrates the real-time inference pipeline.

    PRD §39 architecture:
    PCB → ESP32-S3 → stream → Python receiver → buffer → QC →
    preprocessing → windowing → models → probabilities →
    fusion → temporal consistency → safety engine → output

    Modes: MOCK | REPLAY | LIVE
    """

    def __init__(
        self,
        mode: PipelineMode,
        emg_model=None,
        eeg_model=None,
        confidence_config: Optional[ConfidenceConfig] = None,
        temporal_config: Optional[TemporalConsistencyConfig] = None,
        safety_config: Optional[SafetyConfig] = None,
        final_pipeline_config_path: Optional[Path] = None,
        models_dir: Optional[Path] = None,
        load_default_models: bool = True,
    ) -> None:
        self.mode = mode
        self._emg_model = emg_model
        self._eeg_model = eeg_model
        self.latency_monitor = LatencyMonitor()

        # Load production models if available
        if mode != PipelineMode.MOCK and load_default_models:
            self._load_models_if_available(models_dir)

        # Initialize feature extractors for windowed conversion
        from backend.features.emg_features import EMGFeatureConfig, EMGFeatureExtractor
        from backend.features.eeg_features import EEGFeatureConfig, EEGFeatureExtractor

        self._emg_extractor = EMGFeatureExtractor(EMGFeatureConfig(
            feature_version="v1",
            sampling_rate_hz=2000.0,
            use_rms=True,
            use_mav=True,
            use_wl=True,
            use_zcr=True,
        ))
        self._eeg_extractor = EEGFeatureExtractor(EEGFeatureConfig(
            feature_version="v1",
            sampling_rate_hz=2000.0,
            use_theta_power=True,
            use_alpha_power=True,
            use_beta_power=True,
            use_relative_bands=True,
        ))

        # Default confidence gate threshold is 0.70 when models are loaded
        if confidence_config is None:
            if self._emg_model is not None:
                confidence_config = ConfidenceConfig(
                    emg_min_confidence=0.70,
                    eeg_min_confidence=0.70,
                    config_version="v1",
                )
            else:
                confidence_config = ConfidenceConfig()

        self._confidence_gate = ConfidenceGate(confidence_config)
        self._temporal = TemporalConsistencyFilter(
            temporal_config or TemporalConsistencyConfig()
        )
        self._safety = SafetyEngine(
            safety_config or SafetyConfig(enabled=False)
        )
        self._fusion = DecisionFusion()

        if mode == PipelineMode.LIVE:
            self._check_live_prerequisites(final_pipeline_config_path)

    def _load_models_if_available(self, models_dir: Optional[Path] = None) -> None:
        """Attempt to load trained production models from models/ directory."""
        if models_dir is None:
            models_dir = _repo_root() / "models"
        models_dir = Path(models_dir)

        # 1. EMG Model
        if self._emg_model is None:
            emg_dir = models_dir / "emg_svm"
            if emg_dir.exists() and (emg_dir / "model.pkl").exists():
                try:
                    from backend.models.emg.svm_model import EMGSVMModel
                    m = EMGSVMModel()
                    m.load(emg_dir)
                    self._emg_model = m
                except Exception as e:
                    warnings.warn(f"Failed to auto-load EMG model from {emg_dir}: {e}")
            elif (models_dir / "emg_svm.pkl").exists():
                try:
                    import pickle
                    from backend.models.emg.svm_model import EMGSVMModel
                    m = EMGSVMModel()
                    with open(models_dir / "emg_svm.pkl", "rb") as f:
                        m._pipeline = pickle.load(f)
                    m._trained = True
                    self._emg_model = m
                except Exception as e:
                    warnings.warn(f"Failed to auto-load EMG model from {models_dir / 'emg_svm.pkl'}: {e}")

        # 2. EEG Model
        if self._eeg_model is None:
            eeg_dir = models_dir / "eeg_svm"
            if eeg_dir.exists() and (eeg_dir / "model.pkl").exists():
                try:
                    from backend.models.eeg.svm_model import EEGSVMModel
                    m = EEGSVMModel(label_map={"Neutral": 0, "Stressed": 1})
                    m.load(eeg_dir)
                    self._eeg_model = m
                except Exception as e:
                    warnings.warn(f"Failed to auto-load EEG model from {eeg_dir}: {e}")
            elif (models_dir / "eeg_svm.pkl").exists():
                try:
                    import pickle
                    from backend.models.eeg.svm_model import EEGSVMModel
                    m = EEGSVMModel(label_map={"Neutral": 0, "Stressed": 1})
                    with open(models_dir / "eeg_svm.pkl", "rb") as f:
                        m._pipeline = pickle.load(f)
                    m._trained = True
                    self._eeg_model = m
                except Exception as e:
                    warnings.warn(f"Failed to auto-load EEG model from {models_dir / 'eeg_svm.pkl'}: {e}")

    def _check_live_prerequisites(
        self, config_path: Optional[Path]
    ) -> None:
        """
        Refuse to start LIVE mode if any prerequisite is not met.

        Checks (all fail-closed — missing file = refused, not silently skipped):
          1. hardware.yaml exists at repo root and human_use_ready is True.
          2. final_pipeline.yaml is provided, exists, and contains no TBD fields.

        hardware.yaml is resolved from the repository root, NOT the current
        working directory. This prevents silent bypass when the process is
        started from an arbitrary directory (audit finding C-1).
        """
        # ── 1. hardware.yaml — must exist and human_use_ready must be True ──
        hw_config = _repo_root() / "configs" / "hardware.yaml"
        if not hw_config.exists():
            raise RuntimeError(
                f"LIVE mode refused: configs/hardware.yaml not found at "
                f"{hw_config}. Cannot verify human_use_ready (PRD §10)."
            )
        with open(hw_config) as f:
            hw = yaml.safe_load(f)
        if not hw.get("human_use_ready", False):
            raise RuntimeError(
                "LIVE mode refused: human_use_ready is False in "
                "configs/hardware.yaml. Complete electrical and experimental "
                "safety review before live use (PRD §10)."
            )

        # ── 2. final_pipeline.yaml — must be provided, exist, and have no TBD ──
        if config_path is None:
            raise RuntimeError(
                "LIVE mode refused: final_pipeline_config_path was not provided. "
                "Pass the path to a fully configured final_pipeline.yaml (PRD §47)."
            )
        fp_config = Path(config_path).resolve()
        if not fp_config.exists():
            raise RuntimeError(
                f"LIVE mode refused: final_pipeline.yaml not found at {fp_config} "
                f"(PRD §47)."
            )
        with open(fp_config) as f:
            cfg = yaml.safe_load(f)
        tbd_fields = [k for k, v in _flatten(cfg).items() if v == "TBD"]
        if tbd_fields:
            raise RuntimeError(
                f"LIVE mode refused: final_pipeline.yaml has {len(tbd_fields)} "
                f"TBD field(s): {tbd_fields}. Resolve all before live use (PRD §47)."
            )

    def process_window(
        self,
        emg_window: Optional[object] = None,
        eeg_window: Optional[object] = None,
        signal_quality: str = "OK",
    ) -> PipelineOutput:
        """
        Process one window of preprocessed data through the full pipeline.

        Args:
            emg_window: Preprocessed EMG window (features or sequence).
            eeg_window: Preprocessed EEG window.
            signal_quality: "OK" | "DEGRADED" | "INVALID"

        Returns:
            PipelineOutput
        """
        if self.mode == PipelineMode.MOCK:
            return self._mock_output()

        import numpy as np

        # EMG inference
        emg_prediction = None
        if emg_window is not None and self._emg_model is not None:
            with self.latency_monitor.measure("emg_inference"):
                if isinstance(emg_window, np.ndarray):
                    try:
                        if hasattr(self._emg_model, "_pipeline"):
                            # Feature-based SVM model (expects 16 features)
                            if emg_window.ndim == 2 and emg_window.shape[1] == 4 and emg_window.shape[0] >= 10:
                                feat = self._emg_extractor.extract_window(emg_window)
                                preds = self._emg_model.predict(feat[np.newaxis, :])
                                emg_prediction = preds[0] if preds else None
                            elif (emg_window.ndim == 2 and emg_window.shape[1] == 16) or (emg_window.ndim == 1 and emg_window.shape[0] == 16):
                                X = emg_window[np.newaxis, :] if emg_window.ndim == 1 else emg_window
                                preds = self._emg_model.predict(X)
                                emg_prediction = preds[0] if preds else None
                            elif emg_window.ndim == 2 and emg_window.shape[1] != 16:
                                # Not enough samples or shape mismatch
                                emg_prediction = None
                            else:
                                X = emg_window[np.newaxis] if emg_window.ndim < 2 else emg_window
                                preds = self._emg_model.predict(X)
                                emg_prediction = preds[0] if preds else None
                        else:
                            # Sequence-based model (GRU / CNN-GRU)
                            if emg_window.ndim == 2 and emg_window.shape[1] == 4:
                                X = emg_window[np.newaxis, :, :]
                            else:
                                X = emg_window[np.newaxis] if emg_window.ndim < 2 else emg_window
                            preds = self._emg_model.predict(X)
                            emg_prediction = preds[0] if preds else None
                    except Exception as e:
                        emg_prediction = None

        # EEG inference
        eeg_prediction = None
        if eeg_window is not None and self._eeg_model is not None:
            with self.latency_monitor.measure("eeg_inference"):
                if isinstance(eeg_window, np.ndarray):
                    try:
                        if hasattr(self._eeg_model, "_pipeline"):
                            # Feature-based EEG SVM (expects band power features)
                            if eeg_window.ndim == 2 and eeg_window.shape[1] == 2 and eeg_window.shape[0] >= 10:
                                feat = self._eeg_extractor.extract_window(eeg_window)
                                preds = self._eeg_model.predict(feat[np.newaxis, :])
                                eeg_prediction = preds[0] if preds else None
                            elif (eeg_window.ndim == 2 and eeg_window.shape[1] == 12) or (eeg_window.ndim == 1 and eeg_window.shape[0] == 12):
                                X = eeg_window[np.newaxis, :] if eeg_window.ndim == 1 else eeg_window
                                preds = self._eeg_model.predict(X)
                                eeg_prediction = preds[0] if preds else None
                            elif eeg_window.ndim == 2 and eeg_window.shape[1] != 12:
                                eeg_prediction = None
                            else:
                                X = eeg_window[np.newaxis] if eeg_window.ndim < 2 else eeg_window
                                preds = self._eeg_model.predict(X)
                                eeg_prediction = preds[0] if preds else None
                        else:
                            if eeg_window.ndim == 2 and eeg_window.shape[1] == 2:
                                X = eeg_window[np.newaxis, :, :]
                            else:
                                X = eeg_window[np.newaxis] if eeg_window.ndim < 2 else eeg_window
                            preds = self._eeg_model.predict(X)
                            eeg_prediction = preds[0] if preds else None
                    except Exception as e:
                        eeg_prediction = None

        # Confidence gating
        emg_quality_ok = signal_quality == "OK"
        eeg_quality_ok = signal_quality == "OK"

        if emg_prediction:
            emg_prediction = self._confidence_gate.check_emg(emg_prediction, emg_quality_ok)
        if eeg_prediction:
            eeg_prediction = self._confidence_gate.check_eeg(eeg_prediction, eeg_quality_ok)

        # Temporal consistency
        # Capture the actual result from the temporal filter.
        # Do NOT infer temporal_consistency_met from the fusion output —
        # that would bypass the temporal check entirely (audit finding C-4).
        temporal_consistency_met = False
        if emg_prediction:
            updated = self._temporal.update(emg_prediction)
            # The temporal filter returns NO_DECISION if consistency is not yet met.
            temporal_consistency_met = (updated.label != IntentOutput.NO_DECISION)
            emg_prediction = updated

        # Decision-level fusion
        with self.latency_monitor.measure("fusion"):
            fusion_result = self._fusion.fuse(emg_prediction, eeg_prediction)

        # Safety engine — pass the actual temporal filter result, not the fusion result
        state = eeg_prediction.label if eeg_prediction else None
        state_confidence = eeg_prediction.confidence if eeg_prediction else 0.0
        intent_confidence = emg_prediction.confidence if emg_prediction else 0.0

        safety_decision = self._safety.evaluate(
            intent=fusion_result.intent,
            state=state,
            intent_confidence=intent_confidence,
            state_confidence=state_confidence,
            signal_quality=signal_quality,
            temporal_consistency_met=temporal_consistency_met,
        )

        # Final intent
        final_intent = fusion_result.intent
        if self._safety.config.enabled:
            if fusion_result.intent == self._safety.config.trigger_intent:
                if safety_decision.output != "ACTION":
                    final_intent = IntentOutput.NO_DECISION

        return PipelineOutput(
            mode=self.mode,
            emg_prediction=emg_prediction,
            eeg_prediction=eeg_prediction,
            fusion_result=fusion_result,
            safety_decision=safety_decision.output,
            safety_rule=safety_decision.rule_applied,
            final_intent=final_intent,
            is_mock=False,
        )

    def _mock_output(self) -> PipelineOutput:
        """Generate a mock output for testing. All outputs labelled MOCK_DATA."""
        from backend.realtime.mock_source import MOCK_DATA_LABEL
        mock_fusion = FusionResult(
            intent=IntentOutput.NO_DECISION,
            reason=f"MOCK_DATA — {MOCK_DATA_LABEL}",
        )
        return PipelineOutput(
            mode=PipelineMode.MOCK,
            emg_prediction=None,
            eeg_prediction=None,
            fusion_result=mock_fusion,
            safety_decision=IntentOutput.NO_DECISION,
            safety_rule="mock_mode",
            final_intent=IntentOutput.NO_DECISION,
            is_mock=True,
            notes=f"MOCK_DATA — not for research results",
        )


def _flatten(d: dict, prefix: str = "") -> dict:
    """Flatten nested dict for TBD checking."""
    result = {}
    for k, v in d.items():
        key = f"{prefix}.{k}" if prefix else k
        if isinstance(v, dict):
            result.update(_flatten(v, key))
        else:
            result[key] = v
    return result
