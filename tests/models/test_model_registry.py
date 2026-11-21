"""
Tests for ModelRegistry — status management, version compatibility, final-mode gate.
PRD §46, §49.
"""
import pytest
from pathlib import Path

from backend.models.model_registry import (
    ModelRegistry, ModelEntry, ModelStatus, ModelRegistryError
)


def make_entry(model_id="emg-svm-v1", status=ModelStatus.EXPERIMENTAL) -> ModelEntry:
    return ModelEntry(
        model_id=model_id,
        modality="EMG",
        task="intent_classification",
        architecture="SVM",
        dataset_version="ds-1",
        preprocessing_version="pp-1",
        experiment_id="exp-001",
        status=status,
        validation_metrics={"macro_f1": 0.82},
        artifact_path="/tmp/model",
    )


class TestModelRegistry:

    def test_register_and_get(self, tmp_path):
        reg = ModelRegistry(tmp_path / "registry.json")
        entry = make_entry()
        reg.register(entry)
        retrieved = reg.get("emg-svm-v1")
        assert retrieved.model_id == "emg-svm-v1"

    def test_duplicate_registration_raises(self, tmp_path):
        reg = ModelRegistry(tmp_path / "registry.json")
        reg.register(make_entry())
        with pytest.raises(ModelRegistryError, match="already registered"):
            reg.register(make_entry())

    def test_update_status(self, tmp_path):
        reg = ModelRegistry(tmp_path / "registry.json")
        reg.register(make_entry())
        reg.update_status("emg-svm-v1", ModelStatus.VALIDATED)
        assert reg.get("emg-svm-v1").status == ModelStatus.VALIDATED

    def test_only_selected_can_load_in_final_mode(self, tmp_path):
        reg = ModelRegistry(tmp_path / "registry.json")
        reg.register(make_entry("m1", ModelStatus.EXPERIMENTAL))
        reg.register(make_entry("m2", ModelStatus.VALIDATED))
        reg.register(make_entry("m3", ModelStatus.SELECTED))
        reg.register(make_entry("m4", ModelStatus.RETIRED))

        assert reg.can_load_in_final_mode("m1") is False
        assert reg.can_load_in_final_mode("m2") is False
        assert reg.can_load_in_final_mode("m3") is True
        assert reg.can_load_in_final_mode("m4") is False

    def test_get_selected_returns_selected_model(self, tmp_path):
        reg = ModelRegistry(tmp_path / "registry.json")
        reg.register(make_entry("m1", ModelStatus.VALIDATED))
        reg.register(make_entry("m2", ModelStatus.SELECTED))
        selected = reg.get_selected("EMG")
        assert selected.model_id == "m2"

    def test_get_selected_returns_none_if_none(self, tmp_path):
        reg = ModelRegistry(tmp_path / "registry.json")
        reg.register(make_entry("m1", ModelStatus.EXPERIMENTAL))
        assert reg.get_selected("EMG") is None

    def test_multiple_selected_raises(self, tmp_path):
        reg = ModelRegistry(tmp_path / "registry.json")
        reg.register(make_entry("m1", ModelStatus.SELECTED))
        reg.register(make_entry("m2", ModelStatus.SELECTED))
        with pytest.raises(ModelRegistryError, match="Multiple SELECTED"):
            reg.get_selected("EMG")

    def test_version_compatibility_match(self, tmp_path):
        reg = ModelRegistry(tmp_path / "registry.json")
        reg.register(make_entry())
        issues = reg.check_version_compatibility("emg-svm-v1", "pp-1", "ds-1")
        assert issues == []

    def test_version_compatibility_mismatch(self, tmp_path):
        reg = ModelRegistry(tmp_path / "registry.json")
        reg.register(make_entry())
        issues = reg.check_version_compatibility("emg-svm-v1", "pp-2", "ds-1")
        assert any("reprocessing" in i.lower() for i in issues)

    def test_registry_persists_across_instances(self, tmp_path):
        path = tmp_path / "registry.json"
        reg1 = ModelRegistry(path)
        reg1.register(make_entry("persisted-model"))
        reg2 = ModelRegistry(path)
        assert reg2.get("persisted-model").model_id == "persisted-model"

    def test_unknown_model_raises(self, tmp_path):
        reg = ModelRegistry(tmp_path / "registry.json")
        with pytest.raises(ModelRegistryError, match="not found"):
            reg.get("nonexistent")
