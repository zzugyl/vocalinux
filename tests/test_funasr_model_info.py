"""FunASR catalog: precision-driven loading, license pins, and data-driven wiring.

The FunASR catalog is the only place a model or bundle is described, so these
tests pin the three properties that keep it that way: the loader reads the
weight precision instead of assuming int8, every bundle states its license,
and every downstream helper works off the catalog rather than a hardcoded
second list -- adding a model is an entry, not an edit across the codebase.
"""

import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from vocalinux.speech_recognition.engines import funasr_engine
from vocalinux.speech_recognition.engines.funasr_engine import FunasrEngine
from vocalinux.utils import funasr_model_info as models


def test_int8_is_the_default_precision_and_drives_quantize() -> None:
    """Both published bundles are int8, so quantize stays True without a key."""
    for bundle in models.FUNASR_BUNDLE_INFO:
        assert models.precision_for(bundle) == "int8"
        assert models.quantize_for(bundle) is True


def test_a_precision_the_loader_cannot_load_is_refused(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An unknown precision fails loudly instead of opening the wrong file."""
    with pytest.raises(ValueError, match="Unknown FunASR bundle"):
        models.quantize_for("no-such-bundle")
    with pytest.raises(ValueError, match="Unknown FunASR bundle"):
        models.precision_for("no-such-bundle")

    monkeypatch.setitem(models.FUNASR_BUNDLE_INFO["asr"], "precision", "int4")
    with pytest.raises(ValueError, match="Unknown FunASR precision 'int4'"):
        models.quantize_for("asr")


def test_precision_file_names_agree_with_the_quantize_flag(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Whatever precision a bundle declares, its ONNX files must be that build.

    quantize=True opens ``model_quant.onnx`` (+ ``model_eb_quant.onnx``) and
    quantize=False opens ``model.onnx`` (+ ``model_eb.onnx``), so a catalog
    entry that mixes the two would download one set and load another.
    """
    for bundle in models.FUNASR_BUNDLE_INFO:
        quantized = {name for name in models.model_files(bundle) if name.endswith(".onnx")}
        assert quantized, f"{bundle} ships no ONNX file"
        if models.quantize_for(bundle):
            assert all(name.endswith("_quant.onnx") for name in quantized), bundle
        else:
            assert not any(name.endswith("_quant.onnx") for name in quantized), bundle

    monkeypatch.setitem(models.FUNASR_BUNDLE_INFO["asr"], "precision", "fp32")
    assert models.quantize_for("asr") is False
    assert models.precision_for("asr") == "fp32"


def test_every_bundle_states_a_license() -> None:
    """ModelScope ships no LICENSE file, so the catalog is where it is recorded."""
    for bundle, info in models.FUNASR_BUNDLE_INFO.items():
        assert info.get("license") == "Apache-2.0", bundle


def test_the_loader_opens_the_files_the_catalog_declares(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """init() passes the catalog's quantize flag and bundle dirs to funasr-onnx."""
    fake_funasr_onnx = MagicMock()
    monkeypatch.setitem(sys.modules, "funasr_onnx", fake_funasr_onnx)
    monkeypatch.setattr(funasr_engine, "get_model_path", lambda bundle: str(tmp_path / bundle))

    FunasrEngine().init()

    asr = fake_funasr_onnx.ContextualParaformer.call_args
    punc = fake_funasr_onnx.CT_Transformer.call_args
    assert asr.args[0] == str(tmp_path / "asr")
    assert asr.kwargs["quantize"] is models.quantize_for("asr")
    assert punc.args[0] == str(tmp_path / "punc")
    assert punc.kwargs["quantize"] is models.quantize_for("punc")


def test_a_second_catalog_entry_needs_no_new_code(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Every catalog-driven helper accepts an entry added after the fact.

    This is the "add a model, touch one file" property: paths, download
    checks, listings, deletion, pins and URLs all read FUNASR_MODEL_INFO /
    FUNASR_BUNDLE_INFO, so a new model must not trip over a hardcoded list.
    """
    monkeypatch.setitem(
        models.FUNASR_MODEL_INFO,
        "paraformer-fake",
        {
            "desc": "Synthetic second model",
            "size_mb": 1,
            "params": "1M",
            "bundles": ("asr", "punc"),
        },
    )
    monkeypatch.setattr(models, "get_model_path", lambda bundle: str(tmp_path / "funasr" / bundle))
    monkeypatch.setattr(models, "FUNASR_MODEL_SIZES", list(models.FUNASR_MODEL_INFO))

    assert models.bundles_for("paraformer-fake") == ("asr", "punc")
    assert models.is_model_downloaded("paraformer-fake") is False
    for bundle in models.bundles_for("paraformer-fake"):
        Path(models.get_model_path(bundle)).mkdir(parents=True)
    assert "paraformer-fake" in models.list_downloaded_models()
    for bundle in models.bundles_for("paraformer-fake"):
        for filename in models.model_files(bundle):
            assert models.manifest_key(bundle, filename).startswith(f"funasr-{bundle}-")
            assert models.get_model_file_url(bundle, filename).startswith("https://modelscope.cn/")
    with pytest.raises(ValueError, match="Unknown FunASR model"):
        models.bundles_for("not-in-the-catalog")


def test_the_settings_dropdown_reads_the_catalog() -> None:
    """The engine's model list is derived from the catalog, not restated."""
    from vocalinux.ui.settings_dialog import ENGINE_MODELS

    assert ENGINE_MODELS["funasr"] == list(models.FUNASR_MODEL_SIZES)
    assert models.RECOMMENDED_MODEL in models.FUNASR_MODEL_SIZES
    assert models.FUNASR_MODEL_SIZES == list(models.FUNASR_MODEL_INFO)
