"""FunASR model information for Vocalinux.

Metadata and checksum-gated download URLs for the optional FunASR backend:
the Paraformer-large contextual + CT-Transformer pairing VocoType's native
worker ships, run through the Python ``funasr-onnx`` runtime. Bundles are
stored under ``models_dir()/funasr/`` and each file is verified against
``model_checksums.txt`` before the engine loads it, the same way Parakeet,
faster-whisper and whisper.cpp models are.

Two bundles make up the one selectable model:

* ``asr``  - Paraformer-large contextual (vocab8404): a single model covers
  Chinese and English, so there is no per-language engine routing.
* ``punc`` - CT-Transformer punctuation (Chinese); the engine applies it only
  to utterances that contain CJK text, because it appends Chinese marks.

Every file is pinned to the ModelScope commit it is downloaded from, so a
file replaced upstream cannot change what an install receives. The digest
that guards the bytes lives in ``model_checksums.txt``; ModelScope publishes
it per file, which is how ``scripts/generate-model-checksums.py`` pins these
without downloading gigabytes.
"""

import logging
import os
import shutil
from typing import Any, Dict

from .paths import is_within_directory, models_dir

logger = logging.getLogger(__name__)

_MODELSCOPE_RESOLVE_URL = "https://modelscope.cn/models/{repo}/resolve/{revision}/{filename}"

# The one selectable FunASR model: VocoType's default pairing (no FunASR VAD,
# Vocalinux segments utterances with Silero VAD itself). ``params`` describes
# the ASR network; ``size_mb`` is the whole two-bundle download on disk. The
# id is "paraformer-large" rather than "large" so the generic model-label
# helpers read it as Paraformer Large instead of whisper's "Large v3".
FUNASR_MODEL_INFO: Dict[str, Dict[str, Any]] = {
    "paraformer-large": {
        "desc": "Paraformer-large contextual (zh/en) + punctuation",
        "size_mb": 1129,
        "params": "226M",
        "bundles": ("asr", "punc"),
    },
}

FUNASR_MODEL_SIZES = list(FUNASR_MODEL_INFO)

# There is a single model, so it is always the recommendation; the reason is
# what Settings and the tray show when offering it for download.
RECOMMENDED_MODEL = FUNASR_MODEL_SIZES[0]
RECOMMENDED_REASON = "Bilingual Chinese and English, one model"

# Bundle name -> ModelScope repository and the files the engine loads, each
# pinned to the commit it is fetched from. ``remote`` differs from the local
# name for one file only: funasr-onnx loads the contextual hotword encoder as
# ``model_eb_quant.onnx`` when ``quantize=True``, while ModelScope ships only
# ``model_eb.onnx``; the file is therefore stored under the name the loader
# asks for. ``configuration.json`` and ``seg_dict`` exist upstream but nothing
# in funasr-onnx reads them, so they are not downloaded.
FUNASR_BUNDLE_INFO: Dict[str, Dict[str, Any]] = {
    "asr": {
        "repo": (
            "iic/speech_paraformer-large-contextual_asr_nat-zh-cn-16k-common-" "vocab8404-onnx"
        ),
        "files": {
            "am.mvn": {"remote": "am.mvn", "revision": "1c4c4d2ee95dfc59aa7efa3bacce6b72af4622bc"},
            "config.yaml": {
                "remote": "config.yaml",
                "revision": "0e8b6a277734182ed17edb1302051dc40295622e",
            },
            "model_eb_quant.onnx": {
                "remote": "model_eb.onnx",
                "revision": "8f0881c891ceba7360e215b04e54cad564a68c41",
            },
            "model_quant.onnx": {
                "remote": "model_quant.onnx",
                "revision": "8f0881c891ceba7360e215b04e54cad564a68c41",
            },
            "tokens.json": {
                "remote": "tokens.json",
                "revision": "0e8b6a277734182ed17edb1302051dc40295622e",
            },
        },
    },
    "punc": {
        "repo": "iic/punc_ct-transformer_zh-cn-common-vocab272727-onnx",
        "files": {
            "config.yaml": {
                "remote": "config.yaml",
                "revision": "c9d8ba523014110650428e422a4975343193975c",
            },
            "model_quant.onnx": {
                "remote": "model_quant.onnx",
                "revision": "8f239ff78c6267c4d859233e7eb3bbdb68c61824",
            },
            "tokens.json": {
                "remote": "tokens.json",
                "revision": "c9d8ba523014110650428e422a4975343193975c",
            },
        },
    },
}


def bundles_for(model_size: str) -> tuple[str, ...]:
    """Return the bundle names that make up ``model_size``."""
    if model_size not in FUNASR_MODEL_INFO:
        raise ValueError(f"Unknown FunASR model: {model_size}")
    return tuple(FUNASR_MODEL_INFO[model_size]["bundles"])


def model_files(bundle: str) -> list[str]:
    """Return the local file names the engine loads from ``bundle``."""
    if bundle not in FUNASR_BUNDLE_INFO:
        raise ValueError(f"Unknown FunASR bundle: {bundle}")
    return list(FUNASR_BUNDLE_INFO[bundle]["files"])


def all_model_files() -> list[tuple[str, str]]:
    """Return ``(bundle, local_filename)`` for every file of every bundle."""
    return [(bundle, filename) for bundle in FUNASR_BUNDLE_INFO for filename in model_files(bundle)]


def get_model_path(bundle: str) -> str:
    """Return the directory a FunASR bundle is stored in."""
    if bundle not in FUNASR_BUNDLE_INFO:
        raise ValueError(f"Unknown FunASR bundle: {bundle}")
    funasr_dir = os.path.join(models_dir(), "funasr")
    os.makedirs(funasr_dir, exist_ok=True)
    return os.path.join(funasr_dir, bundle)


def manifest_key(bundle: str, filename: str) -> str:
    """Key this bundle file is pinned under in model_checksums.txt.

    Both bundles ship files called ``config.yaml``, ``model_quant.onnx`` and
    ``tokens.json``, so the bundle name is part of the key rather than the
    bare filename other engines use.
    """
    return f"funasr-{bundle}-{filename}"


def get_model_file_url(bundle: str, filename: str) -> str:
    """Return the ModelScope download URL for one file of a FunASR bundle.

    Args:
        bundle: Bundle name (``asr`` or ``punc``).
        filename: Local file name within the bundle (``model_eb_quant.onnx``
            resolves to its upstream name ``model_eb.onnx``).

    Returns:
        ModelScope download URL pinned to the file's commit.
    """
    info = FUNASR_BUNDLE_INFO.get(bundle)
    if not info:
        raise ValueError(f"Unknown FunASR bundle: {bundle}")
    spec = info["files"].get(filename)
    if spec is None:
        raise ValueError(f"Unknown FunASR file: {bundle}/{filename}")
    return _MODELSCOPE_RESOLVE_URL.format(
        repo=info["repo"], revision=spec["revision"], filename=spec["remote"]
    )


def is_model_downloaded(model_size: str) -> bool:
    """Return True when every bundle file of ``model_size`` exists on disk.

    Existence is only the cheap half: the manager hashes the files against
    their pins before loading them, the same way it treats faster-whisper
    bundles that were copied in rather than downloaded.
    """
    if model_size not in FUNASR_MODEL_INFO:
        return False
    for bundle in bundles_for(model_size):
        path = get_model_path(bundle)
        if not all(
            os.path.exists(os.path.join(path, filename)) for filename in model_files(bundle)
        ):
            return False
    return True


def list_downloaded_models() -> list[str]:
    """Return FunASR models occupying disk, complete or partially downloaded.

    Mirrors the Parakeet and faster-whisper helpers so Settings can offer
    deletion; a cancelled download leaves files behind that must be reclaimable.
    """
    return [
        name
        for name in FUNASR_MODEL_INFO
        if any(os.path.isdir(get_model_path(bundle)) for bundle in bundles_for(name))
    ]


def delete_model(model_size: str) -> str:
    """Delete a downloaded FunASR model's bundle directories.

    Returns:
        The FunASR models directory that held them.

    Raises:
        ValueError: Unknown model name, or path outside the models directory.
        FileNotFoundError: No bundle directory is present.
        OSError: A directory could not be removed.
    """
    if model_size not in FUNASR_MODEL_INFO or os.path.basename(model_size) != model_size:
        raise ValueError(f"Unknown FunASR model: {model_size}")

    removed: list[str] = []
    for bundle in bundles_for(model_size):
        path = get_model_path(bundle)
        if not os.path.isdir(path):
            continue
        if not is_within_directory(path, models_dir()):
            raise ValueError("Refusing to delete a path outside the models directory")
        shutil.rmtree(path)
        removed.append(path)

    if not removed:
        raise FileNotFoundError(os.path.join(models_dir(), "funasr", model_size))

    for path in removed:
        logger.info("Deleted FunASR bundle %s", path)
    return os.path.join(models_dir(), "funasr")
