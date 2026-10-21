"""FunASR speech recognition engine for Vocalinux.

Wraps the ``funasr-onnx`` runtime behind the Engine Protocol: the model pair
VocoType's native worker runs -- Paraformer-large contextual (vocab8404) for
ASR plus CT-Transformer for punctuation -- through Python instead of C++.

Design notes:

* One vocab8404 model covers Chinese and English, so the engine never routes
  by language; ``language`` is accepted for protocol parity and does not
  select a model.
* Punctuation is Chinese-only: CT-Transformer appends Chinese marks, so it
  runs only when the recognized text contains a CJK ideograph.
* ``initial_prompt`` is the space-joined custom-terms prompt, filtered into
  FunASR ``hotwords`` with VocoType's rules (whitespace-free, at most 10
  characters each, deduplicated, at most 1000).
"""

import logging
import os
from typing import Any, Optional

import numpy as np

from ...utils.funasr_model_info import FUNASR_MODEL_SIZES, get_model_path

logger = logging.getLogger(__name__)

# VocoType's hotword filter (TextNormalizer::build_native_hotwords): items are
# whitespace-separated, longer than 10 characters are dropped, duplicates are
# dropped case-insensitively, and at most 1000 survive. funasr-onnx splits on
# single spaces and computes lengths from the split items, so the prompt is
# normalized here rather than passed through.
_MAX_HOTWORDS = 1000
_MAX_HOTWORD_CHARS = 10

# CJK Unified Ideographs (incl. Extension A). Enough to tell Chinese apart
# from English, which is all the punctuation gate needs.
_CJK_RANGES = (("\u3400", "\u9fff"),)


def sanitize_hotwords(prompt: Optional[str]) -> str:
    """Turn a custom-terms prompt into a FunASR ``hotwords`` string.

    Returns ``""`` when nothing survives the filter; ``ContextualParaformer``
    requires the argument even then.
    """
    if not prompt:
        return ""
    seen: set[str] = set()
    hotwords: list[str] = []
    for item in prompt.split():
        if len(item) > _MAX_HOTWORD_CHARS:
            continue
        key = item.casefold()
        if key in seen:
            continue
        seen.add(key)
        hotwords.append(item)
        if len(hotwords) >= _MAX_HOTWORDS:
            break
    return " ".join(hotwords)


def _has_cjk(text: str) -> bool:
    """Return True when ``text`` contains a CJK ideograph."""
    return any(start <= char <= end for char in text for start, end in _CJK_RANGES)


class FunasrEngine:
    """FunASR backend implementing the Engine Protocol."""

    def __init__(
        self,
        model_size: Optional[str] = None,
        language: str = "auto",
        threads: Optional[int] = None,
    ) -> None:
        """Create a FunASR engine instance.

        Args:
            model_size: FunASR model id; the catalog currently holds "large".
            language: Kept for Engine parity. The vocab8404 model is bilingual
                and never reloaded for a language change.
            threads: ONNX intra-op threads; None picks min(4, cpu_count), the
                configuration the performance of this stack was measured at.
        """
        self.model_size = model_size or FUNASR_MODEL_SIZES[0]
        self.language = language
        self._threads = threads or max(1, min(4, os.cpu_count() or 1))
        self._asr: Any = None
        self._punc: Any = None
        self._model_initialized = False

    def init(self) -> None:
        """Load the ASR model, then the punctuation model (best effort)."""
        if self._model_initialized:
            return

        try:
            from funasr_onnx import ContextualParaformer, CT_Transformer
        except ImportError as e:
            logger.error("Failed to import funasr-onnx: %s", e)
            logger.error("Please install with: pip install funasr-onnx")
            raise

        if self.model_size not in FUNASR_MODEL_SIZES:
            logger.warning(
                "Model size '%s' not valid for FunASR. Valid options: %s. Using '%s' instead.",
                self.model_size,
                FUNASR_MODEL_SIZES,
                FUNASR_MODEL_SIZES[0],
            )
            self.model_size = FUNASR_MODEL_SIZES[0]

        asr_dir = get_model_path("asr")
        punc_dir = get_model_path("punc")
        logger.info(
            "Loading FunASR '%s' model from %s (threads=%d)",
            self.model_size,
            asr_dir,
            self._threads,
        )

        # quantize=True selects model_quant.onnx; the contextual hotword
        # encoder is stored as model_eb_quant.onnx because that is the file
        # name this loader asks for (see funasr_model_info).
        self._asr = ContextualParaformer(
            asr_dir,
            batch_size=1,
            quantize=True,
            intra_op_num_threads=self._threads,
        )

        try:
            self._punc = CT_Transformer(
                punc_dir,
                quantize=True,
                intra_op_num_threads=self._threads,
            )
        except Exception:
            # Punctuation is polish, not the transcript: a hand-pruned bundle
            # or a broken session must not keep dictation from starting.
            logger.warning(
                "FunASR punctuation model unavailable; text will carry no marks",
                exc_info=True,
            )
            self._punc = None

        self._model_initialized = True
        logger.info("FunASR engine initialized successfully.")

    def is_ready(self) -> bool:
        """Return True if the model has been loaded successfully."""
        return self._model_initialized and self._asr is not None

    def cleanup(self) -> None:
        """Release the loaded models."""
        self._asr = None
        self._punc = None
        self._model_initialized = False

    @staticmethod
    def _extract_text(result: Any) -> str:
        """Pull the transcript out of a ContextualParaformer result.

        The model returns ``[{"preds": (text, tokens)}]``; silence yields an
        empty list instead.
        """
        if not result:
            return ""
        entry = result[0]
        preds = entry.get("preds") if isinstance(entry, dict) else None
        if isinstance(preds, (tuple, list)):
            return str(preds[0]).strip() if preds else ""
        return str(preds or "").strip()

    def transcribe(
        self,
        audio_buffer: list[bytes],
        language: Optional[str] = None,
        initial_prompt: Optional[str] = None,
    ) -> str:
        """Transcribe the provided audio buffer.

        Args:
            audio_buffer: List of audio data chunks (16-bit PCM at 16kHz).
            language: Per-utterance language snapshot (#805). Unused on
                purpose: one bilingual model serves every language; None uses
                ``self.language``.
            initial_prompt: Custom-terms prompt, converted to hotwords.

        Returns:
            Recognized text, punctuated when it is Chinese.
        """
        if not self.is_ready():
            logger.warning("FunASR engine is not initialized")
            return ""
        if not audio_buffer:
            return ""

        try:
            audio_data = np.frombuffer(b"".join(audio_buffer), dtype=np.int16)
            if audio_data.size == 0:
                return ""
            # funasr-onnx accepts a float32 waveform directly (16kHz mono),
            # so no temporary file or re-encode is needed.
            waveform = audio_data.astype(np.float32) / 32768.0

            result = self._asr(waveform, hotwords=sanitize_hotwords(initial_prompt))
            text = self._extract_text(result)

            if text and self._punc is not None and _has_cjk(text):
                punctuated = self._punc(text)
                if punctuated and isinstance(punctuated[0], str) and punctuated[0].strip():
                    text = punctuated[0].strip()

            if text:
                logger.info("FunASR transcribed: '%s'", text)
            else:
                logger.debug("FunASR returned empty transcription")
            return text
        except Exception:
            # funasr-onnx raises ONNXRuntimeError (a bare Exception) for what
            # the session refuses; a failed segment must degrade to no text
            # instead of killing the dictation worker.
            logger.exception("Error in FunASR transcription")
            return ""
