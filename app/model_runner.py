"""Adapter around this repository's original TensorFlow model."""

from __future__ import annotations

import os
import sys
import threading
from pathlib import Path
from typing import Optional

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


class ModelRunner:
    def __init__(self) -> None:
        self.model = None
        self.reason: Optional[str] = None
        self._lock = threading.Lock()

    def load(self) -> bool:
        checkpoint_state = ROOT / "pretrained" / "checkpoint"
        index_files = list((ROOT / "pretrained").glob("*.index"))
        if not checkpoint_state.is_file() or not index_files:
            self.reason = "No TensorFlow checkpoint found in pretrained/."
            return False
        vocabulary_path = ROOT / "data" / "charList.txt"
        if not vocabulary_path.is_file():
            self.reason = "Required character vocabulary data/charList.txt is missing."
            return False
        try:
            # Import only when a checkpoint exists so the API/PDF stay usable
            # in environments where TensorFlow is not installed yet.
            os.environ["TF_USE_LEGACY_KERAS"] = "1"
            import tensorflow as tf

            tf.compat.v1.disable_eager_execution()
            from Model import DecoderType, Model

            vocabulary = vocabulary_path.read_text(encoding="utf-8")
            self.model = Model(vocabulary, DecoderType.BestPath, mustRestore=True)
            self.reason = None
            return True
        except Exception as exc:  # TensorFlow/runtime/checkpoint incompatibility
            self.model = None
            self.reason = f"Model could not be loaded: {exc}"
            return False

    @property
    def loaded(self) -> bool:
        return self.model is not None

    def predict(self, image_bytes: bytes) -> str:
        if not self.loaded:
            raise RuntimeError(self.reason or "Model is not loaded.")
        import cv2
        import numpy as np
        from DataLoader import Batch
        from SamplePreprocessor import preprocessor
        from Model import Model as OriginalModel

        encoded = np.frombuffer(image_bytes, dtype=np.uint8)
        image = cv2.imdecode(encoded, cv2.IMREAD_GRAYSCALE)
        if image is None or image.size == 0:
            raise ValueError("The uploaded file is not a readable image.")
        prepared = preprocessor(image, OriginalModel.imgSize)
        batch = Batch(None, [prepared])
        with self._lock:
            return self.model.inferBatch(batch)[0]
