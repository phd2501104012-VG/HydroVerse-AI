from typing import Optional, Dict, List, Any, Tuple
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from config import CFG
from utils.logger import log


class DeepLearningModels:
    def __init__(self):
        self._models: Dict[str, Any] = {}
        self._has_tf = False
        self._has_torch = False
        self._check_frameworks()

    def _check_frameworks(self):
        try:
            import tensorflow as tf
            self._has_tf = True
        except ImportError:
            pass
        try:
            import torch
            self._has_torch = True
        except ImportError:
            pass

    def build_lstm(self, input_shape: Tuple[int, int], units: int = 128, num_layers: int = 2) -> Optional[Any]:
        if not self._has_tf:
            log.warning("TensorFlow not available for LSTM")
            return None
        try:
            import tensorflow as tf
            from tensorflow.keras import Sequential, layers
            model = Sequential()
            for i in range(num_layers):
                return_seq = i < num_layers - 1
                model.add(layers.LSTM(
                    units // (2 ** i) if i > 0 else units,
                    return_sequences=return_seq,
                    input_shape=input_shape if i == 0 else None,
                ))
            model.add(layers.Dense(1))
            model.compile(optimizer="adam", loss="mse", metrics=["mae"])
            return model
        except Exception as e:
            log.warning(f"LSTM build failed: {e}")
            return None

    def build_convlstm(self, input_shape: Tuple[int, int, int], num_filters: int = 64) -> Optional[Any]:
        if not self._has_tf:
            log.warning("TensorFlow not available for ConvLSTM")
            return None
        try:
            import tensorflow as tf
            from tensorflow.keras import Sequential, layers
            model = Sequential([
                layers.ConvLSTM2D(num_filters, kernel_size=(3, 3), padding="same", return_sequences=False, input_shape=input_shape),
                layers.Flatten(),
                layers.Dense(64, activation="relu"),
                layers.Dense(1),
            ])
            model.compile(optimizer="adam", loss="mse")
            return model
        except Exception as e:
            log.warning(f"ConvLSTM build failed: {e}")
            return None

    def build_cnn_lstm(self, input_shape: Tuple[int, int]) -> Optional[Any]:
        if not self._has_tf:
            return None
        try:
            import tensorflow as tf
            from tensorflow.keras import Sequential, layers
            model = Sequential([
                layers.Reshape((input_shape[0], input_shape[1], 1), input_shape=input_shape),
                layers.Conv2D(64, kernel_size=(3, 1), padding="same", activation="relu"),
                layers.MaxPooling2D(pool_size=(2, 1)),
                layers.Reshape((-1, 64)),
                layers.LSTM(128),
                layers.Dense(1),
            ])
            model.compile(optimizer="adam", loss="mse")
            return model
        except Exception as e:
            log.warning(f"CNN-LSTM build failed: {e}")
            return None

    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        model_type: str = "lstm",
        **kwargs,
    ) -> Optional[Any]:
        if model_type == "lstm":
            samples, features = X_train.shape
            model = self.build_lstm((samples, features) if False else (1, features))
            if model is None:
                return None
            X_train_3d = X_train.reshape((X_train.shape[0], 1, X_train.shape[1]))
            X_val_3d = X_val.reshape((X_val.shape[0], 1, X_val.shape[1]))
            history = model.fit(
                X_train_3d, y_train,
                validation_data=(X_val_3d, y_val),
                epochs=min(50, kwargs.get("epochs", CFG.ml.epochs)),
                batch_size=kwargs.get("batch_size", CFG.ml.batch_size),
                verbose=0,
            )
            return model
        return None

    def predict(self, model, X: np.ndarray) -> np.ndarray:
        if model is None:
            return np.zeros(X.shape[0])
        try:
            if len(X.shape) == 2:
                X = X.reshape((X.shape[0], 1, X.shape[1]))
            return model.predict(X, verbose=0).flatten()
        except Exception:
            return np.zeros(X.shape[0])
