"""
TinyTemporal model implementation for AI Medication Reminder.

Implements a lightweight temporal neural network according to pipeline specification A.5.
"""

from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd

from ..config import ModelConfig


class TinyTemporalModel:
    """Lightweight temporal neural network for medication reminder response prediction."""

    def __init__(self, config: ModelConfig):
        self.config = config
        self.model = None
        self.scaler = None
        self.is_fitted = False
        self.feature_names = None

        # Try to import torch, but make it optional
        try:
            import torch
            import torch.nn as nn
            import torch.optim as optim
            from torch.utils.data import DataLoader, TensorDataset

            self.torch = torch
            self.nn = nn
            self.optim = optim
            self.DataLoader = DataLoader
            self.TensorDataset = TensorDataset
            self.torch_available = True
        except ImportError:
            self.torch_available = False
            print("Warning: PyTorch not available. TinyTemporal model will not work.")

    def _create_model(self, input_size: int, sequence_length: int) -> "torch.nn.Module":
        """
        Create the TinyTemporal neural network architecture.

        Args:
            input_size: Number of input features
            sequence_length: Length of input sequences

        Returns:
            PyTorch model
        """
        if not self.torch_available:
            raise ImportError("PyTorch is required for TinyTemporal model")

        class TinyTemporalNet(self.nn.Module):
            def __init__(self, input_size, hidden_size, sequence_length):
                super().__init__()
                self.hidden_size = hidden_size
                self.sequence_length = sequence_length

                # Simple LSTM layer
                self.lstm = self.nn.LSTM(
                    input_size=input_size,
                    hidden_size=hidden_size,
                    batch_first=True,
                    dropout=0.1,
                )

                # Output layer
                self.fc = self.nn.Linear(hidden_size, 1)
                self.sigmoid = self.nn.Sigmoid()

            def forward(self, x):
                # x shape: (batch_size, sequence_length, input_size)
                lstm_out, _ = self.lstm(x)

                # Take the last output
                last_output = lstm_out[:, -1, :]

                # Apply final layer
                output = self.fc(last_output)
                output = self.sigmoid(output)

                return output.squeeze()

        return TinyTemporalNet(input_size, self.config.tiny_hidden, sequence_length)

    def _prepare_sequences(
        self, X: pd.DataFrame, y: Optional[pd.Series] = None
    ) -> Tuple:
        """
        Prepare sequence data for training/prediction.

        Args:
            X: DataFrame with 'sequences' column containing numpy arrays
            y: Optional target series

        Returns:
            Tuple of prepared tensors
        """
        if not self.torch_available:
            raise ImportError("PyTorch is required for TinyTemporal model")

        # Extract sequences from DataFrame
        sequences = np.stack(X["sequences"].values)

        # Convert to tensors
        X_tensor = self.torch.FloatTensor(sequences)

        if y is not None:
            y_tensor = self.torch.FloatTensor(y.values)
            return X_tensor, y_tensor
        else:
            return (X_tensor,)

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "TinyTemporalModel":
        """
        Train the TinyTemporal model.

        Args:
            X: DataFrame with sequences
            y: Target Series

        Returns:
            Self for method chaining
        """
        if not self.torch_available:
            raise ImportError("PyTorch is required for TinyTemporal model")

        print(f"Training TinyTemporal model with {len(X)} sequences")

        # Prepare data
        X_tensor, y_tensor = self._prepare_sequences(X, y)

        # Get dimensions
        batch_size, sequence_length, input_size = X_tensor.shape

        # Create model
        self.model = self._create_model(input_size, sequence_length)

        # Create data loader
        dataset = self.TensorDataset(X_tensor, y_tensor)
        dataloader = self.DataLoader(
            dataset,
            batch_size=min(self.config.tiny_batch_size, len(dataset)),
            shuffle=True,
        )

        # Setup training
        criterion = self.nn.BCELoss()
        optimizer = self.optim.Adam(self.model.parameters(), lr=self.config.tiny_lr)

        # Training loop
        self.model.train()
        for epoch in range(self.config.tiny_epochs):
            total_loss = 0
            for batch_X, batch_y in dataloader:
                optimizer.zero_grad()

                outputs = self.model(batch_X)
                loss = criterion(outputs, batch_y)

                loss.backward()
                optimizer.step()

                total_loss += loss.item()

            avg_loss = total_loss / len(dataloader)
            if (epoch + 1) % max(1, self.config.tiny_epochs // 5) == 0:
                print(
                    f"Epoch {epoch + 1}/{self.config.tiny_epochs}, Loss: {avg_loss:.4f}"
                )

        self.is_fitted = True
        print("TinyTemporal model training completed")
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Make predictions.

        Args:
            X: DataFrame with sequences

        Returns:
            Binary predictions
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")

        proba = self.predict_proba(X)
        return (proba > 0.5).astype(int)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict class probabilities.

        Args:
            X: DataFrame with sequences

        Returns:
            Probability predictions
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")

        if not self.torch_available:
            raise ImportError("PyTorch is required for TinyTemporal model")

        # Prepare data
        (X_tensor,) = self._prepare_sequences(X)

        # Make predictions
        self.model.eval()
        with self.torch.no_grad():
            predictions = self.model(X_tensor)

        return predictions.numpy()

    def save(self, path: Path) -> None:
        """
        Save the trained model.

        Args:
            path: Path to save the model
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before saving")

        if not self.torch_available:
            raise ImportError("PyTorch is required for TinyTemporal model")

        model_data = {
            "model_state_dict": self.model.state_dict(),
            "config": self.config,
            "feature_names": self.feature_names,
        }

        self.torch.save(model_data, path)
        print(f"TinyTemporal model saved to {path}")

    @classmethod
    def load(cls, path: Path) -> "TinyTemporalModel":
        """
        Load a trained model.

        Args:
            path: Path to the saved model

        Returns:
            Loaded TinyTemporalModel instance
        """
        try:
            import torch
        except ImportError:
            raise ImportError("PyTorch is required to load TinyTemporal model")

        model_data = torch.load(path, map_location="cpu")

        instance = cls(model_data["config"])
        instance.feature_names = model_data["feature_names"]

        # We need to recreate the model architecture
        # This is a simplified version - in practice you'd store architecture info
        # For now, assume standard dimensions
        input_size = 10  # This should be stored in the saved data
        sequence_length = instance.config.window_size

        instance.model = instance._create_model(input_size, sequence_length)
        instance.model.load_state_dict(model_data["model_state_dict"])
        instance.is_fitted = True

        print(f"TinyTemporal model loaded from {path}")
        return instance
