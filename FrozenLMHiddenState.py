"""
FrozenLMHiddenState.py

Extract frozen hidden states from a pretrained causal language model and
prepare next-token training examples for ObliqueXGBoostClassifier.

Formal experiment default:
    checkpoint:         ./checkpoints/TinyStories-8M
    hidden_state_index: 1

For GPT-2-style Hugging Face causal LMs:
    hidden_states[0] = embedding output before Transformer block 1
    hidden_states[1] = output of Transformer block 1
    hidden_states[2] = output of Transformer block 2
    ...

Thus the default hidden_states[1] is already contextualized by the first
causal self-attention block.

Typical use
-----------
extractor = FrozenLMHiddenStateExtractor(
    model_name="./checkpoints/TinyStories-8M",
    hidden_state_index=1,
    max_length=256,
    batch_size=16,
    context_window=1,
)

X, y = extractor.make_next_token_dataset(texts)

clf = ObliqueXGBoostClassifier(
    input_dim=extractor.output_dim,
    output_dim=extractor.vocab_size,
    ...
)
clf.fit(X, y)
"""

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


TextInput = Union[str, Sequence[str]]


class FrozenLMHiddenStateExtractor:
    """
    Frozen causal-LM feature extractor.

    Parameters
    ----------
    model_name:
        Local checkpoint path or Hugging Face model id.
    hidden_state_index:
        Which item from outputs.hidden_states to use.
        Default = 1, i.e. output of the first Transformer block for
        GPT-2-style Hugging Face models.
    max_length:
        Maximum tokenized sequence length.
    batch_size:
        Number of texts processed per LM forward pass.
    device:
        "cuda", "cpu", "mps", etc. Auto-selected when None.
    context_window:
        Number of consecutive hidden vectors concatenated before passing
        them to the tree model. Because hidden_states[1] is already
        contextualized, context_window=1 is the recommended first baseline.
    dtype:
        NumPy dtype returned to the downstream tree model.
    trust_remote_code:
        Passed to Hugging Face from_pretrained().
    """

    def __init__(
        self,
        model_name: str = "./checkpoints/TinyStories-8M",
        hidden_state_index: int = 1,
        max_length: int = 256,
        batch_size: int = 16,
        device: Optional[str] = None,
        context_window: int = 1,
        dtype=np.float32,
        trust_remote_code: bool = False,
    ) -> None:
        if hidden_state_index < 0:
            raise ValueError("hidden_state_index must be >= 0")
        if max_length < 2:
            raise ValueError("max_length must be >= 2")
        if batch_size < 1:
            raise ValueError("batch_size must be >= 1")
        if context_window < 1:
            raise ValueError("context_window must be >= 1")

        self.model_name = model_name
        self.hidden_state_index = int(hidden_state_index)
        self.max_length = int(max_length)
        self.batch_size = int(batch_size)
        self.context_window = int(context_window)
        self.dtype = dtype
        self.trust_remote_code = trust_remote_code

        if device is None:
            if torch.cuda.is_available():
                device = "cuda"
            elif (
                getattr(torch.backends, "mps", None) is not None
                and torch.backends.mps.is_available()
            ):
                device = "mps"
            else:
                device = "cpu"

        self.device = torch.device(device)

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            trust_remote_code=self.trust_remote_code,
        )

        if self.tokenizer.pad_token_id is None:
            if self.tokenizer.eos_token_id is None:
                raise ValueError(
                    "Tokenizer has neither pad_token_id nor eos_token_id."
                )
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            trust_remote_code=self.trust_remote_code,
        )
        self.model.config.pad_token_id = self.tokenizer.pad_token_id

        # Freeze checkpoint completely.
        self.model.eval()
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)

        self.model.to(self.device)

        self.hidden_size = self._infer_hidden_size()
        self.vocab_size = int(self.model.config.vocab_size)
        self.num_hidden_layers = self._infer_num_hidden_layers()

        # hidden_states contains embeddings + one entry per transformer block.
        max_hidden_index = self.num_hidden_layers
        if self.hidden_state_index > max_hidden_index:
            raise ValueError(
                f"hidden_state_index={self.hidden_state_index} is invalid for "
                f"a model with {self.num_hidden_layers} Transformer blocks. "
                f"Valid indices are 0..{max_hidden_index}."
            )

        self.output_dim = self.hidden_size * self.context_window

    def _infer_hidden_size(self) -> int:
        cfg = self.model.config
        for name in ("hidden_size", "n_embd", "d_model"):
            if hasattr(cfg, name):
                return int(getattr(cfg, name))
        raise AttributeError(
            "Could not infer hidden dimension from model config."
        )

    def _infer_num_hidden_layers(self) -> int:
        cfg = self.model.config
        for name in ("num_hidden_layers", "n_layer", "num_layers"):
            if hasattr(cfg, name):
                return int(getattr(cfg, name))
        raise AttributeError(
            "Could not infer number of Transformer blocks from model config."
        )

    @staticmethod
    def _normalize_texts(texts: TextInput) -> List[str]:
        if isinstance(texts, str):
            texts = [texts]
        else:
            texts = list(texts)

        if not texts:
            raise ValueError("texts must contain at least one string")
        if not all(isinstance(x, str) for x in texts):
            raise TypeError("Every item in texts must be a string")

        return texts

    def _tokenize_batch(self, texts: Sequence[str]):
        return self.tokenizer(
            list(texts),
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
            return_attention_mask=True,
        )

    @torch.inference_mode()
    def extract_hidden_state(
        self,
        texts: TextInput,
        return_numpy: bool = True,
    ):
        """
        Extract the configured hidden state.

        Returns
        -------
        hidden:
            [batch, sequence_length, hidden_size]
        attention_mask:
            [batch, sequence_length]
        input_ids:
            [batch, sequence_length]
        """
        texts = self._normalize_texts(texts)

        all_hidden = []
        all_masks = []
        all_ids = []

        for start in range(0, len(texts), self.batch_size):
            batch_texts = texts[start : start + self.batch_size]

            encoded = self.tokenizer(
                batch_texts,
                padding="max_length",
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
                return_attention_mask=True,
            )

            input_ids = encoded["input_ids"].to(self.device)
            attention_mask = encoded["attention_mask"].to(self.device)

            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
                use_cache=False,
                return_dict=True,
            )

            hidden_states = outputs.hidden_states
            if hidden_states is None:
                raise RuntimeError("Model returned hidden_states=None")

            if self.hidden_state_index >= len(hidden_states):
                raise RuntimeError(
                    f"Requested hidden_states[{self.hidden_state_index}], "
                    f"but model returned only {len(hidden_states)} states."
                )

            hidden = hidden_states[self.hidden_state_index]

            all_hidden.append(hidden.detach().cpu())
            all_masks.append(attention_mask.detach().cpu())
            all_ids.append(input_ids.detach().cpu())

        hidden = torch.cat(all_hidden, dim=0)
        masks = torch.cat(all_masks, dim=0)
        ids = torch.cat(all_ids, dim=0)

        if return_numpy:
            return (
                hidden.numpy().astype(self.dtype, copy=False),
                masks.numpy(),
                ids.numpy(),
            )

        return hidden, masks, ids

    # Compatibility-friendly explicit method name.
    def extract_hidden_state1(
        self,
        texts: TextInput,
        return_numpy: bool = True,
    ):
        if self.hidden_state_index != 1:
            raise ValueError(
                "extract_hidden_state1() requires hidden_state_index=1"
            )
        return self.extract_hidden_state(
            texts=texts,
            return_numpy=return_numpy,
        )

    @torch.inference_mode()
    def make_next_token_dataset(
        self,
        texts: TextInput,
        max_samples: Optional[int] = None,
        shuffle: bool = False,
        random_state: int = 42,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Convert texts into next-token classification examples.

        For context_window = 1:
            X_i = hidden_states[index][t]
            y_i = input_ids[t + 1]

        For context_window = K:
            X_i = concat(
                h[t-K+1], ..., h[t]
            )
            y_i = input_ids[t + 1]

        Padding positions are excluded.
        """
        texts = self._normalize_texts(texts)

        X_parts: List[np.ndarray] = []
        y_parts: List[int] = []

        collected = 0
        K = self.context_window

        for start in range(0, len(texts), self.batch_size):
            batch_texts = texts[start : start + self.batch_size]
            encoded = self._tokenize_batch(batch_texts)

            input_ids = encoded["input_ids"].to(self.device)
            attention_mask = encoded["attention_mask"].to(self.device)

            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
                use_cache=False,
                return_dict=True,
            )

            hidden_states = outputs.hidden_states
            if hidden_states is None:
                raise RuntimeError("Model returned hidden_states=None")

            hidden = hidden_states[self.hidden_state_index]

            batch_size = hidden.shape[0]

            for b in range(batch_size):
                valid_len = int(attention_mask[b].sum().item())

                # Need K visible states and one future target.
                if valid_len <= K:
                    continue

                for t in range(K - 1, valid_len - 1):
                    start_pos = t - K + 1

                    feature = hidden[
                        b,
                        start_pos : t + 1,
                        :
                    ].reshape(-1)

                    target = int(input_ids[b, t + 1].item())

                    X_parts.append(
                        feature.detach().cpu().numpy().astype(
                            self.dtype,
                            copy=False,
                        )
                    )
                    y_parts.append(target)

                    collected += 1

                    if (
                        max_samples is not None
                        and collected >= max_samples
                    ):
                        break

                if (
                    max_samples is not None
                    and collected >= max_samples
                ):
                    break

            if (
                max_samples is not None
                and collected >= max_samples
            ):
                break

        if not X_parts:
            raise ValueError(
                "No next-token samples produced. "
                "Use longer texts or reduce context_window."
            )

        X = np.stack(X_parts, axis=0)
        y = np.asarray(y_parts, dtype=np.int64)

        if shuffle:
            rng = np.random.default_rng(random_state)
            order = rng.permutation(len(y))
            X = X[order]
            y = y[order]

        return X, y

    def describe(self) -> dict:
        return {
            "model_name": self.model_name,
            "device": str(self.device),
            "hidden_state_index": self.hidden_state_index,
            "hidden_size": self.hidden_size,
            "num_hidden_layers": self.num_hidden_layers,
            "context_window": self.context_window,
            "output_dim": self.output_dim,
            "vocab_size": self.vocab_size,
            "max_length": self.max_length,
        }


class FrozenLMObliquePipeline:
    """
    Adapter connecting the frozen LM feature extractor to a classifier
    exposing fit(X, y) and predict()/predict_logits()/predict_proba().
    """

    def __init__(
        self,
        classifier,
        extractor: Optional[FrozenLMHiddenStateExtractor] = None,
        **extractor_kwargs,
    ):
        self.extractor = (
            extractor
            if extractor is not None
            else FrozenLMHiddenStateExtractor(**extractor_kwargs)
        )
        self.classifier = classifier

    @property
    def input_dim(self) -> int:
        return self.extractor.output_dim

    @property
    def output_dim(self) -> int:
        return self.extractor.vocab_size

    def fit(
        self,
        texts: TextInput,
        max_samples: Optional[int] = None,
        shuffle: bool = True,
        random_state: int = 42,
        **classifier_fit_kwargs,
    ):
        X, y = self.extractor.make_next_token_dataset(
            texts=texts,
            max_samples=max_samples,
            shuffle=shuffle,
            random_state=random_state,
        )
        self.classifier.fit(
            X,
            y,
            **classifier_fit_kwargs,
        )
        return self

    def make_dataset(
        self,
        texts: TextInput,
        max_samples: Optional[int] = None,
        shuffle: bool = False,
        random_state: int = 42,
    ):
        return self.extractor.make_next_token_dataset(
            texts=texts,
            max_samples=max_samples,
            shuffle=shuffle,
            random_state=random_state,
        )

    def predict_next_token_ids(
        self,
        texts: TextInput,
        max_samples: Optional[int] = None,
    ):
        X, y = self.extractor.make_next_token_dataset(
            texts=texts,
            max_samples=max_samples,
            shuffle=False,
        )

        if hasattr(self.classifier, "predict"):
            pred = self.classifier.predict(X)
        elif hasattr(self.classifier, "predict_logits"):
            logits = self.classifier.predict_logits(X)
            pred = np.argmax(logits, axis=1)
        elif hasattr(self.classifier, "predict_proba"):
            probs = self.classifier.predict_proba(X)
            pred = np.argmax(probs, axis=1)
        else:
            raise AttributeError(
                "classifier must provide predict(), predict_logits(), "
                "or predict_proba()."
            )

        return np.asarray(pred), y


# Explicit aliases for this experiment.
FrozenLMHiddenState1Extractor = FrozenLMHiddenStateExtractor


if __name__ == "__main__":
    extractor = FrozenLMHiddenStateExtractor(
        model_name="./checkpoints/TinyStories-8M",
        hidden_state_index=1,
        max_length=64,
        batch_size=2,
        context_window=1,
    )

    print(extractor.describe())

    demo_texts = [
        "Once upon a time there was a little girl.",
        "The small dog ran through the garden.",
    ]

    X, y = extractor.make_next_token_dataset(
        demo_texts,
        max_samples=32,
    )

    print("X shape:", X.shape)
    print("y shape:", y.shape)
    print("first target token:", extractor.tokenizer.decode([int(y[0])]))
