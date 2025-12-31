"""
wav2vec2 classifier for parkinson's disease detection.

implements fine-tuning wrapper for wav2vec2 with classification head,
training utilities, and evaluation functions.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import warnings

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import (
    Wav2Vec2Model,
    Wav2Vec2ForSequenceClassification,
    Wav2Vec2FeatureExtractor,
    Wav2Vec2Config,
    TrainingArguments,
    Trainer,
    EvalPrediction
)
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    roc_auc_score,
    confusion_matrix
)


class Wav2Vec2PDClassifier:
    """
    wav2vec2 model wrapper for pd classification.

    handles model initialization, freezing strategies, and provides
    unified interface for training and inference.
    """

    def __init__(
        self,
        model_name: str = "facebook/wav2vec2-base-960h",
        num_labels: int = 2,
        freeze_feature_extractor: bool = True,
        freeze_encoder_layers: Optional[int] = None,
        dropout: float = 0.1,
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        """
        args:
            model_name: pretrained model identifier
            num_labels: number of classification labels
            freeze_feature_extractor: whether to freeze cnn feature encoder
            freeze_encoder_layers: number of transformer layers to freeze (none = freeze all)
            dropout: dropout probability for classification head
            device: device to load model on
        """
        self.model_name = model_name
        self.num_labels = num_labels
        self.device = device

        self.config = Wav2Vec2Config.from_pretrained(
            model_name,
            num_labels=num_labels,
            classifier_dropout=dropout,
            final_dropout=dropout
        )

        self.model = Wav2Vec2ForSequenceClassification.from_pretrained(
            model_name,
            config=self.config,
            ignore_mismatched_sizes=True
        )

        if freeze_feature_extractor:
            self._freeze_feature_extractor()

        if freeze_encoder_layers is not None:
            self._freeze_encoder_layers(freeze_encoder_layers)

        self.model = self.model.to(device)

        self.feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(model_name)

    def _freeze_feature_extractor(self):
        """freeze cnn feature extractor weights."""
        self.model.wav2vec2.feature_extractor._freeze_parameters()

    def _freeze_encoder_layers(self, num_layers: int):
        """freeze first n transformer encoder layers."""
        for i, layer in enumerate(self.model.wav2vec2.encoder.layers):
            if i < num_layers:
                for param in layer.parameters():
                    param.requires_grad = False

    def count_parameters(self) -> Dict[str, int]:
        """count trainable and total parameters."""
        total = sum(p.numel() for p in self.model.parameters())
        trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)

        return {
            'total': total,
            'trainable': trainable,
            'frozen': total - trainable,
            'trainable_percent': 100 * trainable / total if total > 0 else 0
        }

    def forward(self, input_values: torch.Tensor) -> torch.Tensor:
        """
        forward pass through model.

        args:
            input_values: audio tensor [batch_size, sequence_length]

        returns:
            logits [batch_size, num_labels]
        """
        return self.model(input_values.to(self.device)).logits

    def predict(
        self,
        input_values: torch.Tensor,
        return_probs: bool = True
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        make predictions on input audio.

        args:
            input_values: audio tensor
            return_probs: whether to return probabilities

        returns:
            predictions or (predictions, probabilities)
        """
        self.model.eval()

        with torch.no_grad():
            logits = self.forward(input_values)

            if return_probs:
                probs = torch.softmax(logits, dim=-1)
                preds = torch.argmax(probs, dim=-1)
                return preds.cpu(), probs.cpu()
            else:
                preds = torch.argmax(logits, dim=-1)
                return preds.cpu()

    def save(self, output_dir: Union[str, Path]):
        """save model checkpoint."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        self.model.save_pretrained(output_dir)
        self.feature_extractor.save_pretrained(output_dir)

    @classmethod
    def load(cls, checkpoint_dir: Union[str, Path], device: str = "cuda"):
        """load model from checkpoint."""
        checkpoint_dir = Path(checkpoint_dir)

        config = Wav2Vec2Config.from_pretrained(checkpoint_dir)
        model = Wav2Vec2ForSequenceClassification.from_pretrained(checkpoint_dir)
        feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(checkpoint_dir)

        instance = cls.__new__(cls)
        instance.model = model.to(device)
        instance.feature_extractor = feature_extractor
        instance.device = device
        instance.num_labels = config.num_labels
        instance.config = config

        return instance


class DataCollatorWithPadding:
    """
    data collator for wav2vec2 that handles variable length audio.

    pads sequences to same length within batch for efficient processing.
    """

    def __init__(
        self,
        feature_extractor: Wav2Vec2FeatureExtractor,
        padding: Union[bool, str] = True,
        max_length: Optional[int] = None,
        return_attention_mask: bool = True
    ):
        """
        args:
            feature_extractor: wav2vec2 feature extractor
            padding: padding strategy
            max_length: maximum sequence length
            return_attention_mask: whether to return attention mask
        """
        self.feature_extractor = feature_extractor
        self.padding = padding
        self.max_length = max_length
        self.return_attention_mask = return_attention_mask

    def __call__(self, features: List[Dict]) -> Dict[str, torch.Tensor]:
        """
        collate batch of samples.

        args:
            features: list of sample dicts with 'input_values' and 'label'

        returns:
            batched dict with padded tensors
        """
        input_values = [f['input_values'] for f in features]
        labels = torch.tensor([f['label'] for f in features], dtype=torch.long)

        batch = self.feature_extractor(
            input_values,
            sampling_rate=16000,
            return_tensors="pt",
            padding=self.padding,
            max_length=self.max_length,
            truncation=True,
            return_attention_mask=self.return_attention_mask
        )

        batch['labels'] = labels

        return batch


def compute_metrics(pred: EvalPrediction) -> Dict[str, float]:
    """
    compute evaluation metrics for classification.

    args:
        pred: predictions and labels from trainer

    returns:
        dictionary of metrics
    """
    labels = pred.label_ids
    preds = np.argmax(pred.predictions, axis=1)

    probs = torch.softmax(torch.tensor(pred.predictions), dim=-1).numpy()

    accuracy = accuracy_score(labels, preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, preds, average='binary', zero_division=0
    )

    try:
        auc = roc_auc_score(labels, probs[:, 1])
    except ValueError:
        auc = 0.0

    cm = confusion_matrix(labels, preds)

    tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)

    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0

    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'auc': auc,
        'sensitivity': sensitivity,
        'specificity': specificity,
        'true_positives': int(tp),
        'true_negatives': int(tn),
        'false_positives': int(fp),
        'false_negatives': int(fn)
    }


def create_training_args(
    output_dir: Union[str, Path],
    num_epochs: int = 20,
    batch_size: int = 8,
    learning_rate: float = 1e-4,
    warmup_ratio: float = 0.1,
    gradient_accumulation_steps: int = 4,
    eval_strategy: str = "epoch",
    save_strategy: str = "epoch",
    logging_steps: int = 10,
    fp16: bool = True,
    dataloader_num_workers: int = 2,
    seed: int = 42
) -> TrainingArguments:
    """
    create training arguments with sensible defaults for pd classification.

    args:
        output_dir: directory for checkpoints and logs
        num_epochs: number of training epochs
        batch_size: batch size per device
        learning_rate: learning rate
        warmup_ratio: warmup ratio for lr scheduler
        gradient_accumulation_steps: gradient accumulation
        eval_strategy: evaluation strategy
        save_strategy: checkpoint save strategy
        logging_steps: logging frequency
        fp16: whether to use mixed precision
        dataloader_num_workers: number of dataloader workers
        seed: random seed

    returns:
        training arguments object
    """
    return TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=learning_rate,
        warmup_ratio=warmup_ratio,
        weight_decay=0.01,
        fp16=fp16,
        evaluation_strategy=eval_strategy,
        save_strategy=save_strategy,
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        logging_steps=logging_steps,
        logging_dir=str(Path(output_dir) / "logs"),
        dataloader_num_workers=dataloader_num_workers,
        seed=seed,
        gradient_checkpointing=True,
        report_to=["tensorboard"],
        push_to_hub=False
    )


class PDClassifierTrainer:
    """
    trainer wrapper for pd classification with wav2vec2.

    provides higher-level interface for training and evaluation.
    """

    def __init__(
        self,
        model: Wav2Vec2PDClassifier,
        train_dataset,
        eval_dataset,
        training_args: TrainingArguments,
        data_collator: Optional[DataCollatorWithPadding] = None
    ):
        """
        args:
            model: wav2vec2 classifier instance
            train_dataset: training dataset
            eval_dataset: evaluation dataset
            training_args: training arguments
            data_collator: data collator for batching
        """
        self.model = model

        if data_collator is None:
            data_collator = DataCollatorWithPadding(
                feature_extractor=model.feature_extractor
            )

        self.trainer = Trainer(
            model=model.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=data_collator,
            compute_metrics=compute_metrics
        )

    def train(self) -> Dict:
        """train the model."""
        train_result = self.trainer.train()

        metrics = train_result.metrics
        self.trainer.save_model()
        self.trainer.save_state()

        return metrics

    def evaluate(self, dataset=None) -> Dict:
        """evaluate the model."""
        return self.trainer.evaluate(dataset)

    def predict(self, dataset) -> Tuple[np.ndarray, np.ndarray, Dict]:
        """
        make predictions on dataset.

        returns:
            (predictions, labels, metrics)
        """
        output = self.trainer.predict(dataset)

        preds = np.argmax(output.predictions, axis=1)

        return preds, output.label_ids, output.metrics


def evaluate_model_on_dataset(
    model: Wav2Vec2PDClassifier,
    dataset,
    batch_size: int = 16
) -> Dict[str, float]:
    """
    evaluate model on dataset without trainer.

    args:
        model: classifier model
        dataset: evaluation dataset
        batch_size: batch size for evaluation

    returns:
        evaluation metrics
    """
    model.model.eval()

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        collate_fn=DataCollatorWithPadding(model.feature_extractor),
        shuffle=False
    )

    all_preds = []
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for batch in dataloader:
            input_values = batch['input_values'].to(model.device)
            labels = batch['labels']

            logits = model.model(input_values).logits
            probs = torch.softmax(logits, dim=-1)
            preds = torch.argmax(probs, dim=-1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())
            all_probs.extend(probs.cpu().numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)

    accuracy = accuracy_score(all_labels, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average='binary', zero_division=0
    )

    try:
        auc = roc_auc_score(all_labels, all_probs[:, 1])
    except ValueError:
        auc = 0.0

    cm = confusion_matrix(all_labels, all_preds)
    tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)

    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'auc': auc,
        'sensitivity': tp / (tp + fn) if (tp + fn) > 0 else 0.0,
        'specificity': tn / (tn + fp) if (tn + fp) > 0 else 0.0,
        'confusion_matrix': cm.tolist()
    }
