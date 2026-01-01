"""
cross-validation training utilities for wav2vec2 pd classifier.

implements leave-one-subject-out (loso) and k-fold cross-validation
strategies specifically designed for medical speech data.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import warnings
from dataclasses import dataclass, field
from datetime import datetime
import json

import torch
import numpy as np
from torch.utils.data import Subset
from sklearn.model_selection import (
    LeaveOneGroupOut,
    StratifiedKFold,
    StratifiedGroupKFold
)
from tqdm import tqdm

from src.models.classifier import (
    Wav2Vec2PDClassifier,
    DataCollatorWithPadding,
    PDClassifierTrainer,
    create_training_args,
    evaluate_model_on_dataset
)


@dataclass
class CVResults:
    """container for cross-validation results."""
    
    fold_results: List[Dict] = field(default_factory=list)
    aggregated_metrics: Dict = field(default_factory=dict)
    config: Dict = field(default_factory=dict)
    
    def add_fold(self, fold_idx: int, train_metrics: Dict, val_metrics: Dict):
        """add results for a single fold."""
        self.fold_results.append({
            'fold': fold_idx,
            'train': train_metrics,
            'val': val_metrics
        })
    
    def aggregate(self):
        """compute aggregate statistics across folds."""
        if not self.fold_results:
            return
        
        metric_keys = list(self.fold_results[0]['val'].keys())
        
        for key in metric_keys:
            if isinstance(self.fold_results[0]['val'][key], (int, float)):
                values = [f['val'][key] for f in self.fold_results]
                self.aggregated_metrics[key] = {
                    'mean': np.mean(values),
                    'std': np.std(values),
                    'min': np.min(values),
                    'max': np.max(values),
                    'values': values
                }
    
    def summary(self) -> str:
        """generate summary string."""
        if not self.aggregated_metrics:
            self.aggregate()
        
        lines = [
            f"cross-validation results ({len(self.fold_results)} folds)",
            "-" * 50
        ]
        
        for key, stats in self.aggregated_metrics.items():
            if isinstance(stats, dict) and 'mean' in stats:
                lines.append(f"{key}: {stats['mean']:.4f} ± {stats['std']:.4f}")
        
        return "\n".join(lines)
    
    def to_dict(self) -> Dict:
        """convert to dictionary for serialization."""
        return {
            'fold_results': self.fold_results,
            'aggregated_metrics': self.aggregated_metrics,
            'config': self.config
        }
    
    def save(self, path: Union[str, Path]):
        """save results to json."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2, default=str)


class CrossValidationTrainer:
    """
    cross-validation trainer for wav2vec2 pd classification.
    
    supports leave-one-subject-out (loso) and stratified k-fold cv.
    """
    
    def __init__(
        self,
        model_name: str = "facebook/wav2vec2-base-960h",
        cv_strategy: str = "loso",
        n_folds: int = 5,
        freeze_feature_extractor: bool = True,
        num_epochs: int = 10,
        batch_size: int = 8,
        learning_rate: float = 1e-4,
        output_dir: Optional[Union[str, Path]] = None,
        device: str = "cuda"
    ):
        """
        args:
            model_name: pretrained model identifier
            cv_strategy: 'loso' for leave-one-subject-out or 'kfold' for stratified k-fold
            n_folds: number of folds (only used if cv_strategy='kfold')
            freeze_feature_extractor: whether to freeze cnn encoder
            num_epochs: training epochs per fold
            batch_size: batch size
            learning_rate: learning rate
            output_dir: directory for saving results
            device: device to train on
        """
        self.model_name = model_name
        self.cv_strategy = cv_strategy
        self.n_folds = n_folds
        self.freeze_feature_extractor = freeze_feature_extractor
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.device = device
        
        if output_dir:
            self.output_dir = Path(output_dir)
            self.output_dir.mkdir(parents=True, exist_ok=True)
        else:
            self.output_dir = None
    
    def _get_cv_splitter(self, labels: np.ndarray, groups: np.ndarray):
        """get cross-validation splitter."""
        if self.cv_strategy == "loso":
            return LeaveOneGroupOut()
        elif self.cv_strategy == "kfold":
            return StratifiedGroupKFold(n_splits=self.n_folds, shuffle=True, random_state=42)
        else:
            raise ValueError(f"unknown cv strategy: {self.cv_strategy}")
    
    def run_cv(
        self,
        dataset,
        max_folds: Optional[int] = None,
        save_models: bool = False
    ) -> CVResults:
        """
        run cross-validation.
        
        args:
            dataset: pytorch dataset with samples containing 'label' and 'subject_id'
            max_folds: maximum number of folds to run (for debugging)
            save_models: whether to save model checkpoints for each fold
        
        returns:
            CVResults object with all fold results
        """
        labels = np.array([s['label'] for s in dataset.samples])
        groups = np.array([s['subject_id'] for s in dataset.samples])
        
        splitter = self._get_cv_splitter(labels, groups)
        
        results = CVResults()
        results.config = {
            'model_name': self.model_name,
            'cv_strategy': self.cv_strategy,
            'n_folds': self.n_folds,
            'num_epochs': self.num_epochs,
            'batch_size': self.batch_size,
            'learning_rate': self.learning_rate,
            'freeze_feature_extractor': self.freeze_feature_extractor
        }
        
        indices = np.arange(len(dataset))
        splits = list(splitter.split(indices, labels, groups))
        
        if max_folds:
            splits = splits[:max_folds]
        
        n_total_folds = len(splits)
        print(f"running {n_total_folds}-fold cross-validation...")
        
        for fold_idx, (train_idx, val_idx) in enumerate(tqdm(splits, desc="cv folds")):
            print(f"\n{'='*60}")
            print(f"fold {fold_idx + 1}/{n_total_folds}")
            print(f"{'='*60}")
            
            train_metrics, val_metrics = self._train_fold(
                dataset=dataset,
                train_idx=train_idx.tolist(),
                val_idx=val_idx.tolist(),
                fold_idx=fold_idx,
                save_model=save_models
            )
            
            results.add_fold(fold_idx, train_metrics, val_metrics)
            
            print(f"fold {fold_idx + 1} - val accuracy: {val_metrics['accuracy']:.4f}, "
                  f"val f1: {val_metrics['f1']:.4f}")
            
            # cleanup
            torch.cuda.empty_cache()
        
        results.aggregate()
        
        print(f"\n{'='*60}")
        print(results.summary())
        print(f"{'='*60}")
        
        if self.output_dir:
            results.save(self.output_dir / "cv_results.json")
        
        return results
    
    def _train_fold(
        self,
        dataset,
        train_idx: List[int],
        val_idx: List[int],
        fold_idx: int,
        save_model: bool = False
    ) -> Tuple[Dict, Dict]:
        """train single fold."""
        train_dataset = Subset(dataset, train_idx)
        val_dataset = Subset(dataset, val_idx)
        
        # create fresh model for each fold
        classifier = Wav2Vec2PDClassifier(
            model_name=self.model_name,
            num_labels=2,
            freeze_feature_extractor=self.freeze_feature_extractor,
            device=self.device
        )
        
        # create training args
        fold_output_dir = self.output_dir / f"fold_{fold_idx}" if self.output_dir else Path(f"./fold_{fold_idx}")
        fold_output_dir.mkdir(parents=True, exist_ok=True)
        
        training_args = create_training_args(
            output_dir=str(fold_output_dir),
            num_epochs=self.num_epochs,
            batch_size=self.batch_size,
            learning_rate=self.learning_rate,
            fp16=True,
            eval_strategy="epoch",
            save_strategy="no" if not save_model else "epoch"
        )
        
        # create trainer
        data_collator = DataCollatorWithPadding(classifier.feature_extractor)
        
        trainer = PDClassifierTrainer(
            model=classifier,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            training_args=training_args,
            data_collator=data_collator
        )
        
        # train
        train_output = trainer.train()
        train_metrics = {
            'loss': train_output.get('train_loss', 0.0)
        }
        
        # evaluate
        val_metrics = evaluate_model_on_dataset(
            classifier,
            val_dataset,
            batch_size=self.batch_size
        )
        
        # save model if requested
        if save_model and self.output_dir:
            model_path = fold_output_dir / "model"
            classifier.save(model_path)
        
        # cleanup
        del classifier
        del trainer
        
        return train_metrics, val_metrics


def run_loso_cv(
    dataset,
    model_name: str = "facebook/wav2vec2-base-960h",
    num_epochs: int = 10,
    batch_size: int = 8,
    learning_rate: float = 1e-4,
    output_dir: Optional[str] = None,
    max_folds: Optional[int] = None
) -> CVResults:
    """
    convenience function for leave-one-subject-out cross-validation.
    
    args:
        dataset: pytorch dataset
        model_name: pretrained model
        num_epochs: training epochs
        batch_size: batch size
        learning_rate: learning rate
        output_dir: output directory
        max_folds: max folds to run
    
    returns:
        CVResults object
    """
    trainer = CrossValidationTrainer(
        model_name=model_name,
        cv_strategy="loso",
        num_epochs=num_epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        output_dir=output_dir
    )
    
    return trainer.run_cv(dataset, max_folds=max_folds)


def run_stratified_kfold_cv(
    dataset,
    n_folds: int = 5,
    model_name: str = "facebook/wav2vec2-base-960h",
    num_epochs: int = 10,
    batch_size: int = 8,
    learning_rate: float = 1e-4,
    output_dir: Optional[str] = None
) -> CVResults:
    """
    convenience function for stratified k-fold cross-validation.
    
    uses stratified group k-fold to maintain class balance and
    prevent subject leakage.
    
    args:
        dataset: pytorch dataset
        n_folds: number of folds
        model_name: pretrained model
        num_epochs: training epochs
        batch_size: batch size
        learning_rate: learning rate
        output_dir: output directory
    
    returns:
        CVResults object
    """
    trainer = CrossValidationTrainer(
        model_name=model_name,
        cv_strategy="kfold",
        n_folds=n_folds,
        num_epochs=num_epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        output_dir=output_dir
    )
    
    return trainer.run_cv(dataset)
