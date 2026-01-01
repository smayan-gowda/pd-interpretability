"""
experiment tracking and logging utilities.

provides structured experiment management including configuration,
metrics logging, checkpointing, and reproducibility tracking.
"""

from pathlib import Path
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, field, asdict
from datetime import datetime
import json
import hashlib
import subprocess
import sys
import warnings

import numpy as np


@dataclass
class ExperimentConfig:
    """experiment configuration container."""
    
    # experiment metadata
    experiment_name: str = "unnamed_experiment"
    experiment_type: str = "training"
    description: str = ""
    
    # model configuration
    model_name: str = "facebook/wav2vec2-base-960h"
    num_labels: int = 2
    freeze_feature_extractor: bool = True
    freeze_encoder_layers: int = 0
    dropout: float = 0.1
    
    # data configuration
    dataset: str = "italian_pvs"
    max_duration: float = 10.0
    target_sr: int = 16000
    segment_duration: float = 3.0
    segment_overlap: float = 0.5
    
    # training configuration
    num_epochs: int = 20
    batch_size: int = 8
    gradient_accumulation_steps: int = 4
    learning_rate: float = 1e-4
    warmup_ratio: float = 0.1
    weight_decay: float = 0.01
    
    # split configuration
    test_size: float = 0.2
    val_size: float = 0.1
    cv_strategy: str = "holdout"
    n_folds: int = 5
    
    # reproducibility
    random_seed: int = 42
    
    def to_dict(self) -> Dict:
        """convert to dictionary."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, d: Dict) -> "ExperimentConfig":
        """create from dictionary."""
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})
    
    def save(self, path: Union[str, Path]):
        """save configuration to json."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
    
    @classmethod
    def load(cls, path: Union[str, Path]) -> "ExperimentConfig":
        """load configuration from json."""
        with open(path, 'r') as f:
            return cls.from_dict(json.load(f))


class MetricsLogger:
    """
    logger for experiment metrics with history tracking.
    
    tracks metrics over epochs/steps and provides aggregation utilities.
    """
    
    def __init__(self):
        """initialize metrics logger."""
        self.history: Dict[str, List[Dict]] = {}
        self.current_step = 0
        self.current_epoch = 0
    
    def log(
        self,
        metrics: Dict[str, Any],
        step: Optional[int] = None,
        epoch: Optional[int] = None,
        phase: str = "train"
    ):
        """
        log metrics for a step/epoch.
        
        args:
            metrics: dictionary of metric values
            step: current step (optional)
            epoch: current epoch (optional)
            phase: 'train', 'val', or 'test'
        """
        if step is not None:
            self.current_step = step
        if epoch is not None:
            self.current_epoch = epoch
        
        entry = {
            'step': self.current_step,
            'epoch': self.current_epoch,
            'timestamp': datetime.now().isoformat(),
            **metrics
        }
        
        if phase not in self.history:
            self.history[phase] = []
        
        self.history[phase].append(entry)
    
    def get_best(self, metric: str, phase: str = "val", mode: str = "max") -> Dict:
        """
        get best metric value and associated entry.
        
        args:
            metric: metric name
            phase: phase to search
            mode: 'max' or 'min'
        
        returns:
            best entry dictionary
        """
        if phase not in self.history:
            return {}
        
        entries = [e for e in self.history[phase] if metric in e]
        
        if not entries:
            return {}
        
        if mode == "max":
            return max(entries, key=lambda x: x[metric])
        else:
            return min(entries, key=lambda x: x[metric])
    
    def get_metric_history(self, metric: str, phase: str = "train") -> List[float]:
        """get history of a specific metric."""
        if phase not in self.history:
            return []
        
        return [e.get(metric) for e in self.history[phase] if metric in e]
    
    def summary(self) -> Dict:
        """generate summary statistics."""
        summary = {}
        
        for phase, entries in self.history.items():
            if not entries:
                continue
            
            phase_summary = {}
            
            # get all numeric metrics
            metric_names = set()
            for entry in entries:
                for k, v in entry.items():
                    if isinstance(v, (int, float)) and k not in ['step', 'epoch']:
                        metric_names.add(k)
            
            for metric in metric_names:
                values = [e[metric] for e in entries if metric in e]
                if values:
                    phase_summary[metric] = {
                        'final': values[-1],
                        'best': max(values),
                        'mean': np.mean(values),
                        'std': np.std(values)
                    }
            
            summary[phase] = phase_summary
        
        return summary
    
    def to_dict(self) -> Dict:
        """convert to dictionary."""
        return {
            'history': self.history,
            'summary': self.summary()
        }
    
    def save(self, path: Union[str, Path]):
        """save metrics to json."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2, default=str)


class ExperimentTracker:
    """
    comprehensive experiment tracker for reproducibility.
    
    manages experiment lifecycle including configuration, logging,
    checkpointing, and artifact management.
    """
    
    def __init__(
        self,
        experiment_name: str,
        output_dir: Union[str, Path],
        config: Optional[ExperimentConfig] = None
    ):
        """
        initialize experiment tracker.
        
        args:
            experiment_name: unique experiment name
            output_dir: base output directory
            config: experiment configuration
        """
        self.experiment_name = experiment_name
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # create experiment directory
        self.output_dir = Path(output_dir) / f"{experiment_name}_{self.timestamp}"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # subdirectories
        self.checkpoints_dir = self.output_dir / "checkpoints"
        self.logs_dir = self.output_dir / "logs"
        self.figures_dir = self.output_dir / "figures"
        
        for d in [self.checkpoints_dir, self.logs_dir, self.figures_dir]:
            d.mkdir(exist_ok=True)
        
        # configuration
        self.config = config or ExperimentConfig(experiment_name=experiment_name)
        
        # metrics
        self.metrics = MetricsLogger()
        
        # experiment metadata
        self.metadata = {
            'experiment_name': experiment_name,
            'timestamp': self.timestamp,
            'status': 'initialized'
        }
        
        # capture environment
        self._capture_environment()
    
    def _capture_environment(self):
        """capture environment information for reproducibility."""
        import torch
        
        self.metadata['environment'] = {
            'python_version': sys.version,
            'pytorch_version': torch.__version__,
            'cuda_available': torch.cuda.is_available(),
            'cuda_version': torch.version.cuda if torch.cuda.is_available() else None,
            'gpu_name': torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
        }
        
        # capture git info if available
        try:
            git_hash = subprocess.check_output(
                ['git', 'rev-parse', 'HEAD'],
                stderr=subprocess.DEVNULL
            ).decode('utf-8').strip()
            
            git_branch = subprocess.check_output(
                ['git', 'rev-parse', '--abbrev-ref', 'HEAD'],
                stderr=subprocess.DEVNULL
            ).decode('utf-8').strip()
            
            git_dirty = subprocess.check_output(
                ['git', 'status', '--porcelain'],
                stderr=subprocess.DEVNULL
            ).decode('utf-8').strip() != ''
            
            self.metadata['git'] = {
                'commit': git_hash,
                'branch': git_branch,
                'dirty': git_dirty
            }
        except Exception:
            self.metadata['git'] = None
    
    def start(self):
        """mark experiment as started."""
        self.metadata['status'] = 'running'
        self.metadata['start_time'] = datetime.now().isoformat()
        
        # save initial state
        self.config.save(self.output_dir / "config.json")
        self._save_metadata()
    
    def log_metrics(
        self,
        metrics: Dict[str, Any],
        step: Optional[int] = None,
        epoch: Optional[int] = None,
        phase: str = "train"
    ):
        """log metrics."""
        self.metrics.log(metrics, step=step, epoch=epoch, phase=phase)
    
    def log_artifact(self, name: str, data: Any, artifact_type: str = "json"):
        """
        save an artifact.
        
        args:
            name: artifact name
            data: artifact data
            artifact_type: 'json', 'numpy', or 'text'
        """
        artifacts_dir = self.output_dir / "artifacts"
        artifacts_dir.mkdir(exist_ok=True)
        
        if artifact_type == "json":
            path = artifacts_dir / f"{name}.json"
            with open(path, 'w') as f:
                json.dump(data, f, indent=2, default=str)
        
        elif artifact_type == "numpy":
            path = artifacts_dir / f"{name}.npy"
            np.save(path, data)
        
        elif artifact_type == "text":
            path = artifacts_dir / f"{name}.txt"
            with open(path, 'w') as f:
                f.write(str(data))
    
    def save_checkpoint(self, model, name: str = "checkpoint"):
        """save model checkpoint."""
        checkpoint_path = self.checkpoints_dir / name
        model.save(checkpoint_path)
    
    def finish(self, status: str = "completed"):
        """
        mark experiment as finished.
        
        args:
            status: 'completed', 'failed', or 'interrupted'
        """
        self.metadata['status'] = status
        self.metadata['end_time'] = datetime.now().isoformat()
        
        if 'start_time' in self.metadata:
            start = datetime.fromisoformat(self.metadata['start_time'])
            end = datetime.fromisoformat(self.metadata['end_time'])
            self.metadata['duration_seconds'] = (end - start).total_seconds()
        
        # save final state
        self.metrics.save(self.output_dir / "metrics.json")
        self._save_metadata()
        
        # generate summary
        self._generate_summary()
    
    def _save_metadata(self):
        """save metadata to json."""
        with open(self.output_dir / "metadata.json", 'w') as f:
            json.dump(self.metadata, f, indent=2, default=str)
    
    def _generate_summary(self):
        """generate experiment summary."""
        summary = {
            'experiment_name': self.experiment_name,
            'timestamp': self.timestamp,
            'status': self.metadata['status'],
            'config': self.config.to_dict(),
            'metrics_summary': self.metrics.summary(),
            'best_metrics': {}
        }
        
        # find best metrics
        for metric in ['accuracy', 'f1', 'auc']:
            best = self.metrics.get_best(metric, phase='val', mode='max')
            if best:
                summary['best_metrics'][metric] = best.get(metric)
        
        with open(self.output_dir / "summary.json", 'w') as f:
            json.dump(summary, f, indent=2, default=str)


def create_experiment_id(config: ExperimentConfig) -> str:
    """
    create unique experiment id from configuration.
    
    args:
        config: experiment configuration
    
    returns:
        unique id string
    """
    config_str = json.dumps(config.to_dict(), sort_keys=True)
    hash_obj = hashlib.md5(config_str.encode())
    
    return hash_obj.hexdigest()[:8]


def load_experiment(experiment_dir: Union[str, Path]) -> Dict:
    """
    load experiment results from directory.
    
    args:
        experiment_dir: path to experiment directory
    
    returns:
        dictionary with config, metrics, and metadata
    """
    experiment_dir = Path(experiment_dir)
    
    result = {}
    
    config_path = experiment_dir / "config.json"
    if config_path.exists():
        result['config'] = ExperimentConfig.load(config_path)
    
    metrics_path = experiment_dir / "metrics.json"
    if metrics_path.exists():
        with open(metrics_path, 'r') as f:
            result['metrics'] = json.load(f)
    
    metadata_path = experiment_dir / "metadata.json"
    if metadata_path.exists():
        with open(metadata_path, 'r') as f:
            result['metadata'] = json.load(f)
    
    summary_path = experiment_dir / "summary.json"
    if summary_path.exists():
        with open(summary_path, 'r') as f:
            result['summary'] = json.load(f)
    
    return result


def list_experiments(experiments_dir: Union[str, Path]) -> List[Dict]:
    """
    list all experiments in directory.
    
    args:
        experiments_dir: path to experiments directory
    
    returns:
        list of experiment info dictionaries
    """
    experiments_dir = Path(experiments_dir)
    
    experiments = []
    
    for exp_dir in experiments_dir.iterdir():
        if not exp_dir.is_dir():
            continue
        
        summary_path = exp_dir / "summary.json"
        if summary_path.exists():
            with open(summary_path, 'r') as f:
                summary = json.load(f)
                summary['path'] = str(exp_dir)
                experiments.append(summary)
    
    # sort by timestamp
    experiments.sort(key=lambda x: x.get('timestamp', ''), reverse=True)
    
    return experiments
