"""
clinical voice feature extraction using parselmouth (praat).

implements extraction of established clinical biomarkers for parkinson's
disease voice analysis including jitter, shimmer, hnr, and formants.
"""

from pathlib import Path
from typing import Dict, Optional, Union, List
import warnings

import numpy as np
import parselmouth
from parselmouth.praat import call


class ClinicalFeatureExtractor:
    """
    extract clinical voice features from audio using praat algorithms.

    implements standardized clinical measurements used in parkinson's disease
    voice assessment including perturbation measures (jitter, shimmer),
    noise measures (hnr), and spectral characteristics (formants).
    """

    def __init__(
        self,
        f0_min: float = 75.0,
        f0_max: float = 600.0,
        time_step: float = 0.0,
        silence_threshold: float = 0.03,
        voicing_threshold: float = 0.45
    ):
        """
        args:
            f0_min: minimum pitch in hz (typical: 75 for male, 100 for female)
            f0_max: maximum pitch in hz (typical: 600)
            time_step: time step for analysis (0 = auto)
            silence_threshold: intensity threshold for silence
            voicing_threshold: voicing threshold (0.45 = praat default)
        """
        self.f0_min = f0_min
        self.f0_max = f0_max
        self.time_step = time_step
        self.silence_threshold = silence_threshold
        self.voicing_threshold = voicing_threshold

    def extract(
        self,
        audio_path: Union[str, Path]
    ) -> Dict[str, float]:
        """
        extract all clinical features from audio file.

        args:
            audio_path: path to audio file

        returns:
            dictionary of clinical features with keys:
            - pitch features: f0_mean, f0_std, f0_min, f0_max, f0_median, f0_range
            - jitter measures: jitter_local, jitter_rap, jitter_ppq5, jitter_ddp
            - shimmer measures: shimmer_local, shimmer_apq3, shimmer_apq5, shimmer_apq11, shimmer_dda
            - noise measures: hnr_mean, hnr_std
            - formants: f1_mean, f1_std, f2_mean, f2_std, f3_mean, f3_std, f4_mean, f4_std
            - duration: total_duration, voiced_duration, unvoiced_duration, voicing_fraction
        """
        try:
            sound = parselmouth.Sound(str(audio_path))
        except Exception as e:
            raise ValueError(f"failed to load audio from {audio_path}: {e}")

        features = {}

        features.update(self._extract_pitch_features(sound))
        features.update(self._extract_jitter_features(sound))
        features.update(self._extract_shimmer_features(sound))
        features.update(self._extract_hnr_features(sound))
        features.update(self._extract_formant_features(sound))
        features.update(self._extract_duration_features(sound))

        features = self._handle_nan_values(features)

        return features

    def extract_from_array(
        self,
        audio: np.ndarray,
        sample_rate: int
    ) -> Dict[str, float]:
        """
        extract all clinical features from raw audio array.

        args:
            audio: numpy array of audio samples
            sample_rate: sampling rate in hz

        returns:
            dictionary of clinical features (same as extract method)
        """
        try:
            # create parselmouth sound object from numpy array
            sound = parselmouth.Sound(audio, sampling_frequency=sample_rate)
        except Exception as e:
            raise ValueError(f"failed to create sound from audio array: {e}")

        features = {}

        features.update(self._extract_pitch_features(sound))
        features.update(self._extract_jitter_features(sound))
        features.update(self._extract_shimmer_features(sound))
        features.update(self._extract_hnr_features(sound))
        features.update(self._extract_formant_features(sound))
        features.update(self._extract_duration_features(sound))

        features = self._handle_nan_values(features)

        return features

    def _extract_pitch_features(self, sound: parselmouth.Sound) -> Dict[str, float]:
        """extract fundamental frequency (f0) statistics."""
        try:
            pitch = call(sound, "To Pitch", self.time_step, self.f0_min, self.f0_max)

            f0_mean = call(pitch, "Get mean", 0, 0, "Hertz")
            f0_std = call(pitch, "Get standard deviation", 0, 0, "Hertz")
            f0_min = call(pitch, "Get minimum", 0, 0, "Hertz", "Parabolic")
            f0_max = call(pitch, "Get maximum", 0, 0, "Hertz", "Parabolic")
            f0_median = call(pitch, "Get quantile", 0, 0, 0.5, "Hertz")

            f0_range = f0_max - f0_min if (f0_max > 0 and f0_min > 0) else 0.0

            voiced_frames = call(pitch, "Count voiced frames")
            total_frames = call(pitch, "Get number of frames")
            voicing_fraction = voiced_frames / total_frames if total_frames > 0 else 0.0

            return {
                'f0_mean': f0_mean,
                'f0_std': f0_std,
                'f0_min': f0_min,
                'f0_max': f0_max,
                'f0_median': f0_median,
                'f0_range': f0_range,
                'voicing_fraction': voicing_fraction
            }

        except Exception as e:
            warnings.warn(f"pitch extraction failed: {e}")
            return self._get_default_pitch_features()

    def _extract_jitter_features(self, sound: parselmouth.Sound) -> Dict[str, float]:
        """
        extract jitter (pitch perturbation) measures.

        jitter quantifies cycle-to-cycle variation in fundamental frequency,
        which is elevated in parkinson's disease due to laryngeal dysfunction.
        """
        try:
            point_process = call(
                sound, "To PointProcess (periodic, cc)",
                self.f0_min, self.f0_max
            )

            jitter_local = call(
                point_process, "Get jitter (local)",
                0, 0, 0.0001, 0.02, 1.3
            )

            jitter_rap = call(
                point_process, "Get jitter (rap)",
                0, 0, 0.0001, 0.02, 1.3
            )

            jitter_ppq5 = call(
                point_process, "Get jitter (ppq5)",
                0, 0, 0.0001, 0.02, 1.3
            )

            jitter_ddp = call(
                point_process, "Get jitter (ddp)",
                0, 0, 0.0001, 0.02, 1.3
            )

            return {
                'jitter_local': jitter_local,
                'jitter_rap': jitter_rap,
                'jitter_ppq5': jitter_ppq5,
                'jitter_ddp': jitter_ddp
            }

        except Exception as e:
            warnings.warn(f"jitter extraction failed: {e}")
            return self._get_default_jitter_features()

    def _extract_shimmer_features(self, sound: parselmouth.Sound) -> Dict[str, float]:
        """
        extract shimmer (amplitude perturbation) measures.

        shimmer quantifies cycle-to-cycle variation in amplitude,
        reflecting reduced glottal closure in pd patients.
        """
        try:
            point_process = call(
                sound, "To PointProcess (periodic, cc)",
                self.f0_min, self.f0_max
            )

            shimmer_local = call(
                [sound, point_process], "Get shimmer (local)",
                0, 0, 0.0001, 0.02, 1.3, 1.6
            )

            shimmer_apq3 = call(
                [sound, point_process], "Get shimmer (apq3)",
                0, 0, 0.0001, 0.02, 1.3, 1.6
            )

            shimmer_apq5 = call(
                [sound, point_process], "Get shimmer (apq5)",
                0, 0, 0.0001, 0.02, 1.3, 1.6
            )

            shimmer_apq11 = call(
                [sound, point_process], "Get shimmer (apq11)",
                0, 0, 0.0001, 0.02, 1.3, 1.6
            )

            shimmer_dda = call(
                [sound, point_process], "Get shimmer (dda)",
                0, 0, 0.0001, 0.02, 1.3, 1.6
            )

            return {
                'shimmer_local': shimmer_local,
                'shimmer_apq3': shimmer_apq3,
                'shimmer_apq5': shimmer_apq5,
                'shimmer_apq11': shimmer_apq11,
                'shimmer_dda': shimmer_dda
            }

        except Exception as e:
            warnings.warn(f"shimmer extraction failed: {e}")
            return self._get_default_shimmer_features()

    def _extract_hnr_features(self, sound: parselmouth.Sound) -> Dict[str, float]:
        """
        extract harmonics-to-noise ratio.

        hnr measures voice quality by comparing periodic (harmonic) to
        aperiodic (noise) components. lower hnr indicates breathiness
        and vocal irregularity common in pd.
        """
        try:
            harmonicity = call(
                sound, "To Harmonicity (cc)",
                0.01, self.f0_min, 0.1, 1.0
            )

            hnr_mean = call(harmonicity, "Get mean", 0, 0)
            hnr_std = call(harmonicity, "Get standard deviation", 0, 0)

            return {
                'hnr_mean': hnr_mean,
                'hnr_std': hnr_std
            }

        except Exception as e:
            warnings.warn(f"hnr extraction failed: {e}")
            return {'hnr_mean': np.nan, 'hnr_std': np.nan}

    def _extract_formant_features(self, sound: parselmouth.Sound) -> Dict[str, float]:
        """
        extract formant frequencies.

        formants are resonant frequencies of the vocal tract.
        changes in formant structure may reflect articulatory impairment in pd.
        """
        try:
            formant = call(
                sound, "To Formant (burg)",
                self.time_step, 5, 5500, 0.025, 50
            )

            f1_mean = call(formant, "Get mean", 1, 0, 0, "Hertz")
            f1_std = call(formant, "Get standard deviation", 1, 0, 0, "Hertz")

            f2_mean = call(formant, "Get mean", 2, 0, 0, "Hertz")
            f2_std = call(formant, "Get standard deviation", 2, 0, 0, "Hertz")

            f3_mean = call(formant, "Get mean", 3, 0, 0, "Hertz")
            f3_std = call(formant, "Get standard deviation", 3, 0, 0, "Hertz")

            f4_mean = call(formant, "Get mean", 4, 0, 0, "Hertz")
            f4_std = call(formant, "Get standard deviation", 4, 0, 0, "Hertz")

            return {
                'f1_mean': f1_mean,
                'f1_std': f1_std,
                'f2_mean': f2_mean,
                'f2_std': f2_std,
                'f3_mean': f3_mean,
                'f3_std': f3_std,
                'f4_mean': f4_mean,
                'f4_std': f4_std
            }

        except Exception as e:
            warnings.warn(f"formant extraction failed: {e}")
            return self._get_default_formant_features()

    def _extract_duration_features(self, sound: parselmouth.Sound) -> Dict[str, float]:
        """extract duration and voicing statistics."""
        try:
            total_duration = call(sound, "Get total duration")

            pitch = call(sound, "To Pitch", self.time_step, self.f0_min, self.f0_max)
            voiced_fraction = call(pitch, "Count voiced frames") / call(pitch, "Get number of frames")

            voiced_duration = total_duration * voiced_fraction
            unvoiced_duration = total_duration * (1 - voiced_fraction)

            return {
                'total_duration': total_duration,
                'voiced_duration': voiced_duration,
                'unvoiced_duration': unvoiced_duration
            }

        except Exception as e:
            warnings.warn(f"duration extraction failed: {e}")
            return {
                'total_duration': np.nan,
                'voiced_duration': np.nan,
                'unvoiced_duration': np.nan
            }

    def _handle_nan_values(self, features: Dict[str, float]) -> Dict[str, float]:
        """replace undefined values with nan for consistent handling."""
        for key, value in features.items():
            if value is None:
                features[key] = np.nan
            elif isinstance(value, (int, float)):
                if np.isnan(value) or np.isinf(value):
                    features[key] = np.nan
        return features

    def _get_default_pitch_features(self) -> Dict[str, float]:
        """return default nan values for pitch features."""
        return {
            'f0_mean': np.nan,
            'f0_std': np.nan,
            'f0_min': np.nan,
            'f0_max': np.nan,
            'f0_median': np.nan,
            'f0_range': np.nan,
            'voicing_fraction': np.nan
        }

    def _get_default_jitter_features(self) -> Dict[str, float]:
        """return default nan values for jitter features."""
        return {
            'jitter_local': np.nan,
            'jitter_rap': np.nan,
            'jitter_ppq5': np.nan,
            'jitter_ddp': np.nan
        }

    def _get_default_shimmer_features(self) -> Dict[str, float]:
        """return default nan values for shimmer features."""
        return {
            'shimmer_local': np.nan,
            'shimmer_apq3': np.nan,
            'shimmer_apq5': np.nan,
            'shimmer_apq11': np.nan,
            'shimmer_dda': np.nan
        }

    def _get_default_formant_features(self) -> Dict[str, float]:
        """return default nan values for formant features."""
        return {
            'f1_mean': np.nan,
            'f1_std': np.nan,
            'f2_mean': np.nan,
            'f2_std': np.nan,
            'f3_mean': np.nan,
            'f3_std': np.nan,
            'f4_mean': np.nan,
            'f4_std': np.nan
        }


def extract_clinical_features(
    audio_path: Union[str, Path],
    f0_min: float = 75.0,
    f0_max: float = 600.0
) -> Dict[str, float]:
    """
    convenience function to extract clinical features from audio file.

    args:
        audio_path: path to audio file
        f0_min: minimum pitch in hz
        f0_max: maximum pitch in hz

    returns:
        dictionary of clinical features
    """
    extractor = ClinicalFeatureExtractor(f0_min=f0_min, f0_max=f0_max)
    return extractor.extract(audio_path)


def batch_extract_features(
    audio_paths: List[Union[str, Path]],
    f0_min: float = 75.0,
    f0_max: float = 600.0,
    verbose: bool = True
) -> List[Dict[str, float]]:
    """
    extract clinical features from multiple audio files.

    args:
        audio_paths: list of paths to audio files
        f0_min: minimum pitch in hz
        f0_max: maximum pitch in hz
        verbose: whether to print progress

    returns:
        list of feature dictionaries
    """
    extractor = ClinicalFeatureExtractor(f0_min=f0_min, f0_max=f0_max)

    features_list = []

    for i, path in enumerate(audio_paths):
        if verbose and (i + 1) % 100 == 0:
            print(f"processed {i + 1}/{len(audio_paths)} files")

        try:
            features = extractor.extract(path)
            features['file_path'] = str(path)
            features_list.append(features)
        except Exception as e:
            warnings.warn(f"failed to extract features from {path}: {e}")
            features_list.append(None)

    return features_list


def get_clinical_feature_names() -> List[str]:
    """
    get list of all clinical feature names.

    returns:
        list of feature names
    """
    return [
        'f0_mean', 'f0_std', 'f0_min', 'f0_max', 'f0_median', 'f0_range',
        'voicing_fraction',
        'jitter_local', 'jitter_rap', 'jitter_ppq5', 'jitter_ddp',
        'shimmer_local', 'shimmer_apq3', 'shimmer_apq5', 'shimmer_apq11',
        'shimmer_dda',
        'hnr_mean', 'hnr_std',
        'f1_mean', 'f1_std', 'f2_mean', 'f2_std', 'f3_mean', 'f3_std',
        'f4_mean', 'f4_std',
        'total_duration', 'voiced_duration', 'unvoiced_duration'
    ]


def get_pd_discriminative_features() -> List[str]:
    """
    get subset of features most discriminative for parkinson's disease.

    based on clinical literature, these features show strongest differences
    between pd and healthy control groups.

    returns:
        list of discriminative feature names
    """
    return [
        'jitter_local',
        'jitter_rap',
        'jitter_ppq5',
        'shimmer_local',
        'shimmer_apq5',
        'hnr_mean',
        'f0_std',
        'voicing_fraction'
    ]


def create_binary_clinical_labels(
    features: Dict[str, float],
    thresholds: Optional[Dict[str, float]] = None
) -> Dict[str, int]:
    """
    create binary labels for clinical features based on thresholds.

    useful for probing experiments to test if model encodes clinical features.

    args:
        features: dictionary of continuous clinical features
        thresholds: custom thresholds. if none, uses clinical cutoffs

    returns:
        dictionary of binary labels (0=normal, 1=abnormal)
    """
    if thresholds is None:
        thresholds = {
            'jitter_local': 0.01,
            'jitter_rap': 0.005,
            'jitter_ppq5': 0.005,
            'shimmer_local': 0.035,
            'shimmer_apq5': 0.03,
            'hnr_mean': 15.0,
            'f0_std': 5.0,
        }

    binary_labels = {}

    for feature_name in ['jitter_local', 'jitter_rap', 'jitter_ppq5',
                         'shimmer_local', 'shimmer_apq5']:
        if feature_name in features and feature_name in thresholds:
            value = features[feature_name]
            if not np.isnan(value):
                binary_labels[f'{feature_name}_binary'] = int(value > thresholds[feature_name])

    if 'hnr_mean' in features and 'hnr_mean' in thresholds:
        value = features['hnr_mean']
        if not np.isnan(value):
            binary_labels['hnr_binary'] = int(value < thresholds['hnr_mean'])

    if 'f0_std' in features and 'f0_std' in thresholds:
        value = features['f0_std']
        if not np.isnan(value):
            binary_labels['f0_variability_binary'] = int(value > thresholds['f0_std'])

    return binary_labels


def compute_clinical_alignment_score(
    model_features: np.ndarray,
    clinical_features: np.ndarray,
    feature_names: List[str]
) -> float:
    """
    compute correlation between model learned features and clinical features.

    measures how well model representations align with established clinical
    biomarkers. higher scores indicate more clinically interpretable models.

    args:
        model_features: model representations [n_samples, n_dims]
        clinical_features: clinical feature matrix [n_samples, n_features]
        feature_names: names of clinical features

    returns:
        alignment score (mean correlation across features)
    """
    from sklearn.linear_model import Ridge
    from sklearn.metrics import r2_score
    from sklearn.model_selection import cross_val_score

    scores = []

    for i, feat_name in enumerate(feature_names):
        feat_values = clinical_features[:, i]

        valid_indices = ~np.isnan(feat_values)

        if np.sum(valid_indices) < 10:
            continue

        X = model_features[valid_indices]
        y = feat_values[valid_indices]

        if np.std(y) < 1e-6:
            continue

        model = Ridge(alpha=1.0)
        cv_scores = cross_val_score(model, X, y, cv=5, scoring='r2')

        if cv_scores.mean() > 0:
            scores.append(cv_scores.mean())

    if len(scores) == 0:
        return 0.0

    return np.mean(scores)
