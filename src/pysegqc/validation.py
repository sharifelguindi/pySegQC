"""
Data validation, imputation, and cleaning for radiomics clustering analysis.

This module handles missing value imputation, feature standardization, and outlier detection.
"""

import logging
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest

logger = logging.getLogger(__name__)


def impute_missing_values(features_df, strategy='median'):
    """Impute missing values using specified strategy."""
    if features_df.isnull().any().any():
        print(f"\n{'='*70}")
        print(f"Imputing missing values using '{strategy}' strategy")
        print(f"{'='*70}")

        imputer = SimpleImputer(strategy=strategy)
        imputed_data = imputer.fit_transform(features_df)

        imputed_df = pd.DataFrame(imputed_data,
                                   columns=features_df.columns,
                                   index=features_df.index)
        print("✓ Missing values imputed")
        return imputed_df
    else:
        print("\n✓ No imputation needed - no missing values")
        return features_df


def standardize_features(features_df):
    """Standardize features to mean=0, std=1 (critical for PCA)."""
    print(f"\n{'='*70}")
    print("Standardizing features (mean=0, std=1)")
    print(f"{'='*70}")

    scaler = StandardScaler()
    standardized_data = scaler.fit_transform(features_df)

    standardized_df = pd.DataFrame(standardized_data,
                                    columns=features_df.columns,
                                    index=features_df.index)

    print("✓ Features standardized")
    return standardized_df, scaler


def detect_outliers(pca_data, contamination=0.1):
    """
    Detect outliers using Isolation Forest in PCA space.

    Args:
        pca_data: PCA-transformed data
        contamination: Expected proportion of outliers (default: 0.1)

    Returns:
        dict: {
            'outlier_mask': boolean array (True = outlier),
            'outlier_scores': anomaly scores for each sample,
            'outlier_indices': indices of detected outliers
        }
    """
    logger.info(f"Detecting outliers (contamination={contamination})...")

    iso_forest = IsolationForest(contamination=contamination, random_state=42)
    outlier_labels = iso_forest.fit_predict(pca_data)
    outlier_scores = iso_forest.score_samples(pca_data)

    outlier_mask = outlier_labels == -1
    outlier_indices = np.where(outlier_mask)[0]

    logger.info(f"Detected {outlier_mask.sum()} outliers ({outlier_mask.sum()/len(outlier_mask)*100:.1f}%)")

    return {
        'outlier_mask': outlier_mask,
        'outlier_scores': outlier_scores,
        'outlier_indices': outlier_indices
    }
