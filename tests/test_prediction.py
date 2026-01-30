"""Comprehensive tests for prediction module"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
from pysegqc.prediction import predict_new_cases


def test_predict_new_cases_basic(synthetic_radiomics_excel, trained_models_dir):
    """Test basic prediction on new cases"""
    # Run prediction
    predict_new_cases(
        synthetic_radiomics_excel,
        trained_models_dir,
        sheet_name='PCA_Data',
        mode='position',
        position=1
    )

    # Check prediction output exists
    predictions_dir = trained_models_dir / 'synthetic_radiomics_predictions'
    assert predictions_dir.exists()

    predictions_file = predictions_dir / 'synthetic_radiomics_predictions.xlsx'
    assert predictions_file.exists()

    # Check predictions structure
    df = pd.read_excel(predictions_file, sheet_name='Predictions')
    assert 'Predicted_Cluster' in df.columns
    assert 'Confidence_Score' in df.columns
    assert len(df) == 50  # All 50 synthetic samples


def test_predict_feature_validation(trained_models_dir, tmp_path):
    """Test prediction fails gracefully with missing features"""
    # Create Excel with wrong features
    wrong_df = pd.DataFrame({
        'MRN': ['MRN0001', 'MRN0002'],
        '001_wrong_feature': [1.0, 2.0],
        '001_another_wrong': [3.0, 4.0]
    })
    wrong_file = tmp_path / 'wrong_features.xlsx'
    wrong_df.to_excel(wrong_file, sheet_name='PCA_Data', index=False)

    # Should exit with error code 1
    with pytest.raises(SystemExit) as exc_info:
        predict_new_cases(wrong_file, trained_models_dir, 'PCA_Data', 'position', 1)

    assert exc_info.value.code == 1


def test_predict_missing_model_file(synthetic_radiomics_excel, tmp_path):
    """Test prediction fails when model file doesn't exist"""
    empty_dir = tmp_path / 'no_models'
    empty_dir.mkdir()

    with pytest.raises(SystemExit) as exc_info:
        predict_new_cases(synthetic_radiomics_excel, empty_dir, 'PCA_Data', 'position', 1)

    assert exc_info.value.code == 1


def test_predict_confidence_scores(synthetic_radiomics_excel, trained_models_dir):
    """Test that confidence scores are calculated correctly"""
    predict_new_cases(
        synthetic_radiomics_excel,
        trained_models_dir,
        sheet_name='PCA_Data',
        mode='position',
        position=1
    )

    predictions_file = trained_models_dir / 'synthetic_radiomics_predictions' / 'synthetic_radiomics_predictions.xlsx'
    df = pd.read_excel(predictions_file, sheet_name='Predictions')

    # Confidence scores should be between 0 and 1
    assert all(0 <= score <= 1 for score in df['Confidence_Score'])

    # Should have predictions for all samples
    assert not df['Predicted_Cluster'].isnull().any()


def test_predict_output_format(synthetic_radiomics_excel, trained_models_dir):
    """Test prediction output Excel format"""
    predict_new_cases(
        synthetic_radiomics_excel,
        trained_models_dir,
        sheet_name='PCA_Data',
        mode='position',
        position=1
    )

    predictions_file = trained_models_dir / 'synthetic_radiomics_predictions' / 'synthetic_radiomics_predictions.xlsx'

    # Check sheets exist (use context manager to avoid ResourceWarning)
    with pd.ExcelFile(predictions_file) as xl:
        assert 'Predictions' in xl.sheet_names

        # Should have per-cluster sheets
        df = pd.read_excel(predictions_file, sheet_name='Predictions')
        unique_clusters = df['Predicted_Cluster'].unique()
        for cluster_id in unique_clusters:
            assert f'Cluster_{cluster_id}' in xl.sheet_names


def test_predict_with_concat_mode(synthetic_radiomics_excel, tmp_path):
    """Test prediction in concat mode"""
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler
    from sklearn.cluster import AgglomerativeClustering
    import joblib

    # Create trained models for concat mode (preserve DataFrames for feature names)
    df = pd.read_excel(synthetic_radiomics_excel, sheet_name='PCA_Data')
    feature_cols = [c for c in df.columns if c.startswith(('001_', '002_'))]
    X_df = df[feature_cols].fillna(df[feature_cols].median())

    scaler = StandardScaler()
    X_scaled_array = scaler.fit_transform(X_df)
    X_scaled_df = pd.DataFrame(X_scaled_array, columns=X_df.columns, index=X_df.index)

    pca_model = PCA(n_components=5)
    X_pca = pca_model.fit_transform(X_scaled_df)

    clustering_model = AgglomerativeClustering(n_clusters=3)
    labels = clustering_model.fit_predict(X_pca)

    # Calculate centroids
    centroids = np.array([X_pca[labels == i].mean(axis=0) for i in range(3)])

    models_dict = {
        'pca_model': pca_model,
        'pca': pca_model,
        'scaler': scaler,
        'centroids': centroids,
        'feature_names': feature_cols,
        'n_clusters': 3,
        'impute_strategy': 'median',
        'n_components': 5
    }

    models_path = tmp_path / 'trained_models.pkl'
    joblib.dump(models_dict, models_path)

    # Run prediction in concat mode
    predict_new_cases(
        synthetic_radiomics_excel,
        tmp_path,
        sheet_name='PCA_Data',
        mode='concat',
        position=None
    )

    predictions_file = tmp_path / 'synthetic_radiomics_predictions' / 'synthetic_radiomics_predictions.xlsx'
    assert predictions_file.exists()


def test_predict_preserves_metadata(synthetic_radiomics_excel, trained_models_dir):
    """Test that metadata columns are preserved in predictions"""
    predict_new_cases(
        synthetic_radiomics_excel,
        trained_models_dir,
        sheet_name='PCA_Data',
        mode='position',
        position=1
    )

    predictions_file = trained_models_dir / 'synthetic_radiomics_predictions' / 'synthetic_radiomics_predictions.xlsx'
    df = pd.read_excel(predictions_file, sheet_name='Predictions')

    # Check metadata columns preserved
    assert 'MRN' in df.columns
    assert 'Plan_ID' in df.columns
    assert 'Session_ID' in df.columns
