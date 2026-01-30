import pytest
import pandas as pd
from pysegqc.utils import extract_case_metadata, print_banner


def test_extract_case_metadata_full():
    """Test metadata extraction with all columns present."""
    metadata_df = pd.DataFrame({
        'MRN': ['12345', '67890'],
        'Plan_ID': ['PLAN001', 'PLAN002'],
        'Session_ID': ['S1', 'S2'],
        'View_Scan_URL': ['http://ex.com/1', 'http://ex.com/2']
    }, index=[0, 1])

    result = extract_case_metadata(0, metadata_df)

    assert result['Case_ID'] == 0
    assert result['MRN'] == '12345'
    assert result['Plan_ID'] == 'PLAN001'
    assert result['Session_ID'] == 'S1'
    assert result['View_Scan_URL'] == 'http://ex.com/1'


def test_extract_case_metadata_partial():
    """Test metadata extraction with missing columns."""
    metadata_df = pd.DataFrame({
        'MRN': ['12345'],
        'Other_Column': ['data']
    }, index=[0])

    result = extract_case_metadata(0, metadata_df)

    assert result['Case_ID'] == 0
    assert result['MRN'] == '12345'
    assert 'Plan_ID' not in result
    assert 'Other_Column' not in result


def test_extract_case_metadata_none():
    """Test metadata extraction with None DataFrame."""
    result = extract_case_metadata(5, None)

    assert result == {'Case_ID': 5}


def test_extract_case_metadata_missing_index():
    """Test metadata extraction with missing case index."""
    metadata_df = pd.DataFrame({
        'MRN': ['12345']
    }, index=[0])

    result = extract_case_metadata(999, metadata_df)

    assert result == {'Case_ID': 999}


def test_print_banner_default(capsys):
    """Test banner printing with default width."""
    print_banner("Test Message")

    captured = capsys.readouterr()
    assert "="*70 in captured.out
    assert "Test Message" in captured.out


def test_print_banner_custom_width(capsys):
    """Test banner printing with custom width."""
    print_banner("Short", width=20)

    captured = capsys.readouterr()
    assert "="*20 in captured.out
    assert "Short" in captured.out
