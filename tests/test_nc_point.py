"""Test cases for NetCDFPointExtractor."""

from pathlib import Path
import tempfile
import shutil

import pytest
import pandas as pd
import numpy as np

from netcdfkit.point_extractor import NetCDFPointExtractor


@pytest.fixture
def temp_cache_dir():
    """Create a temporary cache directory for testing."""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir)


@pytest.fixture
def sample_points_df():
    """Create a sample points DataFrame for testing."""
    points_data = {
        "lat": [50.0, 50.1, 51.0, 51.1],
        "lon": [4.0, 4.1, 5.0, 5.1],
        "ID": ["P1", "P2", "P3", "P4"]
    }
    return pd.DataFrame(points_data)


@pytest.fixture
def extractor(temp_cache_dir):
    """Create a NetCDFPointExtractor instance with temporary cache."""
    return NetCDFPointExtractor(cache_dir=temp_cache_dir)


def test_init(temp_cache_dir):
    """Test initialization of NetCDFPointExtractor."""
    extractor = NetCDFPointExtractor(cache_dir=temp_cache_dir)
    assert extractor is not None
    assert extractor.cache_dir == Path(temp_cache_dir)
    assert extractor.metadata_dir.exists()
    assert extractor.timeseries_dir.exists()
    assert extractor.points_df is None
    assert extractor.spatial_chunks is None
    assert extractor.dataset_info is None


def test_points_dataframe(extractor, sample_points_df):
    """Test setting and accessing points dataframe."""
    extractor.points_df = sample_points_df
    assert extractor.points_df is not None
    assert len(extractor.points_df) == 4
    assert all(col in extractor.points_df.columns for col in ["lat", "lon", "ID"])
    pd.testing.assert_frame_equal(extractor.points_df, sample_points_df)


def test_coordinate_validation(extractor):
    """Test coordinate validation in points dataframe."""
    invalid_points = pd.DataFrame({
        "lat": [91.0, -91.0, 45.0],  # Invalid latitudes
        "lon": [180.0, 181.0, -181.0],  # Invalid longitude
        "ID": ["P1", "P2", "P3"]
    })
    
    with pytest.raises(ValueError, match="Latitude values must be between -90 and 90"):
        extractor.points_df = invalid_points
