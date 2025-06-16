"""Test cases for NetCDFPolygonExtractor."""

from pathlib import Path

import geopandas as gpd
from shapely.geometry import Polygon

from netcdfkit.polygon_extractor import NetCDFPolygonExtractor


def test_init():
    """Test initialization of NetCDFPolygonExtractor."""
    cache_dir = "test_polygon_cache"
    extractor = NetCDFPolygonExtractor(cache_dir=cache_dir)
    assert extractor is not None
    assert extractor.cache_dir == Path(cache_dir)
    assert extractor.metadata_dir.exists()
    assert extractor.statistics_dir.exists()


def test_simple_polygon():
    """Test with a simple polygon."""
    extractor = NetCDFPolygonExtractor()

    # Create a simple square polygon
    polygon = Polygon([
        (4.0, 50.0),
        (4.0, 51.0),
        (5.0, 51.0),
        (5.0, 50.0),
        (4.0, 50.0)
    ])

    # Create a GeoDataFrame with one polygon
    polygons_data = {
        "geometry": [polygon],
        "NUTS_ID": ["TEST001"]
    }
    polygons_gdf = gpd.GeoDataFrame(polygons_data, crs="EPSG:4326")

    # Set the polygons (this would normally be called internally)
    extractor.polygons_gdf = polygons_gdf
    assert len(extractor.polygons_gdf) == 1
    assert extractor.polygons_gdf.crs.to_string() == "EPSG:4326"
