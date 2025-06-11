# NetCDFKit API Reference

This document provides detailed API documentation for all classes and methods in NetCDFKit.

## Table of Contents

1. [NetCDFPointExtractor](#netcdfpointextractor)
2. [NetCDFPolygonExtractor](#netcdfpolygonextractor)
3. [Common Parameters](#common-parameters)
4. [Error Handling](#error-handling)
5. [Performance Tips](#performance-tips)

## NetCDFPointExtractor

### Class Overview

The `NetCDFPointExtractor` class extracts time series data from NetCDF files at specific geographic coordinates.

```python
from netcdfkit import NetCDFPointExtractor

extractor = NetCDFPointExtractor(cache_dir="my_cache")
```

### Methods

#### `__init__(cache_dir: str | Path = "timeseries_cache")`

Initialize the point extractor.

**Parameters:**
- `cache_dir`: Directory for caching extracted data

#### `extract_and_cache_timeseries(...)`

Extract and cache time series for all points.

**Parameters:**
- `netcdf_path`: Path to NetCDF file
- `points_path`: Path to CSV file with coordinates  
- `variable`: NetCDF variable name
- `date_col`: Date column name (optional)
- `eps_km`: Clustering radius in km (optional)
- `min_samples`: Minimum points per cluster (optional)
- `force_recache`: Force re-extraction (default: False)

**Returns:** Cache ID string

**Example:**
```python
cache_id = extractor.extract_and_cache_timeseries(
    netcdf_path="temperature.nc",
    points_path="stations.csv",
    variable="t2m",
    date_col="date"
)
```

#### `generate_multi_scenario_results(...)`

Generate results for multiple temporal scenarios.

**Parameters:**
- `cache_id`: Cache ID from extraction
- `days_back_list`: List of averaging windows
- `date_col`: Reference date column (optional)
- `output_path`: Output CSV path (optional)

**Returns:** DataFrame with scenario results

**Example:**
```python
results = extractor.generate_multi_scenario_results(
    cache_id=cache_id,
    days_back_list=[7, 30, 90],
    output_path="scenarios.csv"
)
```

#### `load_point_timeseries(...)`

Load cached time series for specific points.

**Parameters:**
- `cache_id`: Cache ID
- `point_ids`: List of point indices or "all"

**Returns:** Dictionary of {point_id: pandas.Series}

**Example:**
```python
timeseries = extractor.load_point_timeseries(
    cache_id=cache_id,
    point_ids=[0, 1, 2]
)
```

#### `export_point_timeseries_csv(...)`

Export time series to CSV format.

**Parameters:**
- `cache_id`: Cache ID
- `point_ids`: List of point indices or "all"
- `output_path`: Output CSV path

**Returns:** DataFrame of exported data

#### `list_cached_extractions()`

List all cached extractions.

**Returns:** DataFrame with cache information

---

## NetCDFPolygonExtractor

### Class Overview

The `NetCDFPolygonExtractor` class calculates statistics within polygon regions.

```python
from netcdfkit import NetCDFPolygonExtractor

extractor = NetCDFPolygonExtractor(cache_dir="polygon_cache")
```

### Methods

#### `__init__(cache_dir: str | Path = "polygon_cache")`

Initialize the polygon extractor.

#### `extract_and_cache_statistics(...)`

Extract and cache statistics for all polygons.

**Parameters:**
- `netcdf_path`: Path to NetCDF file
- `shapefile_path`: Path to shapefile
- `variable`: NetCDF variable name
- `id_column`: Unique ID column in shapefile
- `statistics`: List of statistics to calculate
- `force_recache`: Force re-extraction (default: False)

**Available statistics:**
- `"mean"`: Average value
- `"std"`: Standard deviation
- `"min"`: Minimum value
- `"max"`: Maximum value
- `"median"`: Median value
- `"count"`: Number of valid grid cells

**Example:**
```python
cache_id = extractor.extract_and_cache_statistics(
    netcdf_path="temperature.nc",
    shapefile_path="regions.shp",
    variable="temperature",
    id_column="REGION_ID",
    statistics=["mean", "std", "min", "max"]
)
```

#### `load_polygon_timeseries(...)`

Load time series with filtering options.

**Parameters:**
- `cache_id`: Cache ID
- `polygon_ids`: List of polygon IDs (optional)
- `country_code`: Country code filter (optional)
- `start_date`: Start date filter (optional)
- `end_date`: End date filter (optional)
- `statistic`: Which statistic to load

**Returns:** Dictionary of {polygon_id: pandas.Series}

**Example:**
```python
germany_data = extractor.load_polygon_timeseries(
    cache_id=cache_id,
    country_code="DE",
    statistic="mean"
)
```

#### `export_timeseries_csv(...)`

Export polygon time series to CSV.

**Parameters:**
- `cache_id`: Cache ID
- `output_path`: Output CSV path
- `polygon_ids`: Specific polygon IDs (optional)
- `country_code`: Country code filter (optional)
- `statistic`: Which statistic to export
- `wide_format`: Wide vs long format (default: False)

#### `list_countries(cache_id)`

List available countries in the dataset.

#### `get_polygon_info(...)`

Get detailed polygon metadata.

#### `get_cache_info(cache_id)`

Get cache information and statistics.

---

## Common Parameters

### File Paths
- Use absolute paths when possible
- Forward slashes work on all platforms
- Ensure files exist and are readable

### Coordinate Systems
- Points should be in decimal degrees (WGS84)
- NetCDF files should include CRS information
- Automatic transformation is performed when needed

### Memory Management
- Close other applications during large extractions
- Monitor memory usage with task manager
- Use SSD storage for better performance

---

## Error Handling

### Common Errors

**FileNotFoundError**: File paths are incorrect
```python
# Solution: Check file paths and permissions
import os
assert os.path.exists("your_file.nc"), "File does not exist"
```

**MemoryError**: Insufficient RAM
```python
# Solution: Process smaller batches or increase RAM
# Use force_recache=False to reuse existing cache
```

**KeyError**: Variable not found in NetCDF
```python
# Solution: Check variable names
import xarray as xr
ds = xr.open_dataset("your_file.nc")
print(list(ds.variables))
```

### Best Practices

1. **Test with small datasets first**
2. **Use try-except blocks for robust code**
3. **Monitor memory usage during processing**
4. **Keep NetCDF files on fast storage (SSD)**
5. **Use caching effectively - extract once, analyze many times**

---

## Performance Tips

### Optimization Strategies

1. **Spatial Clustering**: Automatically optimized, but you can tune:
   ```python
   cache_id = extractor.extract_and_cache_timeseries(
       eps_km=200,      # Larger for sparse global data
       min_samples=2    # Smaller for sparse data
   )
   ```

2. **Memory Management**: 
   - Use 16-32GB RAM for large datasets
   - Close unnecessary applications
   - Process during off-peak hours

3. **Storage Optimization**:
   - Store NetCDF files on SSD
   - Use fast cache directory
   - Compress output when possible

4. **Workflow Optimization**:
   - Extract once, analyze multiple times
   - Use appropriate chunk sizes
   - Leverage parallel processing when available

### Performance Expectations

| Dataset | Extraction Time | Memory Usage | Storage |
|---------|----------------|--------------|---------|
| 250 points, 50GB | 5-15 min | 3-8 GB | 0.5-2 GB |
| 1000 points, 100GB | 15-30 min | 4-10 GB | 1-4 GB |
| 5000 points, 200GB | 30-60 min | 8-20 GB | 3-8 GB |

---

For more examples and tutorials, see the `examples/` directory.
