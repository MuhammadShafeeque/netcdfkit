# NetCDFKit - Optimal NetCDF Point Extraction System

[![PyPI](https://img.shields.io/pypi/v/netcdfkit.svg)](https://pypi.org/project/netcdfkit/)
[![Python Version](https://img.shields.io/pypi/pyversions/netcdfkit.svg)](https://pypi.org/project/netcdfkit/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A high-performance Python library for efficient extraction of time series data from large NetCDF files at specific geographic points. Designed for optimal memory usage and speed when working with massive climate/environmental datasets.

## 🚀 Key Features

- **Spatial Clustering**: Automatic detection of spatial clusters for efficient chunk processing
- **Intelligent Caching**: Time series caching with metadata preservation for instant reuse
- **Multi-Scenario Processing**: Fast generation of multiple temporal scenarios (different averaging windows)
- **Memory Efficient**: Handles large datasets (20,000+ points, 200GB+ NetCDF files) within 32GB RAM
- **Easy Integration**: Simple API for complex spatiotemporal analysis workflows
- **Performance Optimized**: 100x+ speed improvement over naive point-by-point extraction

## 📦 Installation

```bash
pip install netcdfkit
```

## 🏁 Quick Start

```python
from netcdfkit import NetCDFPointExtractor

# Initialize extractor
extractor = NetCDFPointExtractor(cache_dir="my_cache")

# One-time extraction and caching (5-15 minutes for ~250 points)
cache_id = extractor.extract_and_cache_timeseries(
    netcdf_path="temperature_data.nc",
    points_path="monitoring_stations.csv",  # CSV with 'lon', 'lat' columns
    variable="temperature",
    date_col="measurement_date"
)

# Fast multi-scenario analysis (10-30 seconds)
results_df = extractor.generate_multi_scenario_results(
    cache_id=cache_id,
    days_back_list=[3, 7, 14, 30],  # Multiple averaging windows
    date_col="measurement_date",
    output_path="multi_scenario_results.csv"
)

print(f"Generated results with {len(results_df)} points")
print(f"Columns: {list(results_df.columns)}")
```

## 📊 Input Data Format

Your CSV file should contain at minimum:

```csv
lon,lat,station_id,measurement_date
8.6821,50.1109,STATION_001,2023-06-15
13.4050,52.5200,STATION_002,2023-06-15
9.9937,53.5511,STATION_003,2023-06-15
```

## 🔧 Core Functionality

### 1. Extract and Cache Time Series

```python
# Extract time series for all points (one-time operation)
cache_id = extractor.extract_and_cache_timeseries(
    netcdf_path="era5_temperature.nc",
    points_path="weather_stations.csv",
    variable="t2m",  # 2-meter temperature
    date_col="observation_date",
    force_recache=False  # Use cache if exists
)
```

### 2. Multi-Scenario Analysis

```python
# Generate multiple averaging scenarios in one operation
results = extractor.generate_multi_scenario_results(
    cache_id=cache_id,
    days_back_list=[1, 3, 7, 14, 30, 90],  # Different averaging windows
    date_col="observation_date",
    output_path="temperature_averages.csv"
)
```

### 3. Access Individual Time Series

```python
# Load specific point time series
point_timeseries = extractor.load_point_timeseries(
    cache_id=cache_id,
    point_ids=[0, 5, 10]  # Specific points, or "all" for all points
)

# Access individual time series (returns pandas Series)
ts_point_0 = point_timeseries[0]
print(f"Time range: {ts_point_0.index.min()} to {ts_point_0.index.max()}")
```

### 4. Export Time Series

```python
# Export time series to CSV
export_df = extractor.export_point_timeseries_csv(
    cache_id=cache_id,
    point_ids=[0, 1, 2],  # Or "all"
    output_path="timeseries_export.csv"
)
```

### 5. Cache Management

```python
# List all cached extractions
cached_list = extractor.list_cached_extractions()
print(cached_list)

# View cache information
# Returns DataFrame with: cache_id, n_points, variable, time_range, n_timesteps, n_chunks
```

## 🎯 Complete Workflow Example

```python
from netcdfkit import NetCDFPointExtractor

# Initialize
extractor = NetCDFPointExtractor(cache_dir="analysis_cache")

# Step 1: One-time extraction
cache_id = extractor.extract_and_cache_timeseries(
    netcdf_path="era5_temperature.nc",
    points_path="weather_stations.csv",
    variable="t2m",
    date_col="observation_date"
)

# Step 2: Multi-scenario analysis
results = extractor.generate_multi_scenario_results(
    cache_id=cache_id,
    days_back_list=[1, 3, 7, 14, 30, 90],
    date_col="observation_date",
    output_path="temperature_scenarios.csv"
)

# Step 3: Export specific station data
station_data = extractor.export_point_timeseries_csv(
    cache_id=cache_id,
    point_ids=[42, 43, 44],  # Specific stations
    output_path="selected_stations.csv"
)

print("Analysis complete!")
```

## ⚡ Performance

### Expected Performance (typical desktop/server):

| Dataset Size | First Extraction | Scenario Generation | Memory Usage |
|--------------|------------------|-------------------|--------------|
| 250 points   | 5-15 minutes    | 10-30 seconds     | 3-8 GB       |
| 20,000 points| 30-90 minutes   | 2-5 minutes       | 5-15 GB      |

### After Caching:
- **Any new scenario**: 10 seconds - 2 minutes
- **Memory usage**: 1-3 GB
- **Storage**: Cached data enables instant reanalysis

## 🔬 Advanced Features

### Custom Spatial Clustering

```python
# Override automatic clustering detection
cache_id = extractor.extract_and_cache_timeseries(
    netcdf_path="data.nc",
    points_path="points.csv",
    variable="temperature",
    eps_km=150,      # Cluster radius in km
    min_samples=3    # Minimum points per cluster
)
```

### Seasonal Analysis

```python
# Extract seasonal averages (custom analysis)
def extract_seasonal_averages(cache_id, output_path):
    all_timeseries = extractor.load_point_timeseries(cache_id, "all")
    # Custom seasonal processing...
    return results_df
```

## 🛠️ System Requirements

- **Python**: ≥3.12
- **Memory**: 8-32 GB RAM (depending on dataset size)
- **Storage**: SSD recommended for large NetCDF files
- **Dependencies**: pandas, xarray, numpy, pyproj, scikit-learn, pyarrow, tqdm, dask

## 📁 Directory Structure

The system creates organized cache directories:

```
cache_dir/
├── metadata/
│   ├── extract_[dataset]_[hash].json      # Extraction metadata
│   └── extract_[dataset]_[hash]_points.csv # Point information
└── timeseries/
    └── extract_[dataset]_[hash]_timeseries.parquet # Time series data
```

## 🎯 Use Cases

- **Climate Data Analysis**: Extract temperature/precipitation time series for weather stations
- **Air Quality Monitoring**: Get pollution measurements for sensor locations
- **Environmental Research**: Analyze satellite data at field measurement sites
- **Oceanographic Studies**: Extract sea surface temperature at buoy locations
- **Agricultural Monitoring**: Get weather/climate data for farm locations

## 🔧 Troubleshooting

| Issue | Solution |
|-------|----------|
| Memory errors | Close other applications, restart kernel |
| Slow extraction | Ensure NetCDF on SSD, check spatial clustering |
| Missing data | Verify NetCDF variable names and coordinate system |
| Cache issues | Set `force_recache=True` to re-extract |

## 👨‍💻 Creator & Support

**Creator**: Muhammad Shafeeque  
**Institution**: [Data Science Support](https://www.awi.de/en/about-us/service/computing-centre/data-management/data-science-support.html), Computing and Data Center, [Alfred Wegener Institute for Polar and Marine Research](https://www.awi.de/en/)  
**Project Support**: [HealthyPlanet Project](https://www.bremen-research.de/en/datanord/research-academy/healthy-planet) under [DataNord](https://www.bremen-research.de/datanord)  
**GitHub**: [MuhammadShafeeque](https://github.com/MuhammadShafeeque)

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📚 Documentation

For more detailed examples and advanced usage patterns, see:
- `examples/quick_usage.py` - Basic usage patterns
- `examples/usageExamples.py` - Complete workflow examples
- `quick-start-guide.md` - Detailed API documentation
- `Usage-guide.md` - Step-by-step setup guide

---

*Built with ❤️ for the climate and environmental science community*