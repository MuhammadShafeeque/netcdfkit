# NetCDFKit - High-Performance NetCDF Data Extraction Toolkit

[![PyPI](https://img.shields.io/pypi/v/netcdfkit.svg)](https://pypi.org/project/netcdfkit/)
[![Python Version](https://img.shields.io/pypi/pyversions/netcdfkit.svg)](https://pypi.org/project/netcdfkit/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Downloads](https://pepy.tech/badge/netcdfkit)](https://pepy.tech/project/netcdfkit)

A comprehensive Python library for efficient extraction of time series data from large NetCDF files at specific geographic points and polygon regions. Designed for optimal memory usage and speed when working with massive climate, environmental, and geospatial datasets up to 200GB+.

**NetCDFKit** provides two powerful extraction modules:
- **🎯 Point Extraction** (`NetCDFPointExtractor`): Extract time series at specific geographic coordinates
- **🌍 Polygon Extraction** (`NetCDFPolygonExtractor`): Calculate comprehensive statistics within polygon regions

## 🚀 Key Features

### 🎯 NetCDFPointExtractor
- **🔍 Spatial Clustering**: Automatic detection of spatial clusters for efficient chunk processing
- **💾 Intelligent Caching**: Time series caching with metadata preservation for instant reuse  
- **⚡ Multi-Scenario Processing**: Fast generation of multiple temporal scenarios (different averaging windows)
- **💿 Memory Efficient**: Handles large datasets (20,000+ points, 200GB+ NetCDF files) within 32GB RAM
- **🗺️ Coordinate Transformation**: Automatic handling of different coordinate reference systems
- **📊 Flexible Analysis**: Support for custom temporal windows and statistical analysis

### 🌍 NetCDFPolygonExtractor  
- **📈 Polygon Statistics**: Calculate mean, std, min, max, median, count within polygon regions
- **🏛️ NUTS/Administrative Regions**: Optimized for European NUTS regions and administrative boundaries
- **🌐 Country-Based Chunking**: Efficient processing organized by country/region codes  
- **🔍 Flexible Filtering**: Filter by country codes, polygon IDs, and time ranges
- **📋 Multiple Output Formats**: Export in wide or long format CSV with comprehensive metadata

### 🔧 Common Features
- **⚡ Performance Optimized**: 100x+ speed improvement over naive point-by-point extraction
- **🔗 Easy Integration**: Simple API for complex spatiotemporal analysis workflows
- **🛡️ Robust Error Handling**: Comprehensive error checking and recovery mechanisms
- **📊 Progress Tracking**: Built-in progress bars and detailed logging for long operations
- **🔄 Caching System**: Smart caching prevents re-extraction of same data
- **📁 Organized Storage**: Clean directory structure for cached data and metadata

## 📦 Installation

### From PyPI (Recommended)
```bash
pip install netcdfkit
```

### From Source (Development)
```bash
git clone https://github.com/MuhammadShafeeque/netcdfkit.git
cd netcdfkit
pip install -e .
```

### System Requirements
- **Python**: ≥ 3.12
- **Memory**: 8-32 GB RAM (depending on dataset size)
- **Storage**: SSD recommended for large NetCDF files and cache
- **OS**: Windows, macOS, Linux

## 🏁 Quick Start

### 🎯 Point Extraction (Weather Stations, Monitoring Sites)

```python
from netcdfkit import NetCDFPointExtractor

# Initialize extractor with cache directory
extractor = NetCDFPointExtractor(cache_dir="my_cache")

# One-time extraction and caching (5-15 minutes for ~250 points)
cache_id = extractor.extract_and_cache_timeseries(
    netcdf_path="temperature_data.nc",
    points_path="weather_stations.csv",  # CSV with 'lon', 'lat' columns
    variable="temperature",
    date_col="measurement_date"  # Optional: for temporal filtering
)

# Fast multi-scenario analysis (10-30 seconds)
results_df = extractor.generate_multi_scenario_results(
    cache_id=cache_id,
    days_back_list=[3, 7, 14, 30, 90],  # Multiple averaging windows
    date_col="measurement_date",
    output_path="multi_scenario_results.csv"
)

print(f"✅ Generated results for {len(results_df)} points")
print(f"📊 Available columns: {list(results_df.columns)}")
```

### 🌍 Polygon Extraction (Administrative Regions, NUTS)

```python
from netcdfkit import NetCDFPolygonExtractor

# Initialize polygon extractor
extractor = NetCDFPolygonExtractor(cache_dir="polygon_cache")

# Extract statistics for NUTS3 regions (one-time operation)
cache_id = extractor.extract_and_cache_statistics(
    netcdf_path="temperature_data.nc",
    shapefile_path="NUTS_RG_L3.shp",
    variable="temperature", 
    id_column="NUTS_ID",
    statistics=["mean", "std", "min", "max", "median"]
)

# Load time series for specific country
germany_data = extractor.load_polygon_timeseries(
    cache_id=cache_id,
    country_code="DE",  # Germany
    statistic="mean"
)

# Export results
extractor.export_timeseries_csv(
    cache_id=cache_id,
    output_path="germany_temperature.csv",
    country_code="DE",
    statistic="mean",
    wide_format=True  # Time as rows, regions as columns
)

print(f"✅ Processed {len(germany_data)} German regions")
```

## 📊 Input Data Formats

### 🎯 Point Data CSV Format

Your CSV file should contain at minimum the required columns:

```csv
lon,lat,station_id,measurement_date
8.6821,50.1109,STATION_001,2023-06-15
13.4050,52.5200,STATION_002,2023-06-15
9.9937,53.5511,STATION_003,2023-06-15
11.5755,48.1372,STATION_004,2023-06-15
```

**Required columns:**
- `lon`: Longitude (decimal degrees)  
- `lat`: Latitude (decimal degrees)

**Optional columns:**
- `date_col`: Reference date for temporal analysis
- Additional metadata columns (preserved in output)

### 🌍 Polygon Data (Shapefiles)

- **Supported formats**: ESRI Shapefile (.shp with .dbf, .prj, .shx files)
- **Required fields**: Unique ID column (e.g., NUTS_ID, region_id, admin_code)
- **Coordinate systems**: Automatic detection and transformation
- **Examples**: NUTS regions, administrative boundaries, custom polygons

**Popular datasets:**
- [NUTS Regions](https://ec.europa.eu/eurostat/web/gisco/geodata/reference-data/administrative-units-statistical-units/nuts)
- [GADM Administrative Areas](https://gadm.org/)
- [Natural Earth](https://www.naturalearthdata.com/)

## 🔧 Core Functionality

### 🎯 Point Extraction API

#### 1. Extract and Cache Time Series

```python
# Extract time series for all points (one-time operation)
cache_id = extractor.extract_and_cache_timeseries(
    netcdf_path="era5_temperature.nc",
    points_path="weather_stations.csv",
    variable="t2m",  # Variable name in NetCDF
    date_col="observation_date",  # Optional: reference date column
    eps_km=100,  # Optional: spatial clustering radius (km)
    min_samples=3,  # Optional: minimum points per cluster
    force_recache=False  # Use existing cache if available
)
```

#### 2. Multi-Scenario Analysis

```python
# Generate multiple averaging scenarios in one operation
results = extractor.generate_multi_scenario_results(
    cache_id=cache_id,
    days_back_list=[1, 3, 7, 14, 30, 60, 90],  # Different averaging windows
    date_col="observation_date",
    output_path="temperature_scenarios.csv"  # Optional: save to file
)
```

#### 3. Access Individual Time Series

```python
# Load specific point time series
point_timeseries = extractor.load_point_timeseries(
    cache_id=cache_id,
    point_ids=[0, 5, 10]  # List of point indices, or "all" for all points
)

# Access individual time series (returns pandas Series with datetime index)
ts_point_0 = point_timeseries[0]
print(f"📅 Time range: {ts_point_0.index.min()} to {ts_point_0.index.max()}")
print(f"📊 Data points: {len(ts_point_0)}")
print(f"📈 Mean value: {ts_point_0.mean():.2f}")
```

#### 4. Export Time Series Data

```python
# Export time series to CSV format
export_df = extractor.export_point_timeseries_csv(
    cache_id=cache_id,
    point_ids=[0, 1, 2, 5],  # Specific points, or "all"
    output_path="selected_timeseries.csv"
)
```

#### 5. Cache Management

```python
# List all cached extractions
cached_list = extractor.list_cached_extractions()
print(cached_list)

# Returns DataFrame with columns:
# cache_id, n_points, variable, time_range, n_timesteps, n_chunks, file_size
```

### 🌍 Polygon Extraction API

#### 1. Extract and Cache Statistics

```python
# Extract statistics for all polygons (one-time operation)
cache_id = extractor.extract_and_cache_statistics(
    netcdf_path="temperature_data.nc",
    shapefile_path="NUTS_RG_L3.shp",
    variable="temperature",
    id_column="NUTS_ID",  # Unique identifier column in shapefile
    statistics=["mean", "std", "min", "max", "median", "count"],
    force_recache=False
)
```

**Available statistics:**
- `mean`: Average value within polygon
- `std`: Standard deviation
- `min`: Minimum value
- `max`: Maximum value  
- `median`: Median value
- `count`: Number of valid grid cells

#### 2. Load Time Series with Filtering

```python
# Load time series with comprehensive filtering options
timeseries_dict = extractor.load_polygon_timeseries(
    cache_id=cache_id,
    polygon_ids=["DE111", "DE112"],  # Specific polygon IDs
    country_code="DE",               # Filter by country code
    start_date="2020-01-01",         # Temporal filtering
    end_date="2020-12-31",
    statistic="mean"                 # Which statistic to load
)
```

#### 3. Export Polygon Data

```python
# Export in long format (one row per polygon-time combination)
extractor.export_timeseries_csv(
    cache_id=cache_id,
    output_path="regions_temperature_long.csv",
    country_code="DE",
    statistic="mean",
    wide_format=False  # Long format
)

# Export in wide format (time as rows, polygons as columns)
extractor.export_timeseries_csv(
    cache_id=cache_id,
    output_path="regions_temperature_wide.csv", 
    country_code="DE",
    statistic="mean",
    wide_format=True  # Wide format for time series analysis
)
```

#### 4. Region Information and Management

```python
# List available countries in the dataset
countries_df = extractor.list_countries(cache_id)
print(countries_df)

# Get detailed polygon metadata
polygon_info = extractor.get_polygon_info(
    cache_id=cache_id,
    polygon_ids=["DE111", "DE112", "FR101"]
)

# Get cache information and statistics
cache_info = extractor.get_cache_info(cache_id)
print(f"📊 Cache info: {cache_info}")
```

## 🎯 Complete Workflow Examples

### 🌡️ Climate Data Analysis Workflow

```python
from netcdfkit import NetCDFPointExtractor, NetCDFPolygonExtractor

# Step 1: Initialize extractors
point_extractor = NetCDFPointExtractor(cache_dir="climate_analysis")
polygon_extractor = NetCDFPolygonExtractor(cache_dir="regional_analysis")

# Step 2: Extract weather station data
station_cache_id = point_extractor.extract_and_cache_timeseries(
    netcdf_path="era5_temperature_2020_2023.nc",
    points_path="weather_stations_europe.csv",
    variable="t2m",
    date_col="observation_date"
)

# Step 3: Generate temporal scenarios for stations
station_scenarios = point_extractor.generate_multi_scenario_results(
    cache_id=station_cache_id,
    days_back_list=[1, 7, 30, 90],
    date_col="observation_date",
    output_path="station_temperature_scenarios.csv"
)

# Step 4: Extract regional statistics
regional_cache_id = polygon_extractor.extract_and_cache_statistics(
    netcdf_path="era5_temperature_2020_2023.nc",
    shapefile_path="NUTS_RG_L3.shp",
    variable="t2m",
    id_column="NUTS_ID",
    statistics=["mean", "std", "min", "max"]
)

# Step 5: Compare countries
countries = ["DE", "FR", "ES", "IT"]
for country in countries:
    # Export country-specific data
    polygon_extractor.export_timeseries_csv(
        cache_id=regional_cache_id,
        output_path=f"{country}_regional_temperature.csv",
        country_code=country,
        statistic="mean"
    )

print("✅ Complete climate analysis workflow finished!")
```

### 🌾 Environmental Monitoring Workflow

```python
# Air quality monitoring example
air_quality_extractor = NetCDFPointExtractor(cache_dir="air_quality_cache")

# Extract NO2 data at monitoring stations
no2_cache_id = air_quality_extractor.extract_and_cache_timeseries(
    netcdf_path="copernicus_no2_daily.nc",
    points_path="air_quality_stations.csv",
    variable="no2_surface_concentration",
    date_col="measurement_date"
)

# Generate exposure scenarios
exposure_results = air_quality_extractor.generate_multi_scenario_results(
    cache_id=no2_cache_id,
    days_back_list=[1, 7, 30],  # Daily, weekly, monthly averages
    date_col="measurement_date",
    output_path="no2_exposure_scenarios.csv"
)

# Regional health assessment
health_extractor = NetCDFPolygonExtractor(cache_dir="health_assessment")

regional_no2_cache = health_extractor.extract_and_cache_statistics(
    netcdf_path="copernicus_no2_daily.nc",
    shapefile_path="administrative_regions.shp",
    variable="no2_surface_concentration",
    id_column="region_id",
    statistics=["mean", "max", "count"]
)

print("✅ Environmental monitoring workflow completed!")
```

## ⚡ Performance Benchmarks

### Expected Performance (Typical Desktop/Server)

| Dataset Configuration | First Extraction | Scenario Generation | Memory Usage | Storage |
|----------------------|------------------|-------------------|--------------|---------|
| 250 points, 50GB NetCDF | 5-15 minutes | 10-30 seconds | 3-8 GB | 500MB-2GB |
| 1,000 points, 100GB NetCDF | 15-30 minutes | 30-60 seconds | 4-10 GB | 1-4GB |
| 5,000 points, 200GB NetCDF | 30-60 minutes | 1-3 minutes | 8-20 GB | 3-8GB |
| 20,000 points, 200GB NetCDF | 45-90 minutes | 2-5 minutes | 10-25 GB | 5-15GB |
| **Polygon Extraction** | | | | |
| 100 NUTS3 regions, 50GB | 10-25 minutes | 15-45 seconds | 4-12 GB | 1-3GB |
| 500 NUTS3 regions, 100GB | 20-45 minutes | 30-90 seconds | 6-15 GB | 2-6GB |
| 1,500 NUTS3 regions, 200GB | 30-90 minutes | 1-3 minutes | 8-20 GB | 4-12GB |

### ⚡ After Caching (Subsequent Analyses)

- **Any new scenario**: 10 seconds - 2 minutes
- **Memory usage**: 1-5 GB
- **Storage**: Cached data enables instant reanalysis
- **Reusability**: Same cache for multiple analysis scenarios

### 🔧 Performance Optimization Tips

1. **🔥 Use SSD storage** for NetCDF files and cache directory
2. **💾 Ensure adequate RAM** (16-32GB recommended for large datasets)
3. **🖥️ Close other applications** to free memory during extraction
4. **🌐 Leverage spatial clustering** - automatically optimized for your point distribution
5. **💾 Use caching effectively** - extract once, analyze multiple times
6. **⚙️ Adjust chunk sizes** based on your system capabilities

## 🔬 Advanced Features

### 🎯 Custom Spatial Clustering

```python
# Override automatic clustering for specific use cases
cache_id = extractor.extract_and_cache_timeseries(
    netcdf_path="global_climate_data.nc",
    points_path="sparse_stations.csv",
    variable="temperature",
    eps_km=200,      # Larger radius for sparse global stations
    min_samples=2    # Minimum points per cluster for sparse data
)
```

### 📊 Custom Temporal Analysis

```python
# Seasonal analysis example
def extract_seasonal_averages(cache_id, output_path):
    """Extract seasonal averages for all points"""
    all_timeseries = extractor.load_point_timeseries(cache_id, "all")
    
    results = []
    for point_id, ts in all_timeseries.items():
        # Group by meteorological seasons
        ts_with_season = ts.to_frame('value')
        ts_with_season['month'] = ts_with_season.index.month
        ts_with_season['season'] = ts_with_season['month'].map({
            12: 'Winter', 1: 'Winter', 2: 'Winter',
            3: 'Spring', 4: 'Spring', 5: 'Spring', 
            6: 'Summer', 7: 'Summer', 8: 'Summer',
            9: 'Autumn', 10: 'Autumn', 11: 'Autumn'
        })
        
        seasonal_means = ts_with_season.groupby('season')['value'].mean()
        for season, value in seasonal_means.items():
            results.append({
                'point_id': point_id,
                'season': season,
                'mean_value': value
            })
    
    results_df = pd.DataFrame(results)
    results_df.to_csv(output_path, index=False)
    return results_df

# Use the custom function
seasonal_results = extract_seasonal_averages(
    cache_id=cache_id,
    output_path="seasonal_climate_analysis.csv"
)
```

### 🌍 Multi-Statistic Analysis

```python
# Extract all available statistics for comprehensive analysis
cache_id = polygon_extractor.extract_and_cache_statistics(
    netcdf_path="precipitation_data.nc",
    shapefile_path="watersheds.shp",
    variable="precipitation",
    id_column="watershed_id",
    statistics=["mean", "std", "min", "max", "median", "count"]
)

# Load different statistics for comparison
mean_data = polygon_extractor.load_polygon_timeseries(cache_id, statistic="mean")
variability_data = polygon_extractor.load_polygon_timeseries(cache_id, statistic="std")

# Analyze spatial variability within regions
for region_id in mean_data.keys():
    if region_id in variability_data:
        mean_ts = mean_data[region_id]
        std_ts = variability_data[region_id]
        cv_ts = std_ts / mean_ts  # Coefficient of variation
        print(f"Region {region_id} - Average CV: {cv_ts.mean():.2f}")
```

## 📁 Directory Structure

The system creates an organized cache structure for efficient data management:

### 🎯 Point Extraction Cache

```
cache_dir/
├── metadata/
│   ├── extract_[dataset]_[hash].json         # Extraction configuration & metadata
│   └── extract_[dataset]_[hash]_points.csv   # Point coordinates & metadata
└── timeseries/
    └── extract_[dataset]_[hash]_timeseries.parquet  # Compressed time series data
```

### 🌍 Polygon Extraction Cache

```
cache_dir/
├── metadata/
│   ├── extract_[dataset].json                # Extraction configuration
│   └── extract_[dataset]_polygons.parquet    # Polygon geometries & metadata
└── statistics/
    └── extract_[dataset]_statistics.parquet  # Statistical time series by region
```

### 📊 Cache File Details

- **JSON metadata**: Configuration, timestamps, data ranges, processing parameters
- **Parquet files**: Compressed, efficient storage with fast read/write
- **CSV exports**: Human-readable output for analysis and visualization
- **Automatic cleanup**: Organized structure prevents cache bloat

## 🎯 Use Cases & Applications

### 🌡️ Climate & Environmental Science

- **🌡️ Temperature Analysis**: Extract temperature time series for weather station validation
- **🌧️ Precipitation Studies**: Get rainfall data for hydrological catchment analysis
- **🌬️ Air Quality Monitoring**: Analyze pollution measurements at sensor network locations
- **🌾 Agricultural Research**: Get weather/climate data for farm and field locations
- **🌊 Oceanographic Studies**: Extract sea surface temperature at buoy locations
- **❄️ Cryosphere Research**: Analyze ice/snow data for polar and alpine regions

### 🗺️ Geospatial Analysis

- **🏛️ Administrative Regions**: Calculate statistics for NUTS regions, counties, states
- **🏙️ Urban Planning**: Analyze environmental conditions within city boundaries  
- **🌲 Natural Resources**: Monitor conditions within protected areas and forest zones
- **⚠️ Risk Assessment**: Calculate exposure metrics for administrative regions
- **🚨 Emergency Response**: Rapid extraction for disaster monitoring and response
- **📊 Policy Analysis**: Support evidence-based environmental policy decisions

### 🔬 Research Applications

- **🌍 Climate Data Analysis**: Process ERA5, CORDEX, CMIP6 climate model outputs
- **🛰️ Satellite Data Processing**: Extract time series from satellite-derived products
- **📡 Environmental Monitoring**: Process sensor network data with spatial context
- **🏭 Industrial Applications**: Environmental impact assessment and monitoring
- **🎓 Academic Research**: Support for PhD/postdoc research projects
- **📈 Operational Services**: Real-time environmental monitoring systems

## 🔧 Troubleshooting

### Common Issues and Solutions

| Issue | Symptoms | Solution |
|-------|----------|----------|
| **Memory errors** | Process killed, out of memory | Close other applications, restart Python kernel, use smaller point batches |
| **Slow extraction** | Long processing times | Ensure NetCDF on SSD, check spatial clustering efficiency, verify coordinate systems |
| **Missing data** | NaN values, empty results | Verify NetCDF variable names, check coordinate bounds, inspect time ranges |
| **Cache issues** | Corrupted cache, repeated processing | Set `force_recache=True`, clear cache directory, check disk space |
| **Coordinate problems** | Misaligned data, wrong locations | Check NetCDF CRS metadata, verify lat/lon order, inspect projection settings |
| **Polygon errors** | Failed region processing | Ensure shapefile has valid geometries, check required ID column exists |

### 🐛 Getting Help

1. **Check the documentation** and examples first
2. **Verify your input data** format and coordinate systems  
3. **Monitor memory usage** during processing
4. **Check GitHub Issues** for similar problems
5. **Create detailed bug reports** with data specifications

### 💡 Performance Issues

```python
# Monitor memory usage during processing
import psutil

def monitor_memory():
    memory_mb = psutil.Process().memory_info().rss / 1024 / 1024
    print(f"Current memory usage: {memory_mb:.1f} MB")

# Call periodically during long operations
monitor_memory()
```


## 🤝 Contributing

We welcome contributions! Here's how you can help:

### 💻 Development Setup

```bash
# Clone the repository
git clone https://github.com/MuhammadShafeeque/netcdfkit.git
cd netcdfkit

# Create development environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install in development mode
pip install -e .[dev]

# Run tests
pytest tests/
```

### 🐛 Bug Reports

Please include:
- NetCDF file characteristics (size, variables, coordinate system)
- Point/polygon data format and size
- System specifications (RAM, OS, Python version)
- Complete error traceback
- Minimal reproducible example

### ✨ Feature Requests

We're especially interested in:
- Additional statistical functions for polygon extraction
- Support for new coordinate systems
- Performance optimizations
- Integration with other geospatial libraries
- New output formats

## 👨‍💻 Creator & Support

**Creator**: Muhammad Shafeeque  
**Email**: muhammad.shafeeque@awi.de  
**Institution**: [Data Science Support](https://www.awi.de/en/about-us/service/computing-centre/data-management/data-science-support.html), Computing and Data Center  
**Organization**: [Alfred Wegener Institute for Polar and Marine Research](https://www.awi.de/en/)  
**Project**: [HealthyPlanet Project](https://www.bremen-research.de/en/datanord/research-academy/healthy-planet) under [DataNord](https://www.bremen-research.de/datanord)  
**GitHub**: [MuhammadShafeeque](https://github.com/MuhammadShafeeque)

### 🆘 Support Channels

- **📧 Email**: Technical questions and collaboration inquiries
- **🐛 GitHub Issues**: Bug reports and feature requests  
- **📖 Documentation**: Comprehensive guides and examples
- **💬 Discussions**: Community support and use case sharing

## 📄 License

This project is licensed under the **MIT License** - see the [LICENSE](LICENSE) file for details.

### 📋 Citation

If you use NetCDFKit in your research, please cite:

```bibtex
@software{netcdfkit2024,
  title={NetCDFKit: High-Performance NetCDF Data Extraction Toolkit},
  author={Shafeeque, Muhammad},
  year={2024},
  url={https://github.com/MuhammadShafeeque/netcdfkit},
  institution={Alfred Wegener Institute for Polar and Marine Research}
}
```

## 🎉 Acknowledgments

- **Alfred Wegener Institute** for institutional support
- **HealthyPlanet Project** and **DataNord** for funding
- **Open source community** for the fantastic libraries that make this possible
- **Climate and environmental research community** for feedback and use cases

---

<div align="center">

**🌍 Built with ❤️ for the climate and environmental science community**

[![PyPI](https://img.shields.io/pypi/v/netcdfkit.svg)](https://pypi.org/project/netcdfkit/)
[![Python Version](https://img.shields.io/pypi/pyversions/netcdfkit.svg)](https://pypi.org/project/netcdfkit/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

[🏠 Homepage](https://github.com/MuhammadShafeeque/netcdfkit) • 
[📚 Documentation](https://github.com/MuhammadShafeeque/netcdfkit/tree/main/examples) • 
[🐛 Issues](https://github.com/MuhammadShafeeque/netcdfkit/issues) • 
[💬 Discussions](https://github.com/MuhammadShafeeque/netcdfkit/discussions)

</div>