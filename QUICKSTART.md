# NetCDFKit Quick Start Guide

This guide will get you up and running with NetCDFKit in just a few minutes.

## Installation

```bash
pip install netcdfkit
```

## Basic Usage

### 1. Point Extraction (Most Common Use Case)

Extract time series data at specific geographic coordinates:

```python
from netcdfkit import NetCDFPointExtractor

# Initialize
extractor = NetCDFPointExtractor(cache_dir="my_cache")

# Extract time series (one-time operation, 5-15 minutes)
cache_id = extractor.extract_and_cache_timeseries(
    netcdf_path="your_data.nc",      # Your NetCDF file
    points_path="coordinates.csv",    # CSV with lon,lat columns
    variable="temperature",           # Variable name in NetCDF
    date_col="measurement_date"       # Optional: date column in CSV
)

# Generate analysis results (30 seconds)
results = extractor.generate_multi_scenario_results(
    cache_id=cache_id,
    days_back_list=[7, 30, 90],      # 7-day, 30-day, 90-day averages
    output_path="results.csv"
)

print(f"✅ Generated results for {len(results)} points")
```

### 2. Polygon Extraction (Administrative Regions)

Calculate statistics within polygon regions:

```python
from netcdfkit import NetCDFPolygonExtractor

# Initialize
extractor = NetCDFPolygonExtractor(cache_dir="polygon_cache")

# Extract statistics (one-time operation, 10-30 minutes)
cache_id = extractor.extract_and_cache_statistics(
    netcdf_path="your_data.nc",
    shapefile_path="regions.shp",    # Shapefile with polygons
    variable="temperature",
    id_column="REGION_ID",           # Unique ID column in shapefile
    statistics=["mean", "std", "min", "max"]
)

# Load data for specific country
germany_data = extractor.load_polygon_timeseries(
    cache_id=cache_id,
    country_code="DE",              # Germany (first 2 chars of region ID)
    statistic="mean"
)

# Export to CSV
extractor.export_timeseries_csv(
    cache_id=cache_id,
    output_path="germany_results.csv",
    country_code="DE",
    statistic="mean"
)

print(f"✅ Processed {len(germany_data)} German regions")
```

## Required Data Formats

### Point Data CSV
Your CSV file needs `lon` and `lat` columns:

```csv
lon,lat,station_name,date
8.68,50.11,Station_A,2023-01-01
13.40,52.52,Station_B,2023-01-01
```

### Polygon Data
- Shapefile (.shp) with unique ID column
- Examples: NUTS regions, administrative boundaries
- Download from [Eurostat](https://ec.europa.eu/eurostat/web/gisco/geodata) or [GADM](https://gadm.org/)

## Key Features

### Point Extraction
- ✅ **Automatic spatial clustering** for efficiency
- ✅ **Intelligent caching** - extract once, analyze many times
- ✅ **Multi-scenario processing** - different time windows in one operation
- ✅ **Memory efficient** - handles 20,000+ points within 32GB RAM

### Polygon Extraction
- ✅ **Multiple statistics** - mean, std, min, max, median, count
- ✅ **Country filtering** - easy analysis by country/region
- ✅ **Flexible output** - long or wide format CSV
- ✅ **NUTS optimized** - perfect for European administrative analysis

## Performance Expectations

| Dataset Size | First Extraction | Later Analysis | Memory Usage |
|--------------|------------------|----------------|--------------|
| 250 points   | 5-15 minutes    | 10-30 seconds  | 3-8 GB       |
| 1,000 points | 15-30 minutes   | 30-60 seconds  | 4-10 GB      |
| 5,000 points | 30-60 minutes   | 1-3 minutes    | 8-20 GB      |

## Tips for Success

1. **Use SSD storage** for NetCDF files and cache
2. **Ensure adequate RAM** (16-32GB recommended for large datasets)
3. **Extract once, analyze many times** - caching is your friend
4. **Check coordinate systems** - ensure your points match NetCDF grid
5. **Start small** - test with a subset before processing large datasets

## Common Issues

| Problem | Solution |
|---------|----------|
| Out of memory | Close other applications, use smaller datasets |
| Slow performance | Move NetCDF to SSD, check spatial clustering |
| Missing data | Verify variable names and coordinate bounds |
| Import errors | Ensure all dependencies installed |

## Next Steps

- Check `examples/` folder for complete workflows
- Read the full documentation in `README.md`
- Join discussions on GitHub for community support

## Getting Help

- 📧 Email: muhammad.shafeeque@awi.de
- 🐛 Issues: [GitHub Issues](https://github.com/MuhammadShafeeque/netcdfkit/issues)
- 📖 Examples: See `examples/` directory

Happy extracting! 🌍📊
