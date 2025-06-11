from __future__ import annotations
from pathlib import Path
from typing import Optional, Dict, List, Union, Tuple
import pandas as pd
import geopandas as gpd
import xarray as xr
import numpy as np
from pyproj import CRS, Transformer
import json
from tqdm import tqdm
import warnings
from sklearn.cluster import DBSCAN
import regionmask
import rasterio.features
from shapely.geometry import Point, Polygon, box
import dask

warnings.filterwarnings("ignore")


class NetCDFPolygonExtractor:
    """
    Optimal hybrid spatial-temporal NetCDF polygon extraction system for NUTS3 regions

    Features:
    - Automatic spatial clustering for efficient chunk processing
    - Intelligent polygon statistics caching with metadata preservation
    - Flexible time series extraction for any period and NUTS_ID selection
    - Memory-efficient processing within 32GB RAM limits
    - Support for multiple statistics (mean, std, min, max, percentiles)
    - Easy filtering by country/region codes
    - Fixed coordinate transformation and JSON serialization issues
    """

    def __init__(self, cache_dir: str | Path = "polygon_cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)

        # Cache subdirectories
        self.metadata_dir = self.cache_dir / "metadata"
        self.statistics_dir = self.cache_dir / "statistics"
        self.metadata_dir.mkdir(exist_ok=True)
        self.statistics_dir.mkdir(exist_ok=True)

        # Internal state
        self.polygons_gdf = None
        self.spatial_chunks = None
        self.dataset_info = None

    def analyze_country_distribution(
        self, polygons_gdf: gpd.GeoDataFrame, max_chunk_size: int = 500
    ) -> Dict:
        """
        Create country-based chunks for efficient NetCDF access

        Parameters:
        -----------
        max_chunk_size : int
            Maximum polygons per chunk
        """

        print(f"Analyzing country distribution of {len(polygons_gdf)} polygons...")

        # Get country codes
        if "CNTR_CODE" in polygons_gdf.columns:
            country_col = "CNTR_CODE"
        else:
            # Infer from NUTS_ID (first 2 characters)
            country_col = "inferred_country"
            polygons_gdf[country_col] = polygons_gdf.index.str[:2]

        country_counts = polygons_gdf[country_col].value_counts()

        cluster_info = {
            "n_countries": len(country_counts),
            "country_summary": {},
            "cluster_labels": [],
        }

        print(f"Found {len(country_counts)} countries")

        # Analyze each country
        for country_code, count in country_counts.items():
            country_polygons = polygons_gdf[polygons_gdf[country_col] == country_code]
            total_bounds = country_polygons.total_bounds

            country_summary = {
                "n_polygons": count,
                "country_code": country_code,
                "bounds": {
                    "min_lon": total_bounds[0],
                    "max_lon": total_bounds[2],
                    "min_lat": total_bounds[1],
                    "max_lat": total_bounds[3],
                },
            }

            cluster_info["country_summary"][country_code] = country_summary
            print(f"  Country {country_code}: {count} polygons")

        return cluster_info

    def create_country_chunks(
        self,
        polygons_gdf: gpd.GeoDataFrame,
        cluster_info: Dict,
        max_chunk_size: int = 500,
    ) -> List[Dict]:
        """
        Create country-based chunks for efficient NetCDF access

        Parameters:
        -----------
        max_chunk_size : int
            Maximum polygons per chunk (split large countries)
        """

        print("Creating country-based chunks...")

        chunks = []
        chunk_id = 0

        # Get country column
        if "CNTR_CODE" in polygons_gdf.columns:
            country_col = "CNTR_CODE"
        else:
            country_col = "inferred_country"

        # Process each country
        for country_code, summary in cluster_info["country_summary"].items():
            country_polygons = polygons_gdf[
                polygons_gdf[country_col] == country_code
            ].copy()

            # Split large countries if needed
            if len(country_polygons) > max_chunk_size:
                n_sub_chunks = int(np.ceil(len(country_polygons) / max_chunk_size))
                print(
                    f"    Splitting {country_code} ({len(country_polygons)} polygons) into {n_sub_chunks} chunks"
                )

                # Split into roughly equal chunks
                polygon_indices = list(country_polygons.index)
                sub_chunks_indices = np.array_split(polygon_indices, n_sub_chunks)

                for i, sub_indices in enumerate(sub_chunks_indices):
                    chunk = {
                        "chunk_id": chunk_id,
                        "country_code": country_code,
                        "sub_chunk": i,
                        "n_polygons": len(sub_indices),
                        "polygon_indices": sub_indices.tolist(),
                    }
                    chunks.append(chunk)
                    chunk_id += 1
            else:
                # Single chunk for this country
                chunk = {
                    "chunk_id": chunk_id,
                    "country_code": country_code,
                    "sub_chunk": 0,
                    "n_polygons": len(country_polygons),
                    "polygon_indices": country_polygons.index.tolist(),
                }
                chunks.append(chunk)
                chunk_id += 1

        print(f"Created {len(chunks)} country-based chunks")
        for chunk in chunks:
            sub_info = f"_part{chunk['sub_chunk']}" if chunk["sub_chunk"] > 0 else ""
            print(
                f"  Chunk {chunk['chunk_id']}: {chunk['country_code']}{sub_info} - {chunk['n_polygons']} polygons"
            )

        return chunks

    def extract_and_cache_statistics(
        self,
        netcdf_path: str | Path,
        shapefile_path: str | Path,
        variable: str,
        id_column: str = "NUTS_ID",
        statistics: List[str] = ["mean"],
        force_recache: bool = False,
    ) -> str:
        """
        Extract and cache polygon statistics for all polygons

        Parameters:
        -----------
        netcdf_path : str | Path
            Path to NetCDF file
        shapefile_path : str | Path
            Path to NUTS3 shapefile
        variable : str
            Variable to extract from NetCDF
        id_column : str
            Column name containing polygon IDs (default: 'NUTS_ID')
        statistics : List[str]
            Statistics to calculate: ['mean', 'std', 'min', 'max', 'median', 'count']
        force_recache : bool
            If True, ignore existing cache and re-extract

        Returns:
        --------
        str
            Cache ID for later retrieval
        """
        # Generate cache ID
        cache_id = f"extract_{Path(shapefile_path).stem}"

        # Setup cache paths
        cache_path = self.metadata_dir / f"{cache_id}.json"
        polygons_cache_path = self.metadata_dir / f"{cache_id}_polygons.parquet"

        # Check for existing cache
        if not force_recache and cache_path.exists() and polygons_cache_path.exists():
            print(f"Found existing cache: {cache_id}")
            try:
                # Try to load cache
                with open(cache_path, "r") as f:
                    self.dataset_info = json.load(f)
                self.polygons_gdf = gpd.read_parquet(polygons_cache_path)
                return cache_id
            except (json.JSONDecodeError, Exception) as e:
                print(f"Cache appears corrupted ({str(e)}), re-extracting...")
                force_recache = True

        # Load polygon data
        print("Loading NUTS3 polygon data...")
        self.polygons_gdf = self._load_polygons(shapefile_path, id_column)

        # Open NetCDF dataset first to get CRS info
        print("Opening NetCDF dataset...")
        ds = xr.open_dataset(netcdf_path, chunks={"time": 50})

        # Debug: Print dataset info
        print(f"Dataset dimensions: {list(ds.dims.keys())}")
        print(f"Dataset coordinates: {list(ds.coords.keys())}")
        print(f"Variable '{variable}' dimensions: {ds[variable].dims}")

        # Setup coordinate mapping and transformation
        coord_mapping = self._detect_coordinate_mapping(ds, variable)
        print(f"Detected coordinate mapping: {coord_mapping}")

        ds_crs = self._get_dataset_crs(ds, variable)

        # Transform polygons to NetCDF CRS upfront
        if ds_crs:
            print(f"Transforming all polygons from EPSG:4326 to NetCDF CRS...")
            self.polygons_gdf = self._transform_all_polygons(self.polygons_gdf, ds_crs)
            print(f"Polygon transformation complete")
        else:
            print("No CRS transformation needed")

        # Analyze country distribution
        cluster_info = self.analyze_country_distribution(self.polygons_gdf)

        # Create country-based chunks
        self.spatial_chunks = self.create_country_chunks(
            self.polygons_gdf, cluster_info
        )

        # Extract statistics chunk by chunk
        print(f"\nExtracting statistics for {len(self.spatial_chunks)} chunks...")
        print(f"Statistics to calculate: {statistics}")

        all_statistics = {}

        for chunk in tqdm(self.spatial_chunks, desc="Processing chunks"):
            chunk_statistics = self._extract_chunk_statistics(
                ds, variable, chunk, coord_mapping, statistics
            )
            all_statistics.update(chunk_statistics)

        # Save cached statistics
        print("Saving cached statistics...")
        if all_statistics:
            self._save_statistics_cache(all_statistics, cache_id, statistics)
            print(f"Saved statistics for {len(all_statistics)} polygons")
        else:
            print("Warning: No statistics were successfully extracted!")
            # Create empty cache file to avoid issues
            cache_file = self.statistics_dir / f"{cache_id}_statistics.parquet"
            empty_cols = [id_column, "time"] + [f"{stat}_value" for stat in statistics]
            empty_df = pd.DataFrame(columns=empty_cols)
            empty_df.to_parquet(cache_file, index=False)
            print("Created empty cache file")

        # Save metadata
        self.dataset_info = {
            "cache_id": cache_id,
            "netcdf_path": str(netcdf_path),
            "shapefile_path": str(shapefile_path),
            "variable": variable,
            "id_column": id_column,
            "n_polygons": len(self.polygons_gdf),
            "statistics": statistics,
            "country_chunks": self._convert_numpy_types(self.spatial_chunks),
            "cluster_info": self._convert_numpy_types(cluster_info),
            "time_range": {
                "start": str(ds.time.min().values),
                "end": str(ds.time.max().values),
                "n_timesteps": int(len(ds.time)),
            },
        }

        with open(cache_path, "w") as f:
            json.dump(self.dataset_info, f, indent=2)

        # Save polygons metadata (now in NetCDF CRS)
        self.polygons_gdf.to_parquet(polygons_cache_path, index=True)

        print(f"\nCaching complete! Cache ID: {cache_id}")
        print(f"Statistics cached for {len(all_statistics)} polygons")

        return cache_id

    def _detect_coordinate_mapping(
        self, ds: xr.Dataset, variable: str
    ) -> Dict[str, str]:
        """Detect the coordinate names used in the NetCDF file"""

        var_dims = ds[variable].dims
        coord_mapping = {}

        # Common coordinate name patterns
        lon_patterns = ["lon", "longitude", "x", "X", "XLONG"]
        lat_patterns = ["lat", "latitude", "y", "Y", "XLAT"]

        # Find longitude coordinate
        for pattern in lon_patterns:
            if pattern in var_dims or pattern in ds.coords:
                coord_mapping["lon"] = pattern
                break

        # Find latitude coordinate
        for pattern in lat_patterns:
            if pattern in var_dims or pattern in ds.coords:
                coord_mapping["lat"] = pattern
                break

        return coord_mapping

    def _transform_all_polygons(
        self, polygons_gdf: gpd.GeoDataFrame, ds_crs: CRS
    ) -> gpd.GeoDataFrame:
        """Transform all polygons from EPSG:4326 to NetCDF CRS"""

        from pyproj import Transformer

        # Create transformer
        input_crs = CRS.from_epsg(4326)
        transformer = Transformer.from_crs(input_crs, ds_crs, always_xy=True)

        # Transform all geometries
        transformed_gdf = polygons_gdf.copy()
        transformed_gdf["geometry"] = transformed_gdf["geometry"].apply(
            lambda geom: self._transform_geometry(geom, transformer)
        )

        # Update the CRS
        transformed_gdf.crs = ds_crs

        return transformed_gdf

    def _extract_chunk_statistics(
        self,
        ds: xr.Dataset,
        variable: str,
        chunk: Dict,
        coord_mapping: Dict[str, str],
        statistics: List[str],
    ) -> Dict:
        """Extract statistics for all polygons in a chunk using rasterio rasterization"""

        # Get polygons in this chunk
        chunk_polygon_indices = chunk["polygon_indices"]
        chunk_polygons = self.polygons_gdf.loc[chunk_polygon_indices]

        print(
            f"    Processing {chunk['country_code']} - {len(chunk_polygons)} polygons"
        )

        # Get coordinate names
        lon_coord = coord_mapping.get("lon", "x")
        lat_coord = coord_mapping.get("lat", "y")

        print(
            f"    Using full dataset ({ds[lon_coord].size} x {ds[lat_coord].size} grid)"
        )

        # Get coordinate arrays
        x_coords = ds[lon_coord].values
        y_coords = ds[lat_coord].values

        # Create coordinate grids
        X, Y = np.meshgrid(x_coords, y_coords)

        # Get grid properties for rasterization
        transform = self._get_affine_transform(x_coords, y_coords)

        print(f"    Grid transform: {transform}")

        # Extract statistics for each polygon
        chunk_statistics = {}

        for polygon_idx, polygon_row in chunk_polygons.iterrows():
            try:
                polygon_geom = polygon_row.geometry

                # Create mask using rasterio
                mask = rasterio.features.rasterize(
                    [polygon_geom],
                    out_shape=(len(y_coords), len(x_coords)),
                    transform=transform,
                    fill=0,
                    default_value=1,
                    dtype=np.uint8,
                )

                # Convert to boolean mask
                polygon_mask = mask.astype(bool)

                # Check if we have any valid data
                if not polygon_mask.any():
                    print(
                        f"    ✗ Polygon {polygon_idx}: No grid cells found within polygon"
                    )
                    continue

                # Create xarray DataArray mask with proper coordinates
                mask_da = xr.DataArray(
                    polygon_mask,
                    coords={lat_coord: y_coords, lon_coord: x_coords},
                    dims=[lat_coord, lon_coord],
                )

                # Apply mask to data
                masked_data = ds[variable].where(mask_da)

                # Calculate statistics over spatial dimensions
                polygon_stats = {}

                for stat in statistics:
                    if stat == "mean":
                        stat_values = masked_data.mean(
                            dim=[lon_coord, lat_coord], skipna=True
                        )
                    elif stat == "std":
                        stat_values = masked_data.std(
                            dim=[lon_coord, lat_coord], skipna=True
                        )
                    elif stat == "min":
                        stat_values = masked_data.min(
                            dim=[lon_coord, lat_coord], skipna=True
                        )
                    elif stat == "max":
                        stat_values = masked_data.max(
                            dim=[lon_coord, lat_coord], skipna=True
                        )
                    elif stat == "median":
                        stat_values = masked_data.median(
                            dim=[lon_coord, lat_coord], skipna=True
                        )
                    elif stat == "count":
                        stat_values = masked_data.count(dim=[lon_coord, lat_coord])
                    else:
                        print(f"Unknown statistic: {stat}")
                        continue

                    # Convert to pandas Series
                    if not stat_values.isnull().all():
                        stat_series = stat_values.to_series()
                        polygon_stats[stat] = stat_series
                    else:
                        print(f"    ✗ Polygon {polygon_idx}: No valid data for {stat}")

                if polygon_stats:
                    chunk_statistics[polygon_idx] = polygon_stats
                    n_timesteps = len(list(polygon_stats.values())[0])
                    n_cells = int(polygon_mask.sum())
                    print(
                        f"    ✓ Polygon {polygon_idx}: {len(statistics)} statistics over {n_timesteps} timesteps ({n_cells} grid cells)"
                    )
                else:
                    print(
                        f"    ✗ Polygon {polygon_idx}: No valid statistics calculated"
                    )

            except Exception as e:
                print(f"    ✗ Polygon {polygon_idx}: Failed to extract: {str(e)}")
                import traceback

                print(f"    Error details: {traceback.format_exc()}")
                continue

        return chunk_statistics

    def _get_affine_transform(self, x_coords: np.ndarray, y_coords: np.ndarray):
        """Create affine transform for rasterio from coordinate arrays"""
        from rasterio.transform import from_bounds

        # Calculate pixel sizes
        dx = x_coords[1] - x_coords[0] if len(x_coords) > 1 else 1000  # default 1km
        dy = y_coords[1] - y_coords[0] if len(y_coords) > 1 else 1000  # default 1km

        # Get bounds
        west, east = x_coords[0], x_coords[-1]
        north, south = y_coords[0], y_coords[-1]

        # For regular grids, use from_bounds
        width = len(x_coords)
        height = len(y_coords)

        transform = from_bounds(west, south, east, north, width, height)

        return transform

    def _transform_geometry(self, geometry, transformer):
        """Transform a shapely geometry using pyproj transformer"""
        if geometry.geom_type == "Polygon":
            exterior_coords = list(geometry.exterior.coords)
            transformed_coords = [
                transformer.transform(lon, lat) for lon, lat in exterior_coords
            ]

            # Handle interior holes if present
            holes = []
            for interior in geometry.interiors:
                hole_coords = list(interior.coords)
                transformed_hole = [
                    transformer.transform(lon, lat) for lon, lat in hole_coords
                ]
                holes.append(transformed_hole)

            return Polygon(transformed_coords, holes)

        elif geometry.geom_type == "MultiPolygon":
            transformed_polygons = []
            for polygon in geometry.geoms:
                transformed_polygons.append(
                    self._transform_geometry(polygon, transformer)
                )
            return type(geometry)(transformed_polygons)

        else:
            # For other geometry types, use shapely's transform method
            from shapely.ops import transform

            return transform(lambda lon, lat: transformer.transform(lon, lat), geometry)

    def _save_statistics_cache(
        self, all_statistics: Dict, cache_id: str, statistics: List[str]
    ):
        """Save statistics data efficiently using Parquet format"""

        # Convert to DataFrame format suitable for Parquet
        statistics_data = []

        for polygon_idx, polygon_stats in all_statistics.items():
            if polygon_stats:
                # Get time index from first statistic
                time_index = list(polygon_stats.values())[0].index

                for time_point in time_index:
                    row = {
                        "polygon_id": str(polygon_idx),
                        "time": time_point,
                    }  # Ensure polygon_id is string

                    # Add each statistic value
                    for stat in statistics:
                        if stat in polygon_stats:
                            row[f"{stat}_value"] = polygon_stats[stat].loc[time_point]
                        else:
                            row[f"{stat}_value"] = np.nan

                    statistics_data.append(row)

        if statistics_data:
            # Combine all statistics
            combined_df = pd.DataFrame(statistics_data)

            # Save as Parquet for efficient storage and retrieval
            cache_file = self.statistics_dir / f"{cache_id}_statistics.parquet"
            combined_df.to_parquet(cache_file, index=False)

            print(
                f"Saved statistics for {len(all_statistics)} polygons to {cache_file}"
            )
            print(f"DataFrame columns: {list(combined_df.columns)}")
            print(f"DataFrame shape: {combined_df.shape}")
        else:
            print("Warning: No statistics data to save!")

    def load_polygon_timeseries(
        self,
        cache_id: str,
        polygon_ids: Union[str, List[str], None] = None,
        country_code: Optional[str] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        statistic: str = "mean",
    ) -> Dict[str, pd.Series]:
        """
        Load time series for specific polygons with flexible filtering

        Parameters:
        -----------
        cache_id : str
            Cache identifier from extract_and_cache_statistics()
        polygon_ids : str, list, or None
            Specific polygon ID(s) to load, or None for all
        country_code : str, optional
            Country code to filter by (e.g., 'DE' for Germany)
        start_date : str, optional
            Start date for time filtering (YYYY-MM-DD)
        end_date : str, optional
            End date for time filtering (YYYY-MM-DD)
        statistic : str
            Which statistic to extract (default: 'mean')

        Returns:
        --------
        Dict[polygon_id, time_series]
        """

        cache_file = self.statistics_dir / f"{cache_id}_statistics.parquet"

        if not cache_file.exists():
            raise FileNotFoundError(f"Cache file not found: {cache_file}")

        # Load parquet data
        df = pd.read_parquet(cache_file)

        # Debug: Print available columns
        print(f"Available columns in cache: {list(df.columns)}")

        # Handle different column naming conventions
        polygon_id_col = None
        for possible_col in ["polygon_id", "NUTS_ID", "id"]:
            if possible_col in df.columns:
                polygon_id_col = possible_col
                break

        if polygon_id_col is None:
            # Try to find any column that might be the ID
            id_candidates = [
                col
                for col in df.columns
                if "id" in col.lower() or col in df.columns[:3]
            ]
            if id_candidates:
                polygon_id_col = id_candidates[0]
                print(f"Using '{polygon_id_col}' as polygon ID column")
            else:
                raise ValueError(
                    f"Could not find polygon ID column. Available columns: {list(df.columns)}"
                )

        # Check if the requested statistic exists
        stat_col = f"{statistic}_value"
        if stat_col not in df.columns:
            available_stats = [
                col.replace("_value", "")
                for col in df.columns
                if col.endswith("_value")
            ]
            raise ValueError(
                f"Statistic '{statistic}' not found. Available: {available_stats}"
            )

        # Load polygon metadata for filtering
        polygons_file = self.metadata_dir / f"{cache_id}_polygons.parquet"
        polygons_gdf = gpd.read_parquet(polygons_file)

        # Apply polygon filtering
        filtered_polygon_ids = self._filter_polygons(
            polygons_gdf, polygon_ids, country_code
        )

        # Filter data to selected polygons
        df = df[df[polygon_id_col].isin(filtered_polygon_ids)]

        if df.empty:
            print("No data found for the specified criteria")
            return {}

        # Apply time filtering
        if start_date or end_date:
            df = self._filter_by_date_range(df, start_date, end_date)

        # Convert back to time series format
        timeseries_dict = {}

        for polygon_id in df[polygon_id_col].unique():
            polygon_data = df[df[polygon_id_col] == polygon_id].copy()
            polygon_data["time"] = pd.to_datetime(polygon_data["time"])
            polygon_data = polygon_data.set_index("time").sort_index()
            timeseries = polygon_data[stat_col]
            timeseries_dict[polygon_id] = timeseries

        print(f"Loaded time series for {len(timeseries_dict)} polygons")
        if timeseries_dict:
            sample_ts = list(timeseries_dict.values())[0]
            print(f"Time range: {sample_ts.index.min()} to {sample_ts.index.max()}")
            print(f"Time steps: {len(sample_ts)}")

        return timeseries_dict

    def _filter_polygons(
        self,
        polygons_gdf: gpd.GeoDataFrame,
        polygon_ids: Union[str, List[str], None],
        country_code: Optional[str],
    ) -> List[str]:
        """Filter polygons based on IDs and/or country code"""

        filtered_gdf = polygons_gdf.copy()

        # Filter by country code if provided
        if country_code:
            if "CNTR_CODE" in filtered_gdf.columns:
                filtered_gdf = filtered_gdf[filtered_gdf["CNTR_CODE"] == country_code]
                print(
                    f"Filtered to {len(filtered_gdf)} polygons in country: {country_code}"
                )
            elif "country" in filtered_gdf.columns:
                filtered_gdf = filtered_gdf[filtered_gdf["country"] == country_code]
                print(
                    f"Filtered to {len(filtered_gdf)} polygons in country: {country_code}"
                )
            else:
                # Try to infer country from NUTS_ID (first 2 characters)
                country_mask = filtered_gdf.index.str.startswith(country_code)
                filtered_gdf = filtered_gdf[country_mask]
                print(
                    f"Filtered to {len(filtered_gdf)} polygons starting with: {country_code}"
                )

        # Filter by specific polygon IDs if provided
        if polygon_ids is not None:
            if isinstance(polygon_ids, str):
                polygon_ids = [polygon_ids]

            # Keep only polygons that exist in both the filtered set and requested IDs
            available_ids = set(filtered_gdf.index)
            requested_ids = set(polygon_ids)
            valid_ids = available_ids.intersection(requested_ids)

            if not valid_ids:
                print(
                    f"Warning: None of the requested polygon IDs found in the filtered dataset"
                )
                return []

            filtered_gdf = filtered_gdf.loc[list(valid_ids)]
            print(f"Selected {len(filtered_gdf)} specific polygon IDs")

        return filtered_gdf.index.tolist()

    def _filter_by_date_range(
        self, df: pd.DataFrame, start_date: Optional[str], end_date: Optional[str]
    ) -> pd.DataFrame:
        """Filter dataframe by date range"""

        df = df.copy()
        df["time"] = pd.to_datetime(df["time"])

        if start_date:
            start_dt = pd.to_datetime(start_date)
            df = df[df["time"] >= start_dt]
            print(f"Filtered to dates >= {start_date}")

        if end_date:
            end_dt = pd.to_datetime(end_date)
            df = df[df["time"] <= end_dt]
            print(f"Filtered to dates <= {end_date}")

        return df

    def export_timeseries_csv(
        self,
        cache_id: str,
        output_path: str | Path,
        polygon_ids: Union[str, List[str], None] = None,
        country_code: Optional[str] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        statistic: str = "mean",
        wide_format: bool = False,
    ) -> pd.DataFrame:
        """
        Export time series to CSV with flexible filtering

        Parameters:
        -----------
        wide_format : bool
            If True, export in wide format (polygons as columns)
            If False, export in long format (one row per polygon-time combination)
        """

        # Get time series data
        timeseries_dict = self.load_polygon_timeseries(
            cache_id=cache_id,
            polygon_ids=polygon_ids,
            country_code=country_code,
            start_date=start_date,
            end_date=end_date,
            statistic=statistic,
        )

        if not timeseries_dict:
            print("No data to export")
            return pd.DataFrame()

        if wide_format:
            # Wide format: time as index, polygons as columns
            export_df = pd.DataFrame(timeseries_dict)
            export_df.index.name = "time"
        else:
            # Long format: one row per polygon-time combination
            export_data = []
            for polygon_id, timeseries in timeseries_dict.items():
                for timestamp, value in timeseries.items():
                    export_data.append(
                        {
                            "polygon_id": polygon_id,
                            "time": timestamp,
                            f"{statistic}_value": value,
                        }
                    )
            export_df = pd.DataFrame(export_data)

        # Save to CSV
        export_df.to_csv(output_path, index=True if wide_format else False)
        print(f"Exported time series to: {output_path}")
        print(f"Format: {'Wide' if wide_format else 'Long'}")
        print(f"Shape: {export_df.shape}")

        return export_df

    def clear_cache(self, cache_id: str) -> None:
        """
        Clear a specific cache to force recreation

        Parameters:
        -----------
        cache_id : str
            Cache identifier to clear
        """

        cache_files = [
            self.metadata_dir / f"{cache_id}.json",
            self.metadata_dir / f"{cache_id}_polygons.parquet",
            self.statistics_dir / f"{cache_id}_statistics.parquet",
        ]

        removed_count = 0
        for cache_file in cache_files:
            if cache_file.exists():
                cache_file.unlink()
                removed_count += 1
                print(f"Removed: {cache_file}")

        if removed_count > 0:
            print(f"Cleared cache '{cache_id}' - removed {removed_count} files")
        else:
            print(f"No cache files found for '{cache_id}'")

    def debug_cache_structure(self, cache_id: str) -> None:
        """
        Debug the structure of cached data

        Parameters:
        -----------
        cache_id : str
            Cache identifier to debug
        """

        print(f"=== Debug Cache Structure: {cache_id} ===")

        # Check metadata
        cache_path = self.metadata_dir / f"{cache_id}.json"
        if cache_path.exists():
            with open(cache_path, "r") as f:
                metadata = json.load(f)
            print(f"Metadata keys: {list(metadata.keys())}")
            print(f"Statistics: {metadata.get('statistics', 'Not found')}")
        else:
            print("Metadata file not found")

        # Check polygon data
        polygons_file = self.metadata_dir / f"{cache_id}_polygons.parquet"
        if polygons_file.exists():
            polygons_gdf = gpd.read_parquet(polygons_file)
            print(f"Polygons shape: {polygons_gdf.shape}")
            print(f"Polygon columns: {list(polygons_gdf.columns)}")
            print(f"First 3 polygon IDs: {list(polygons_gdf.index[:3])}")
        else:
            print("Polygons file not found")

        # Check statistics data
        stats_file = self.statistics_dir / f"{cache_id}_statistics.parquet"
        if stats_file.exists():
            stats_df = pd.read_parquet(stats_file)
            print(f"Statistics shape: {stats_df.shape}")
            print(f"Statistics columns: {list(stats_df.columns)}")
            if len(stats_df) > 0:
                print(f"First row: {stats_df.iloc[0].to_dict()}")
            else:
                print("Statistics file is empty")
        else:
            print("Statistics file not found")

    def get_polygon_info(
        self, cache_id: str, polygon_ids: Union[str, List[str], None] = None
    ) -> pd.DataFrame:
        """
        Get detailed information about polygons

        Parameters:
        -----------
        cache_id : str
            Cache identifier
        polygon_ids : str, list, or None
            Specific polygon ID(s) to get info for

        Returns:
        --------
        DataFrame with polygon metadata
        """

        polygons_file = self.metadata_dir / f"{cache_id}_polygons.parquet"
        if not polygons_file.exists():
            raise FileNotFoundError(f"Polygon metadata not found: {polygons_file}")

        polygons_gdf = gpd.read_parquet(polygons_file)

        if polygon_ids is not None:
            if isinstance(polygon_ids, str):
                polygon_ids = [polygon_ids]
            polygons_gdf = polygons_gdf.loc[
                polygons_gdf.index.intersection(polygon_ids)
            ]

        # Convert to regular DataFrame (remove geometry for easier viewing)
        info_df = polygons_gdf.drop(
            columns=["geometry"] if "geometry" in polygons_gdf.columns else []
        )

        return info_df

    def list_countries(self, cache_id: str) -> pd.DataFrame:
        """List available countries in the dataset"""

        polygons_file = self.metadata_dir / f"{cache_id}_polygons.parquet"
        polygons_gdf = gpd.read_parquet(polygons_file)

        if "CNTR_CODE" in polygons_gdf.columns:
            country_counts = polygons_gdf["CNTR_CODE"].value_counts()
            country_df = pd.DataFrame(
                {
                    "country_code": country_counts.index,
                    "n_polygons": country_counts.values,
                }
            )
        else:
            # Infer from NUTS_ID (first 2 characters)
            country_codes = polygons_gdf.index.str[:2]
            country_counts = country_codes.value_counts()
            country_df = pd.DataFrame(
                {
                    "country_code": country_counts.index,
                    "n_polygons": country_counts.values,
                }
            )

        return country_df.reset_index(drop=True)

    def get_cache_info(self, cache_id: str) -> Dict:
        """Get detailed information about a cached extraction"""

        cache_path = self.metadata_dir / f"{cache_id}.json"
        if not cache_path.exists():
            raise FileNotFoundError(f"Cache metadata not found: {cache_path}")

        with open(cache_path, "r") as f:
            info = json.load(f)

        return info

    def _load_polygons(self, path: str | Path, id_column: str) -> gpd.GeoDataFrame:
        """Load polygon data from shapefile"""

        print(f"Loading polygons from: {path}")
        gdf = gpd.read_file(path)

        # Ensure required columns exist
        if id_column not in gdf.columns:
            raise ValueError(f"ID column '{id_column}' not found in shapefile")

        # Ensure geometries are valid
        gdf = gdf[gdf.geometry.is_valid]

        # Set index to ID column for easier access
        if gdf.index.name != id_column:
            gdf = gdf.set_index(id_column)

        print(f"Loaded {len(gdf)} valid polygons")
        print(f"CRS: {gdf.crs}")
        print(f"Bounds: {gdf.total_bounds}")

        return gdf

    def _get_dataset_crs(self, ds: xr.Dataset, variable: str) -> CRS | None:
        """Extract CRS from dataset (same as point extractor)"""
        try:
            # Check if variable has grid_mapping attribute
            grid_mapping_name = ds[variable].attrs.get("grid_mapping")
            if grid_mapping_name and grid_mapping_name in ds:
                gm = ds[grid_mapping_name]

                # Print CRS variable attributes for debugging
                print(f"Found CRS variable '{grid_mapping_name}' with attributes:")
                for attr, value in gm.attrs.items():
                    print(f"  {attr}: {value}")

                # Try different CRS attribute names
                if "crs_wkt" in gm.attrs:
                    return CRS.from_wkt(gm.attrs["crs_wkt"])
                if "spatial_ref" in gm.attrs:
                    return CRS.from_wkt(gm.attrs["spatial_ref"])
                if "epsg_code" in gm.attrs:
                    return CRS.from_epsg(int(gm.attrs["epsg_code"]))

                # Try to construct CRS from CF attributes
                if "grid_mapping_name" in gm.attrs:
                    return self._crs_from_cf_attributes(gm.attrs)

            # Fallback: check dataset-level CRS attributes
            if "crs_wkt" in ds.attrs:
                return CRS.from_wkt(ds.attrs["crs_wkt"])
            if "spatial_ref" in ds.attrs:
                return CRS.from_wkt(ds.attrs["spatial_ref"])

        except Exception as e:
            print(f"Warning: Could not extract CRS: {str(e)}")

        return None

    def _crs_from_cf_attributes(self, attrs: dict) -> CRS | None:
        """Try to construct CRS from CF convention attributes (same as point extractor)"""
        try:
            grid_mapping_name = attrs.get("grid_mapping_name", "")

            # Handle Lambert Azimuthal Equal Area
            if "lambert_azimuthal_equal_area" in grid_mapping_name:
                central_lon = attrs.get("longitude_of_projection_origin", 0)
                central_lat = attrs.get("latitude_of_projection_origin", 0)
                false_easting = attrs.get("false_easting", 0)
                false_northing = attrs.get("false_northing", 0)

                proj_string = f"+proj=laea +lat_0={central_lat} +lon_0={central_lon} +x_0={false_easting} +y_0={false_northing} +datum=WGS84 +units=m +no_defs"
                return CRS.from_proj4(proj_string)

            # Add other common projections as needed
            print(f"Unhandled grid mapping: {grid_mapping_name}")
            return None

        except Exception as e:
            print(f"Warning: Could not construct CRS from CF attributes: {str(e)}")
            return None

    def list_cached_extractions(self) -> pd.DataFrame:
        """List all cached extractions with summary information"""

        cache_files = list(self.metadata_dir.glob("extract_*.json"))

        if not cache_files:
            print("No cached extractions found")
            return pd.DataFrame()

        summaries = []

        for cache_file in cache_files:
            with open(cache_file, "r") as f:
                info = json.load(f)

            # Handle both old and new metadata structures
            n_chunks = len(info.get("country_chunks", info.get("spatial_chunks", [])))

            summary = {
                "cache_id": info["cache_id"],
                "n_polygons": info["n_polygons"],
                "variable": info["variable"],
                "statistics": ", ".join(info.get("statistics", ["unknown"])),
                "time_range": f"{info['time_range']['start'][:10]} to {info['time_range']['end'][:10]}",
                "n_timesteps": info["time_range"]["n_timesteps"],
                "n_chunks": n_chunks,
                "cache_file": cache_file.name,
            }
            summaries.append(summary)

        return pd.DataFrame(summaries)

    def _convert_numpy_types(self, obj):
        """Convert numpy types to native Python types for JSON serialization (same as point extractor)"""
        if isinstance(obj, dict):
            converted_dict = {}
            for k, v in obj.items():
                if isinstance(k, (np.integer, np.int8, np.int16, np.int32, np.int64)):
                    key = int(k)
                elif isinstance(k, (np.floating, np.float16, np.float32, np.float64)):
                    key = float(k)
                elif isinstance(k, np.bool_):
                    key = bool(k)
                elif hasattr(k, "item") and not isinstance(k, (list, dict, str)):
                    try:
                        key = k.item()
                    except (ValueError, AttributeError):
                        key = str(k)
                else:
                    key = k

                converted_dict[key] = self._convert_numpy_types(v)
            return converted_dict
        elif isinstance(obj, list):
            return [self._convert_numpy_types(i) for i in obj]
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.integer, np.int8, np.int16, np.int32, np.int64)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float16, np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, np.bool_):
            return bool(obj)
        elif hasattr(obj, "item") and not isinstance(obj, (list, dict, str)):
            try:
                return obj.item()
            except (ValueError, AttributeError):
                return obj
        else:
            return obj
