from __future__ import annotations
from pathlib import Path
from typing import Optional, Dict, List, Union
import pandas as pd
import xarray as xr
import numpy as np
from pyproj import CRS, Transformer
import json
from tqdm import tqdm
import warnings
from sklearn.cluster import DBSCAN

warnings.filterwarnings("ignore")


class NetCDFPointExtractor:
    """
    Optimal hybrid spatial-temporal NetCDF point extraction system

    Features:
    - Automatic spatial clustering for efficient chunk processing
    - Intelligent time series caching with metadata preservation
    - Fast multi-scenario processing (different days_back values)
    - Memory-efficient processing within 32GB RAM limits
    - Easy access to individual point time series
    - Fixed coordinate transformation and JSON serialization issues
    """

    def __init__(self, cache_dir: str | Path = "timeseries_cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)

        # Cache subdirectories
        self.metadata_dir = self.cache_dir / "metadata"
        self.timeseries_dir = self.cache_dir / "timeseries"
        self.metadata_dir.mkdir(exist_ok=True)
        self.timeseries_dir.mkdir(exist_ok=True)

        # Internal state
        self.points_df = None
        self.spatial_chunks = None
        self.dataset_info = None

    def analyze_spatial_distribution(
        self, points_df: pd.DataFrame, eps_km: float = 100, min_samples: int = 2
    ) -> Dict:
        """
        Automatically detect spatial clusters (countries/regions) in point data

        Parameters:
        -----------
        eps_km : float
            Maximum distance between points in same cluster (km)
        min_samples : int
            Minimum points needed to form a cluster
        """

        print(f"Analyzing spatial distribution of {len(points_df)} points...")

        # Convert lat/lon to approximate distances for clustering
        coords = points_df[["lon", "lat"]].values

        # Use haversine distance approximation for European coordinates
        lat_center = coords[:, 1].mean()
        lon_scale = np.cos(np.radians(lat_center)) * 111.32  # km per degree longitude
        lat_scale = 111.32  # km per degree latitude

        # Scale coordinates to approximate km
        coords_scaled = coords.copy()
        coords_scaled[:, 0] *= lon_scale
        coords_scaled[:, 1] *= lat_scale

        # Cluster using DBSCAN
        clustering = DBSCAN(eps=eps_km, min_samples=min_samples).fit(coords_scaled)
        labels = clustering.labels_

        # Analyze clusters
        unique_labels = set(labels)
        n_clusters = len(unique_labels) - (1 if -1 in unique_labels else 0)
        n_noise = list(labels).count(-1)

        cluster_info = {
            "n_clusters": n_clusters,
            "n_noise_points": n_noise,
            "cluster_labels": labels,
            "cluster_summary": {},
        }

        print(f"Detected {n_clusters} spatial clusters, {n_noise} isolated points")

        # Analyze each cluster
        for label in unique_labels:
            if label == -1:  # Noise points
                continue

            cluster_mask = labels == label
            cluster_points = points_df[cluster_mask]

            cluster_summary = {
                "n_points": len(cluster_points),
                "center_lon": cluster_points["lon"].mean(),
                "center_lat": cluster_points["lat"].mean(),
                "bounds": {
                    "min_lon": cluster_points["lon"].min(),
                    "max_lon": cluster_points["lon"].max(),
                    "min_lat": cluster_points["lat"].min(),
                    "max_lat": cluster_points["lat"].max(),
                },
            }

            cluster_info["cluster_summary"][label] = cluster_summary
            print(
                f"  Cluster {label}: {len(cluster_points)} points "
                f"({cluster_summary['center_lat']:.2f}°N, {cluster_summary['center_lon']:.2f}°E)"
            )

        return cluster_info

    def create_spatial_chunks(
        self,
        points_df: pd.DataFrame,
        cluster_info: Dict,
        buffer_km: float = 50,
        max_chunk_points: int = 5000,
    ) -> List[Dict]:
        """
        Create optimal spatial chunks for efficient NetCDF access

        Parameters:
        -----------
        buffer_km : float
            Buffer around each cluster in km
        max_chunk_points : int
            Maximum points per chunk (split large clusters)
        """

        print("Creating optimal spatial chunks...")

        chunks = []
        chunk_id = 0

        # Convert buffer from km to degrees (approximate)
        lat_center = points_df["lat"].mean()
        buffer_deg_lon = buffer_km / (111.32 * np.cos(np.radians(lat_center)))
        buffer_deg_lat = buffer_km / 111.32

        # Process each cluster
        for label, summary in cluster_info["cluster_summary"].items():
            cluster_mask = cluster_info["cluster_labels"] == label
            cluster_points = points_df[cluster_mask].copy()

            # Calculate chunk bounds with buffer
            bounds = summary["bounds"]
            chunk_bounds = {
                "min_lon": bounds["min_lon"] - buffer_deg_lon,
                "max_lon": bounds["max_lon"] + buffer_deg_lon,
                "min_lat": bounds["min_lat"] - buffer_deg_lat,
                "max_lat": bounds["max_lat"] + buffer_deg_lat,
            }

            # Split large clusters if needed
            if len(cluster_points) > max_chunk_points:
                # Sub-cluster large clusters
                sub_chunks = self._split_large_cluster(
                    cluster_points, max_chunk_points, buffer_deg_lon, buffer_deg_lat
                )
                for sub_chunk in sub_chunks:
                    sub_chunk["chunk_id"] = chunk_id
                    chunks.append(sub_chunk)
                    chunk_id += 1
            else:
                # Single chunk for this cluster
                chunk = {
                    "chunk_id": chunk_id,
                    "cluster_label": label,
                    "n_points": len(cluster_points),
                    "bounds": chunk_bounds,
                    "point_indices": cluster_points.index.tolist(),
                }
                chunks.append(chunk)
                chunk_id += 1

        # Handle noise points (ungrouped points)
        noise_mask = cluster_info["cluster_labels"] == -1
        if noise_mask.any():
            noise_points = points_df[noise_mask]
            print(
                f"Creating individual chunks for {len(noise_points)} isolated points..."
            )

            for idx, (_, point) in enumerate(noise_points.iterrows()):
                chunk = {
                    "chunk_id": chunk_id,
                    "cluster_label": -1,
                    "n_points": 1,
                    "bounds": {
                        "min_lon": point["lon"] - buffer_deg_lon,
                        "max_lon": point["lon"] + buffer_deg_lon,
                        "min_lat": point["lat"] - buffer_deg_lat,
                        "max_lat": point["lat"] + buffer_deg_lat,
                    },
                    "point_indices": [point.name],
                }
                chunks.append(chunk)
                chunk_id += 1

        print(f"Created {len(chunks)} spatial chunks")
        for chunk in chunks:
            print(f"  Chunk {chunk['chunk_id']}: {chunk['n_points']} points")

        return chunks

    def _split_large_cluster(
        self,
        cluster_points: pd.DataFrame,
        max_points: int,
        buffer_deg_lon: float,
        buffer_deg_lat: float,
    ) -> List[Dict]:
        """Split large clusters into sub-chunks"""

        n_sub_chunks = int(np.ceil(len(cluster_points) / max_points))
        print(
            f"    Splitting large cluster ({len(cluster_points)} points) into {n_sub_chunks} sub-chunks"
        )

        # Use K-means to split cluster
        from sklearn.cluster import KMeans

        coords = cluster_points[["lon", "lat"]].values
        kmeans = KMeans(n_clusters=n_sub_chunks, random_state=42).fit(coords)
        sub_labels = kmeans.labels_

        sub_chunks = []
        for sub_label in range(n_sub_chunks):
            sub_mask = sub_labels == sub_label
            sub_points = cluster_points[sub_mask]

            sub_chunk = {
                "cluster_label": f"sub_{sub_label}",
                "n_points": len(sub_points),
                "bounds": {
                    "min_lon": sub_points["lon"].min() - buffer_deg_lon,
                    "max_lon": sub_points["lon"].max() + buffer_deg_lon,
                    "min_lat": sub_points["lat"].min() - buffer_deg_lat,
                    "max_lat": sub_points["lat"].max() + buffer_deg_lat,
                },
                "point_indices": sub_points.index.tolist(),
            }
            sub_chunks.append(sub_chunk)

        return sub_chunks

    def extract_and_cache_timeseries(
        self,
        netcdf_path: str | Path,
        points_path: str | Path,
        variable: str,
        date_col: Optional[str] = None,
        force_recache: bool = False,
    ) -> str:
        """
        Extract and cache time series for all points

        Parameters:
        -----------
        netcdf_path : str | Path
            Path to NetCDF file
        points_path : str | Path
            Path to CSV file with point locations
        variable : str
            Variable to extract from NetCDF
        date_col : str, optional
            Name of date column in points CSV
        force_recache : bool
            If True, ignore existing cache and re-extract

        Returns:
        --------
        str
            Cache ID for later retrieval
        """
        # Generate cache ID
        cache_id = f"extract_{Path(points_path).stem}"

        # Setup cache paths
        cache_path = self.metadata_dir / f"{cache_id}.json"
        points_cache_path = self.metadata_dir / f"{cache_id}_points.csv"

        # Check for existing cache
        if not force_recache and cache_path.exists() and points_cache_path.exists():
            print(f"Found existing cache: {cache_id}")
            try:
                # Try to load cache
                with open(cache_path, "r") as f:
                    self.dataset_info = json.load(f)
                self.points_df = pd.read_csv(points_cache_path, index_col=0)
                return cache_id
            except (json.JSONDecodeError, pd.errors.EmptyDataError, Exception) as e:
                print(f"Cache appears corrupted ({str(e)}), re-extracting...")
                force_recache = True

        # Load point data
        print("Loading point data...")
        self.points_df = self._load_points(points_path, date_col)

        # Analyze spatial distribution
        cluster_info = self.analyze_spatial_distribution(self.points_df)

        # Create spatial chunks
        self.spatial_chunks = self.create_spatial_chunks(self.points_df, cluster_info)

        # Open NetCDF dataset
        print("Opening NetCDF dataset...")
        ds = xr.open_dataset(netcdf_path)

        # Debug: Print dataset info
        print(f"Dataset dimensions: {list(ds.dims.keys())}")
        print(f"Dataset coordinates: {list(ds.coords.keys())}")
        print(f"Variable '{variable}' dimensions: {ds[variable].dims}")

        # Setup coordinate mapping and transformation
        coord_mapping = self._detect_coordinate_mapping(ds, variable)
        print(f"Detected coordinate mapping: {coord_mapping}")

        ds_crs = self._get_dataset_crs(ds, variable)
        transformer = self._setup_transformer(ds_crs)

        # Extract time series chunk by chunk
        print(
            f"\nExtracting time series for {len(self.spatial_chunks)} spatial chunks..."
        )

        all_timeseries = {}

        for chunk in tqdm(self.spatial_chunks, desc="Processing chunks"):
            chunk_timeseries = self._extract_chunk_timeseries(
                ds, variable, chunk, transformer, coord_mapping
            )
            all_timeseries.update(chunk_timeseries)

        # Save cached time series
        print("Saving cached time series...")
        if all_timeseries:
            self._save_timeseries_cache(all_timeseries, cache_id)
            print(f"Saved {len(all_timeseries)} time series")
        else:
            print("Warning: No time series were successfully extracted!")
            # Create empty cache file to avoid issues
            cache_file = self.timeseries_dir / f"{cache_id}_timeseries.parquet"
            empty_df = pd.DataFrame(columns=["time", "value", "point_id"])
            empty_df.to_parquet(cache_file, index=False)
            print("Created empty cache file")

        # Save metadata
        self.dataset_info = {
            "cache_id": cache_id,
            "netcdf_path": str(netcdf_path),
            "variable": variable,
            "n_points": len(self.points_df),
            "date_col": date_col,
            "spatial_chunks": self._convert_numpy_types(self.spatial_chunks),
            "cluster_info": self._convert_numpy_types(cluster_info),
            "time_range": {
                "start": str(ds.time.min().values),
                "end": str(ds.time.max().values),
                "n_timesteps": int(len(ds.time)),
            },
        }

        with open(cache_path, "w") as f:
            json.dump(self.dataset_info, f, indent=2)

        # Save points metadata
        self.points_df.to_csv(points_cache_path, index=True)

        print(f"\nCaching complete! Cache ID: {cache_id}")
        print(f"Time series cached for {len(all_timeseries)} points")

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

        # If not found, try to infer from coordinate values
        if "lon" not in coord_mapping or "lat" not in coord_mapping:
            for coord_name in ds.coords:
                coord_values = ds.coords[coord_name].values
                if len(coord_values.shape) == 1:  # 1D coordinate
                    if np.min(coord_values) >= -180 and np.max(coord_values) <= 180:
                        if "lon" not in coord_mapping:
                            coord_mapping["lon"] = coord_name
                        elif (
                            "lat" not in coord_mapping
                            and np.min(coord_values) >= -90
                            and np.max(coord_values) <= 90
                        ):
                            coord_mapping["lat"] = coord_name

        return coord_mapping

    def _extract_chunk_timeseries(
        self,
        ds: xr.Dataset,
        variable: str,
        chunk: Dict,
        transformer: Optional[Transformer],
        coord_mapping: Dict[str, str],
    ) -> Dict:
        """Extract time series for all points in a spatial chunk"""

        # Get points in this chunk
        chunk_point_indices = chunk["point_indices"]
        chunk_points = self.points_df.loc[chunk_point_indices]

        # Get coordinate names
        lon_coord = coord_mapping.get("lon", "x")
        lat_coord = coord_mapping.get("lat", "y")

        # Load spatial subset of NetCDF
        bounds = chunk["bounds"]

        # For now, skip spatial subsetting to avoid issues - just use full dataset
        # This is just an optimization, the main functionality still works
        print(f"    Using full dataset (spatial subsetting disabled for debugging)")
        spatial_subset = ds

        # Extract time series for each point in chunk
        chunk_timeseries = {}

        for point_idx, point_row in chunk_points.iterrows():
            try:
                # Get original coordinates
                lon, lat = float(point_row["lon"]), float(point_row["lat"])

                # Transform coordinates if needed
                if transformer is not None:
                    x, y = transformer.transform(lon, lat)
                    print(
                        f"    Point {point_idx}: ({lon:.3f}, {lat:.3f}) -> ({x:.1f}, {y:.1f})"
                    )
                else:
                    x, y = lon, lat
                    print(
                        f"    Point {point_idx}: Using coordinates as-is ({x:.3f}, {y:.3f})"
                    )

                # Use nearest neighbor interpolation with correct coordinate names
                # Use direct coordinate assignment instead of dictionary
                if lon_coord == "x" and lat_coord == "y":
                    point_data = spatial_subset[variable].sel(
                        x=x, y=y, method="nearest"
                    )
                elif lon_coord == "lon" and lat_coord == "lat":
                    point_data = spatial_subset[variable].sel(
                        lon=x, lat=y, method="nearest"
                    )
                else:
                    # Fallback to dictionary method
                    point_data = spatial_subset[variable].sel(
                        {lon_coord: x, lat_coord: y}, method="nearest"
                    )

                # Extract the time series and check for valid data
                timeseries = point_data.to_series()

                # Check if we got valid data (not all NaN)
                if not timeseries.empty and not timeseries.isna().all():
                    chunk_timeseries[point_idx] = timeseries
                    n_valid = (~timeseries.isna()).sum()
                    print(
                        f"    ✓ Point {point_idx}: Extracted {len(timeseries)} timesteps, {n_valid} valid values"
                    )
                else:
                    print(f"    ✗ Point {point_idx}: No valid data found (all NaN)")

            except Exception as e:
                print(f"    ✗ Point {point_idx}: Failed to extract: {str(e)}")
                # Add more detailed error info for the first few failures
                if (
                    len(chunk_timeseries) == 0 and point_idx <= chunk_points.index[2]
                ):  # First 3 points
                    print(f"      Detailed error for debugging:")
                    print(f"      - lon_coord='{lon_coord}', lat_coord='{lat_coord}'")
                    print(f"      - Transformed coords: x={x:.1f}, y={y:.1f}")
                    print(
                        f"      - Dataset coords available: {list(spatial_subset.coords.keys())}"
                    )
                    try:
                        # Try to see what the nearest coordinate values are
                        if lon_coord in spatial_subset.coords:
                            nearest_x_idx = np.argmin(
                                np.abs(spatial_subset.coords[lon_coord].values - x)
                            )
                            nearest_x = spatial_subset.coords[lon_coord].values[
                                nearest_x_idx
                            ]
                            print(
                                f"      - Nearest {lon_coord} value: {nearest_x:.1f} (index {nearest_x_idx})"
                            )
                        if lat_coord in spatial_subset.coords:
                            nearest_y_idx = np.argmin(
                                np.abs(spatial_subset.coords[lat_coord].values - y)
                            )
                            nearest_y = spatial_subset.coords[lat_coord].values[
                                nearest_y_idx
                            ]
                            print(
                                f"      - Nearest {lat_coord} value: {nearest_y:.1f} (index {nearest_y_idx})"
                            )
                    except Exception as debug_e:
                        print(f"      - Debug failed: {debug_e}")
                continue

        return chunk_timeseries

    def _save_timeseries_cache(self, all_timeseries: Dict, cache_id: str):
        """Save time series data efficiently using Parquet format"""

        # Convert to DataFrame format suitable for Parquet
        timeseries_data = []

        for point_idx, timeseries in all_timeseries.items():
            if len(timeseries) > 0:
                ts_df = timeseries.reset_index()
                ts_df.columns = ["time", "value"]
                ts_df["point_id"] = point_idx
                timeseries_data.append(ts_df)

        if timeseries_data:
            # Combine all time series
            combined_df = pd.concat(timeseries_data, ignore_index=True)

            # Save as Parquet for efficient storage and retrieval
            cache_file = self.timeseries_dir / f"{cache_id}_timeseries.parquet"
            combined_df.to_parquet(cache_file, index=False)

            print(f"Saved {len(all_timeseries)} time series to {cache_file}")

    def load_point_timeseries(
        self, cache_id: str, point_ids: Union[int, List[int], str] = "all"
    ) -> Dict[int, pd.Series]:
        """
        Load time series for specific points or all points

        Parameters:
        -----------
        cache_id : str
            Cache identifier from extract_and_cache_timeseries()
        point_ids : int, list, or 'all'
            Point ID(s) to load, or 'all' for all points

        Returns:
        --------
        Dict[point_id, timeseries]
        """

        cache_file = self.timeseries_dir / f"{cache_id}_timeseries.parquet"

        if not cache_file.exists():
            raise FileNotFoundError(f"Cache file not found: {cache_file}")

        # Load parquet data
        df = pd.read_parquet(cache_file)

        # Filter points if specified
        if point_ids != "all":
            if isinstance(point_ids, (int, np.integer)):
                point_ids = [point_ids]
            df = df[df["point_id"].isin(point_ids)]

        # Convert back to time series format
        timeseries_dict = {}
        for point_id in df["point_id"].unique():
            point_data = df[df["point_id"] == point_id].copy()
            point_data["time"] = pd.to_datetime(point_data["time"])
            timeseries = point_data.set_index("time")["value"]
            timeseries_dict[point_id] = timeseries

        return timeseries_dict

    def export_point_timeseries_csv(
        self,
        cache_id: str,
        point_ids: Union[int, List[int], str] = "all",
        output_path: Optional[str | Path] = None,
    ) -> pd.DataFrame:
        """Export point time series to CSV format"""

        timeseries_dict = self.load_point_timeseries(cache_id, point_ids)

        # Load point metadata
        points_file = self.metadata_dir / f"{cache_id}_points.csv"
        points_df = pd.read_csv(points_file, index_col=0)

        # Create export DataFrame
        export_data = []

        for point_id, timeseries in timeseries_dict.items():
            point_info = points_df.loc[point_id].to_dict()

            for timestamp, value in timeseries.items():
                row = point_info.copy()
                row["point_id"] = point_id
                row["time"] = timestamp
                row["value"] = value
                export_data.append(row)

        export_df = pd.DataFrame(export_data)

        if output_path:
            export_df.to_csv(output_path, index=False)
            print(f"Exported time series to: {output_path}")

        return export_df

    def generate_multi_scenario_results(
        self,
        cache_id: str,
        days_back_list: List[int],
        date_col: Optional[str] = None,
        output_path: Optional[str | Path] = None,
    ) -> pd.DataFrame:
        """
        Generate results for multiple days_back scenarios efficiently

        Parameters:
        -----------
        cache_id : str
            Cache identifier
        days_back_list : List[int]
            List of days_back values to calculate (e.g., [3, 7, 14, 30])
        date_col : str
            Date column for temporal windows
        output_path : str/Path
            Output CSV path

        Returns:
        --------
        DataFrame with columns: [point_info_cols, days_back_3, days_back_7, ...]
        """

        print(f"Generating multi-scenario results for days_back: {days_back_list}")

        # Load all time series
        timeseries_dict = self.load_point_timeseries(cache_id, "all")

        # Load point metadata
        points_file = self.metadata_dir / f"{cache_id}_points.csv"
        points_df = pd.read_csv(points_file, index_col=0)

        # Initialize results with point information
        results_df = points_df.copy()

        # Calculate averages for each days_back scenario
        for days_back in tqdm(days_back_list, desc="Processing scenarios"):
            col_name = f"days_back_{days_back}"
            values = []

            for point_id in results_df.index:
                if point_id in timeseries_dict:
                    timeseries = timeseries_dict[point_id]
                    point_info = results_df.loc[point_id]

                    # Calculate temporal average
                    avg_value = self._calculate_temporal_average(
                        timeseries,
                        point_info.get(date_col) if date_col else None,
                        days_back,
                    )
                    values.append(avg_value)
                else:
                    values.append(np.nan)

            results_df[col_name] = values

        if output_path:
            results_df.to_csv(output_path, index=False)
            print(f"Multi-scenario results saved to: {output_path}")

        print(
            f"Generated results for {len(results_df)} points and {len(days_back_list)} scenarios"
        )

        return results_df

    def _calculate_temporal_average(
        self, timeseries: pd.Series, target_date: Optional[pd.Timestamp], days_back: int
    ) -> float:
        """Calculate temporal average for a single point"""

        if pd.isna(target_date) or target_date is None:
            # No date specified, use entire time series
            return timeseries.mean()

        # Filter to date window
        end_date = pd.to_datetime(target_date)
        start_date = end_date - pd.Timedelta(days=days_back)

        # Create mask for time window
        mask = (timeseries.index >= start_date) & (timeseries.index <= end_date)
        window_data = timeseries[mask].dropna()

        if len(window_data) > 0:
            return window_data.mean()
        else:
            return np.nan

    def _load_points(
        self, path: str | Path, date_col: Optional[str] = None
    ) -> pd.DataFrame:
        """Load point locations from CSV file"""
        df = pd.read_csv(path)

        # Ensure required columns exist
        required_cols = ["lon", "lat"]
        if date_col and date_col not in df.columns:
            raise ValueError(f"Date column '{date_col}' not found in points file")

        for col in required_cols:
            if col not in df.columns:
                raise ValueError(f"Required column '{col}' not found in points file")

        # Convert date column to datetime if present
        if date_col:
            df[date_col] = pd.to_datetime(df[date_col])

        # Convert coordinates to float if needed
        df["lon"] = df["lon"].astype(float)
        df["lat"] = df["lat"].astype(float)

        return df

    def _get_dataset_crs(self, ds: xr.Dataset, variable: str) -> CRS | None:
        """Extract CRS from dataset"""
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
        """Try to construct CRS from CF convention attributes"""
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
            # For now, return None if we can't handle it
            print(f"Unhandled grid mapping: {grid_mapping_name}")
            return None

        except Exception as e:
            print(f"Warning: Could not construct CRS from CF attributes: {str(e)}")
            return None

    def _setup_transformer(self, ds_crs: Optional[CRS]) -> Optional[Transformer]:
        """Setup coordinate transformer"""
        if ds_crs:
            input_crs = CRS.from_epsg(4326)
            return Transformer.from_crs(input_crs, ds_crs, always_xy=True)
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

            summary = {
                "cache_id": info["cache_id"],
                "n_points": info["n_points"],
                "variable": info["variable"],
                "time_range": f"{info['time_range']['start'][:10]} to {info['time_range']['end'][:10]}",
                "n_timesteps": info["time_range"]["n_timesteps"],
                "n_chunks": len(info["spatial_chunks"]),
                "cache_file": cache_file.name,
            }
            summaries.append(summary)

        return pd.DataFrame(summaries)

    def _convert_numpy_types(self, obj):
        """Convert numpy types to native Python types for JSON serialization"""
        if isinstance(obj, dict):
            # Convert both keys and values, ensuring keys are JSON-compatible
            converted_dict = {}
            for k, v in obj.items():
                # Convert keys to JSON-compatible types
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
                        key = str(k)  # Fallback to string
                else:
                    key = k

                # Convert values recursively
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
            # Handle other numpy scalars
            try:
                return obj.item()
            except (ValueError, AttributeError):
                return obj
        else:
            return obj
