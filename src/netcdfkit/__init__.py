"""
NetCDFKit - High-Performance NetCDF Data Extraction Toolkit

A comprehensive Python library for efficient extraction of time series data
from large NetCDF files at specific geographic points and polygon regions.
"""

from .ncPntExtractor import NetCDFPointExtractor
from .ncPolyExtraction import NetCDFPolygonExtractor

__version__ = "0.1.1"
__author__ = "Muhammad Shafeeque"
__email__ = "muhammad.shafeeque@awi.de | shafeequ@uni-bremen.de"

__all__ = [
    "NetCDFPointExtractor",
    "NetCDFPolygonExtractor",
]
