import unittest
from netcdfkit.nc_point import NetCDFPointExtractor

class TestNetCDFPointExtractor(unittest.TestCase):
    def test_init(self):
        extractor = NetCDFPointExtractor()
        self.assertIsNotNone(extractor)

if __name__ == '__main__':
    unittest.main()
