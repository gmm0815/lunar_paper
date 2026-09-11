import rasterio

with rasterio.open("Lunar_LRO_LOLA_Global_LDEM_118m_Mar2014.tif") as dem:
    print("CRS:", dem.crs)
    print("Is geographic (lat/lon degrees):", dem.crs.is_geographic)
    print("Is projected (x/y in meters):", dem.crs.is_projected)
    print("Bounds:", dem.bounds)
    print("Transform:", dem.transform)