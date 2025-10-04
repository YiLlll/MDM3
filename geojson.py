import geopandas as gpd
import matplotlib.pyplot as plt
from pathlib import Path

# Load GeoJSON file
file_path = Path(__file__).parent / "roads.geojson.json"
gdf = gpd.read_file(file_path)

# Print basic data information
print("Data Overview:")
print(f"- Coordinate System: {gdf.crs}")
print(f"- Geometry Types: {gdf.geom_type.unique().tolist()}")
print(f"- Bounds: {gdf.total_bounds}")
print(f"- Feature Count: {len(gdf)}")

# Plot the data
fig, ax = plt.subplots(figsize=(10, 10))
gdf.plot(ax=ax, linewidth=0.5, color="blue")
ax.set_title("Roads GeoJSON Data", fontsize=14)
ax.set_xlabel("Longitude")
ax.set_ylabel("Latitude")
plt.show()


