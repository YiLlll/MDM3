import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import warnings
import pandas as pd
from shapely.geometry import Point
import math
import networkx as nx
from sklearn.cluster import KMeans
from pyproj import Transformer
from geopy.distance import geodesic

# Suppress font warnings
warnings.filterwarnings('ignore', category=UserWarning, module='matplotlib')

def haversine(lat1, lon1, lat2, lon2):
    R = 6371.0088  # Mean Earth radius in km (more precise)

    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)

    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    a = min(1.0, max(0.0, a))
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

    return R * c

def plot_area6_map():
    """Plot Area 6 road network map"""
    print("Loading Area 6 route data...")
    
    try:
        # Load GeoJSON file
        gdf = gpd.read_file(GEOJSON_PATH)
        print(f"Successfully loaded {len(gdf)} features")
        
        # Set font to avoid Chinese character issues
        plt.rcParams['font.family'] = 'DejaVu Sans'
        
        # Create the plot
        fig, ax = plt.subplots(figsize=(12, 10))
        
        # Get unique routes and assign colors
        unique_routes = gdf['route_name'].unique()
        colors = plt.cm.tab20(np.linspace(0, 1, len(unique_routes)))
        
        # Plot all routes
        for i, route in enumerate(unique_routes):
            route_data = gdf[gdf['route_name'] == route]
            route_data.plot(ax=ax, color=colors[i], linewidth=1.5, alpha=0.8)
        
        # Set plot properties
        ax.set_title("Area 6 Road Network", fontsize=16, fontweight='bold')
        ax.set_xlabel("Easting (m)")
        ax.set_ylabel("Northing (m)")
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal')
        
        # Adjust layout and show
        plt.tight_layout()
        # plt.show()
        
        print(f"Map plotting completed. Total routes: {len(unique_routes)}")
        
    except FileNotFoundError:
        print(f"Error: File not found - {GEOJSON_PATH}")
    except Exception as e:
        print(f"Error: {e}")


def depot_loc(zones):
    df = pd.read_excel("List of depot details.xlsx")
    filtered = df[(df["Area"].isin(zones)) & (df["Operational?"]=='Y')]

    transformer = Transformer.from_crs("epsg:4326", "epsg:27700", always_xy=True)
    
    filtered["Easting"], filtered["Northing"] = transformer.transform(
        filtered["Longitude"].values, 
        filtered["Latitude"].values
    )

    return filtered

# Configuration
SCRIPT_DIR = Path(__file__).parent
GEOJSON_PATH = SCRIPT_DIR / "area6_combined_routes.geojson"

def read_route_data():
    routes = []

    gdf = gpd.read_file(GEOJSON_PATH)

    unique_routes = gdf['route_name'].unique()

    for route in unique_routes:
        route_data = gdf[gdf['route_name'] == route].reset_index(drop=True)
        
        min = route_data.index[route_data["TREATMENT"] != "Free travel"].min()
        max = route_data.index[route_data["TREATMENT"] != "Free travel"].max()

        area = route_data.iloc[min:max+1]
        distance = area["DISTANCE"].sum()

        # lan lon where the truck starts salting
        geom = route_data.iloc[min]["geometry"]
        line = list(geom.coords)[0]

        point_gdf = gpd.GeoDataFrame(geometry=[Point(line)], crs=gdf.crs)
        point_wgs84 = point_gdf.to_crs(epsg=4326)
        lon, lat = point_wgs84.geometry.iloc[0].coords[0]

        # print(f"Latitude: {lat}, Longitude: {lon}")

        # adding the difference from start and stop salting
        geom = route_data.iloc[max]["geometry"]
        line_1 = list(geom.coords)[0]

        point_gdf = gpd.GeoDataFrame(geometry=[Point(line)], crs=gdf.crs)
        point_wgs84 = point_gdf.to_crs(epsg=4326)
        lon_e, lat_e = point_wgs84.geometry.iloc[0].coords[0]

        diff = haversine(lat, lon, lat_e, lon_e)
        distance = round(distance+diff, 2)

        routes.append({"x":line[0], "y":line[1], "x2":line_1[0], "y2":line_1[1], "time":distance/60})

    return routes

dep = depot_loc([6])
route = read_route_data()

def plot_routes(routes):
    for route in routes:
        plt.scatter(route["x"], route["y"], c='blue', marker='o')
        plt.scatter(route["x2"], route["y2"], c='blue', marker='o')

plot_area6_map()
plot_routes(route)
plt.scatter(dep[["Easting"]], dep[["Northing"]], c='red', marker='x')

plt.show()


# Example route and depots
# route = [{"x": ..., "y": ..., "distance": ...}, ...]
# dep = [...]  # initial depot locations if any

k = len(dep)  # number of clusters (one truck per cluster)

# -----------------------------
# STEP 1: Prepare data for clustering
# -----------------------------
X = np.array([[r["x"], r["y"]] for r in route])

# -----------------------------
# STEP 2: Cluster routes (one truck per cluster)
# -----------------------------
kmeans = KMeans(n_clusters=k, init="k-means++", random_state=42)
labels = kmeans.fit_predict(X)
centers = kmeans.cluster_centers_

for i, r in enumerate(route):
    r["cluster"] = int(labels[i])

# -----------------------------
# STEP 3: Build road network
# -----------------------------
G = nx.Graph()
for i, r in enumerate(route):
    G.add_node(i, pos=(r["x"], r["y"]))

# Travel parameters
AVG_SPEED_KMH = 50
MAX_TRAVEL_HOURS = 2

def distance_to_time(distance_km):
    """Convert distance in km to hours."""
    return distance_km / AVG_SPEED_KMH

# Add edges between sequential route points
for i in range(len(route) - 1):
    travel_time = route[i]["time"]
    G.add_edge(i, i + 1, time=travel_time)

# -----------------------------
# STEP 4: Per-truck travel time calculation
# -----------------------------
truck_times = []

for c in range(k):
    cluster_nodes = [i for i, r in enumerate(route) if r["cluster"] == c]
    
    center = centers[c]

    if not cluster_nodes:
        # print(f"⚠️ Cluster {c} has no assigned roads — skipping.")
        continue

    # Build subgraph for this truck's cluster
    subG = G.subgraph(cluster_nodes)
    total_time = 0.0

    route_times = []

    # Sum all edge travel times within the cluster
    for (u, v, data) in subG.edges(data=True):
        # Get node coordinates (projected: e.g., meters)
        x_u, y_u = G.nodes[u]["pos"]
        x_v, y_v = G.nodes[v]["pos"]

        # Compute midpoint of the edge (approx. route location)
        mid_x = (x_u + x_v) / 2
        mid_y = (y_u + y_v) / 2

        # Calculate Euclidean distance from edge midpoint to cluster center
        dist_meters = np.sqrt((mid_x - center[0])**2 + (mid_y - center[1])**2)

        # Convert meters --> miles
        dist_miles = dist_meters / 1609.34

        # Convert distance to hours (assuming 50 mph)
        time_to_center = dist_miles / 50.0

        # Combine route's travel time with travel time to center
        times = data["time"] + time_to_center
        route_times.append(times)

    # Optional: Add round trip (back to start)
    # total_time_round_trip = total_time
    
    # 🩹 SAFETY CHECK: skip clusters with no edges
    if not route_times:
        # print(f"⚠️ Cluster {c} has no connected edges ~ skipping time calculation.")
        continue

    truck_times.append({
        "cluster": c,
        "number of routes": len(cluster_nodes),
        "average time": sum(route_times)/len(route_times),
        "max time": max(route_times)
    })

# -----------------------------
# STEP 5: Output per-truck summary
# -----------------------------
print("Per-Truck (Cluster) Travel Time Summary:")

for t in truck_times:
    print(f"Cluster {t['cluster']}: "
          f"{t['number of routes']} number of route(s), "
          f"average time = {t['average time']:.2f}h, "
          f"max time = {t['max time']:.2f}h")

# -----------------------------
# STEP 6: Visualization (optional)
# -----------------------------
plt.figure(figsize=(8, 6))
for c in range(k):
    cluster_points = np.array([[r["x"], r["y"]] for r in route if r["cluster"] == c])
    if cluster_points.size == 0:
        continue
    plt.scatter(cluster_points[:, 0], cluster_points[:, 1], label=f'Truck {c+1}', s=60)

plt.scatter(centers[:, 0], centers[:, 1], marker='*', c='black', s=200, label='Best depot location')

pos = nx.get_node_attributes(G, 'pos')
nx.draw_networkx_edges(G, pos, alpha=0.3, width=1.0)

plt.title("Truck Routes (Per-Cluster Travel Time in Hours)")
plt.xlabel("Longitude (x)")
plt.ylabel("Latitude (y)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()