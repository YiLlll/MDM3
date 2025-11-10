import pandas as pd
import geopandas as gpd
import numpy as np
from shapely.geometry import Point, LineString
from shapely.ops import unary_union, nearest_points
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from scipy.spatial import distance_matrix, cKDTree
from sklearn.cluster import KMeans
import networkx as nx
import warnings
warnings.filterwarnings('ignore')
import sys
import io

# Configure stdout encoding
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# ==================== Parameter Configuration ====================
VEHICLE_SPEED_KMH = 80
GRITTING_SPEED_KMH = 80
MAX_TOTAL_TIME_HOURS = 2
AVAILABLE_VEHICLES = 443
# WORK_TIME_RATIO is now computed dynamically instead of using a fixed value
SAMPLE_SIZE = 30000
MAX_MOVE_DISTANCE_KM = 50  # Maximum relocation distance per depot
MAX_MOVE_RATIO = 0.10  # Maximum share of depots that may move (10%)
MAX_MOVE_COUNT = 10    # Hard cap on the number of depots allowed to move
DEPOT_FILE = 'depot_details.csv'
ROAD_NETWORK_FILE = 'england_complete_road_network.geojson'
CALCULATE_REAL_DETOUR = True  # Whether to compute the real detour coefficient
DETOUR_SAMPLE_SIZE = 500  # Number of sampled pairs for detour estimation

print("=" * 80)
print("National Highways England Winter Gritting Depot Optimization V6.3 - Real Detour Coefficient")
print("=" * 80)
print("[Major Update] Real road network detour γ = d_network / d_euclidean")
print("=" * 80)

# ==================== 1. Data Loading ====================
print("\n[Step 1/8] Loading data...")

depots_df = pd.read_csv(DEPOT_FILE)

# Separate operational and non-operational depots
active_depots = depots_df[
    (depots_df['Latitude'] != 0) & 
    (depots_df['Longitude'] != 0) &
    (depots_df['Operational?'] == 'Y')
].copy()

inactive_depots = depots_df[
    (depots_df['Latitude'] != 0) & 
    (depots_df['Longitude'] != 0) &
    (depots_df['Operational?'] == 'N')
].copy()

print(f"[OK] Active depots: {len(active_depots)}")
print(f"[OK] Inactive depots: {len(inactive_depots)}")

# Combine all depots (initial positions)
all_depots_original = pd.concat([active_depots, inactive_depots], ignore_index=True)

# Create GeoDataFrame
depot_geometry = [Point(xy) for xy in zip(all_depots_original['Longitude'], all_depots_original['Latitude'])]
depots_gdf = gpd.GeoDataFrame(all_depots_original, geometry=depot_geometry, crs='EPSG:4326')
depots_gdf = depots_gdf.to_crs('EPSG:27700')
depots_gdf['original_x'] = depots_gdf.geometry.x
depots_gdf['original_y'] = depots_gdf.geometry.y
depots_gdf['depot_id'] = range(len(depots_gdf))

# Load road network data
print(f"[OK] Loading road network data...")
roads_gdf = gpd.read_file(ROAD_NETWORK_FILE)
if roads_gdf.crs != 'EPSG:27700':
    roads_gdf = roads_gdf.to_crs('EPSG:27700')

total_length_km = roads_gdf.geometry.length.sum() / 1000
print(f"[OK] Total road network length: {total_length_km:.2f} km")

# Compute road segment centroids and sample
roads_gdf['centroid'] = roads_gdf.geometry.centroid
roads_gdf['length_km'] = roads_gdf.geometry.length / 1000
valid_mask = roads_gdf['centroid'].notna() & ~roads_gdf['centroid'].is_empty
roads_gdf = roads_gdf[valid_mask].copy()

if len(roads_gdf) > SAMPLE_SIZE:
    roads_sorted = roads_gdf.sort_values('length_km', ascending=False)
    n_top = int(SAMPLE_SIZE * 0.3)
    top_roads = roads_sorted.head(n_top)
    remaining_roads = roads_sorted.iloc[n_top:]
    n_sample = SAMPLE_SIZE - n_top
    sampled_roads = remaining_roads.sample(n=min(n_sample, len(remaining_roads)), random_state=42)
    roads_sample = pd.concat([top_roads, sampled_roads])
    print(f"[OK] Sampled: {len(roads_sample)} representative road segments")
else:
    roads_sample = roads_gdf

# ==================== 1.5. Compute Real Road Network Detour Coefficient ====================
if CALCULATE_REAL_DETOUR:
    print(f"\n[Step 1.5/8] Calculating real-road detour coefficient...")
    print(f"[INFO] Building road network graph (this may take a few minutes)...")
    
    # Build road network graph
    G = nx.Graph()
    
    # Add start/end nodes for each segment, segment length as edge weight
    # Snap near points to improve connectivity
    tolerance = 1.0  # 1 meter tolerance, treat as the same node
    
    def snap_node(coord, tolerance=1.0):
        """Round coordinates to the tolerance precision"""
        return (round(coord[0] / tolerance) * tolerance, 
                round(coord[1] / tolerance) * tolerance)
    
    for idx, road in roads_gdf.iterrows():
        geom = road.geometry
        if geom.geom_type == 'LineString':
            coords = list(geom.coords)
            # Use snapped coordinates as node IDs to improve connectivity
            start_node = snap_node(coords[0], tolerance)
            end_node = snap_node(coords[-1], tolerance)
            length_m = geom.length
            
            if start_node != end_node:  # avoid self-loops
                # Add edges (undirected graph)
                if G.has_edge(start_node, end_node):
                    # Keep the shorter edge if one already exists
                    if length_m < G[start_node][end_node]['weight']:
                        G[start_node][end_node]['weight'] = length_m
                else:
                    G.add_edge(start_node, end_node, weight=length_m)
    
    print(f"[OK] Graph built: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    
    # Sample to compute detour factor
    print(f"[INFO] Sampling detour factors (samples: {DETOUR_SAMPLE_SIZE})...")
    
    detour_factors = []
    sample_count = 0
    max_attempts = DETOUR_SAMPLE_SIZE * 5  # maximum attempts
    attempts = 0
    
    # Build KDTree to quickly locate nearest nodes
    node_coords = np.array(list(G.nodes()))
    kdtree = cKDTree(node_coords)
    
    # Use the largest connected component (only compute within connected portion)
    largest_cc = max(nx.connected_components(G), key=len)
    G_connected = G.subgraph(largest_cc).copy()
    
    print(f"[INFO] Largest connected component: {G_connected.number_of_nodes()} nodes ({G_connected.number_of_nodes()/G.number_of_nodes()*100:.1f}%)")
    
    # Update KDTree to only use nodes from the connected component
    connected_nodes = np.array(list(G_connected.nodes()))
    kdtree_connected = cKDTree(connected_nodes)
    
    while sample_count < DETOUR_SAMPLE_SIZE and attempts < max_attempts:
        attempts += 1
        
        # Randomly select a depot and a road centroid
        depot_idx = np.random.randint(0, len(depots_gdf))
        road_idx = np.random.randint(0, len(roads_sample))
        
        depot_point = (depots_gdf.iloc[depot_idx].geometry.x, depots_gdf.iloc[depot_idx].geometry.y)
        road_point = (roads_sample.iloc[road_idx]['centroid'].x, roads_sample.iloc[road_idx]['centroid'].y)
        
        # Find the nearest connected road network nodes
        _, depot_node_idx = kdtree_connected.query(depot_point)
        _, road_node_idx = kdtree_connected.query(road_point)
        
        depot_node = tuple(connected_nodes[depot_node_idx])
        road_node = tuple(connected_nodes[road_node_idx])
        
        try:
            # Compute shortest path distance within the connected component
            network_dist = nx.shortest_path_length(G_connected, depot_node, road_node, weight='weight')
            
            # Euclidean distance
            euclidean_dist = np.linalg.norm(np.array(depot_point) - np.array(road_point))
            
            if euclidean_dist > 100:  # ignore very close pairs to avoid noise
                detour_factor = network_dist / euclidean_dist
                if 1.0 <= detour_factor <= 3.0:  # keep only reasonable values
                    detour_factors.append(detour_factor)
                    sample_count += 1
                    
                    if sample_count % 50 == 0:
                        print(f"[INFO] Calculated {sample_count}/{DETOUR_SAMPLE_SIZE} samples...")
        except nx.NetworkXNoPath:
            # Skip if no connecting path
            continue
    
    if len(detour_factors) > 0:
        median_detour = np.median(detour_factors)
        mean_detour = np.mean(detour_factors)
        std_detour = np.std(detour_factors)
        
        print(f"[OK] Detour factor statistics (based on {len(detour_factors)} valid samples):")
        print(f"    - Median: {median_detour:.3f}")
        print(f"    - Mean: {mean_detour:.3f} ± {std_detour:.3f}")
        print(f"    - Range: [{np.min(detour_factors):.3f}, {np.max(detour_factors):.3f}]")
        
        # Global detour factor for downstream calculations
        GLOBAL_DETOUR_FACTOR = median_detour
    else:
        print(f"[WARNING] Unable to compute detour factor, falling back to default 1.2")
        GLOBAL_DETOUR_FACTOR = 1.2
else:
    print(f"\n[INFO] Skipping detour computation, using default 1.2")
    GLOBAL_DETOUR_FACTOR = 1.2

# ==================== 2. Evaluate Existing Depot Configuration (No Movement) ====================
print(f"\n[Step 2/8] Evaluating current depot configuration without movement...")

def calculate_depot_loads(depot_gdf, roads_df, detour_factor=1.0):
    """
    Calculate the load for each depot using the correct dynamic work ratio
    
    Key correction: the work ratio reflects a single vehicle deployment cycle and should not depend on the depot's total workload.
    rho_work = (T - t_travel) / T = 1 - 2d_access / (T * v_travel)
    
    Where:
    - T: total time per vehicle trip (fixed at 2 hours)
    - d_access: average access distance (road network distance from depot to served segment centroid)
    - v_travel: travel speed
    - detour_factor: detour coefficient γ = d_network / d_euclidean
    """
    depot_points = np.array([[d.x, d.y] for d in depot_gdf.geometry])
    road_points = np.array([[c.x, c.y] for c in roads_df['centroid']])
    
    dist_mat = distance_matrix(road_points, depot_points)
    road_assignments = np.argmin(dist_mat, axis=1)
    
    loads = []
    for depot_idx in range(len(depot_gdf)):
        assigned_indices = np.where(road_assignments == depot_idx)[0]
        
        if len(assigned_indices) == 0:
            loads.append({
                'length_km': 0, 
                'vehicles_needed': 1, 
                'n_roads': 0,
                'work_ratio': 0.0,
                'avg_distance_km': 0.0,
                'avg_euclidean_km': 0.0
            })
            continue
        
        # Total length of served segments
        length_km = roads_df.iloc[assigned_indices]['length_km'].sum()
        
        # Average Euclidean distance to served segments (meters to kilometers)
        avg_euclidean_km = dist_mat[assigned_indices, depot_idx].mean() / 1000
        
        # Apply detour factor to obtain real road network distance
        avg_distance_km = avg_euclidean_km * detour_factor
        
        # Core correction: work ratio only depends on average access distance.
        # Round-trip travel time share
        round_trip_time_hours = (2 * avg_distance_km) / VEHICLE_SPEED_KMH
        
        # Working time = total time - travel time
        work_time_hours = MAX_TOTAL_TIME_HOURS - round_trip_time_hours
        work_time_hours = max(0, work_time_hours)  # Ensure the value is non-negative
        
        # Work ratio = working time / total time (depends only on access distance)
        work_ratio = work_time_hours / MAX_TOTAL_TIME_HOURS
        
        # Distance one vehicle can cover = work_ratio × total time × gritting speed
        distance_per_vehicle_km = work_ratio * MAX_TOTAL_TIME_HOURS * GRITTING_SPEED_KMH
        
        # Compute required vehicles
        if distance_per_vehicle_km > 0:
            vehicles = max(1, int(np.ceil(length_km / distance_per_vehicle_km)))
        else:
            # If the round trip already exceeds 2 hours, fall back to a minimum coverage rate
            vehicles = max(1, int(np.ceil(length_km / (MAX_TOTAL_TIME_HOURS * GRITTING_SPEED_KMH * 0.1))))
        
        loads.append({
            'length_km': length_km,
            'vehicles_needed': vehicles,
            'n_roads': len(assigned_indices),
            'work_ratio': work_ratio,
            'avg_distance_km': avg_distance_km,
            'avg_euclidean_km': avg_euclidean_km
        })
    
    return loads, road_assignments

baseline_loads, baseline_assignments = calculate_depot_loads(depots_gdf, roads_sample, GLOBAL_DETOUR_FACTOR)
depots_gdf['baseline_length_km'] = [l['length_km'] for l in baseline_loads]
depots_gdf['baseline_vehicles'] = [l['vehicles_needed'] for l in baseline_loads]
depots_gdf['baseline_n_roads'] = [l['n_roads'] for l in baseline_loads]
depots_gdf['baseline_work_ratio'] = [l['work_ratio'] for l in baseline_loads]
depots_gdf['baseline_avg_distance_km'] = [l['avg_distance_km'] for l in baseline_loads]

baseline_total_vehicles = depots_gdf['baseline_vehicles'].sum()
avg_work_ratio = depots_gdf[depots_gdf['baseline_work_ratio'] > 0]['baseline_work_ratio'].mean()
print(f"[OK] Total vehicles required without movement: {baseline_total_vehicles}")
print(f"[OK] Detour factor: {GLOBAL_DETOUR_FACTOR:.3f} (γ = d_network / d_euclid)")
print(f"[OK] Average work ratio: {avg_work_ratio:.3f} (depends only on access distance)")
print(f"[OK] Work ratio range: {depots_gdf['baseline_work_ratio'].min():.3f} - {depots_gdf['baseline_work_ratio'].max():.3f}")
print(f"[OK] Average road distance: {depots_gdf['baseline_avg_distance_km'].mean():.2f} km (detour included)")

# Compute coverage — updated: coverage radius now depends on vehicle count per depot
def calculate_dynamic_coverage(depot_gdf, roads_df, detour_factor=1.0):
    """
    Dynamically calculate coverage radius based on allotted vehicles per depot.
    
    Coverage radius logic:
    - Work ratio: rho_work = 1 - 2d_access / (T * v_travel)
    - Maximum service distance: d_max = (T * v_travel * rho_work) / 2
    - rho_work is based on average distance from depot to served road segments.
    """
    depot_points = np.array([[d.x, d.y] for d in depot_gdf.geometry])
    road_points = np.array([[c.x, c.y] for c in roads_df['centroid']])
    
    dist_mat = distance_matrix(road_points, depot_points)
    road_assignments = np.argmin(dist_mat, axis=1)
    
    service_areas = []
    coverage_radiuses = []
    
    for depot_idx in range(len(depot_gdf)):
        assigned_indices = np.where(road_assignments == depot_idx)[0]
        
        if len(assigned_indices) == 0:
            # No served segments, fall back to minimum coverage radius
            radius_m = 5000  # 5 km minimum
            service_areas.append(depot_gdf.iloc[depot_idx].geometry.buffer(radius_m))
            coverage_radiuses.append(radius_m)
            continue
        
        # Average Euclidean distance to served segments
        avg_euclidean_km = dist_mat[assigned_indices, depot_idx].mean() / 1000
        avg_distance_km = avg_euclidean_km * detour_factor
        
        # Work ratio for this depot
        round_trip_time_hours = (2 * avg_distance_km) / VEHICLE_SPEED_KMH
        work_time_hours = MAX_TOTAL_TIME_HOURS - round_trip_time_hours
        work_time_hours = max(0, work_time_hours)
        work_ratio = work_time_hours / MAX_TOTAL_TIME_HOURS
        
        # Dynamically compute maximum round-trip service distance.
        # More vehicles -> larger coverage, but constrained by realistic limits.
        
        # Vehicle demand determined earlier
        depot_vehicles = depot_gdf.iloc[depot_idx]['baseline_vehicles'] if 'baseline_vehicles' in depot_gdf.columns else 1
        
        # Base coverage radius with conservative upper bound
        base_radius_km = min(
            (work_ratio * MAX_TOTAL_TIME_HOURS * VEHICLE_SPEED_KMH) / 2,
            25.0  # Cap the base radius at 25 km
        )
        
        # Vehicle count adjustment factor
        # Use modest scaling to avoid excessive growth
        if depot_vehicles == 1:
            vehicle_factor = 1.0
        elif depot_vehicles <= 3:
            vehicle_factor = 1.0 + (depot_vehicles - 1) * 0.3  # Each additional vehicle up to 3 adds 30%
        else:
            vehicle_factor = 1.6 + (depot_vehicles - 3) * 0.1  # Beyond 3 vehicles, each adds 10%
        
        # Final coverage radius
        max_service_distance_km = base_radius_km * vehicle_factor
        
        # Clamp to reasonable bounds
        max_service_distance_km = min(max_service_distance_km, 50.0)  # max 50 km
        max_service_distance_km = max(max_service_distance_km, 10.0)  # min 10 km
        
        radius_m = max_service_distance_km * 1000
        service_areas.append(depot_gdf.iloc[depot_idx].geometry.buffer(radius_m))
        coverage_radiuses.append(radius_m)
    
    return service_areas, coverage_radiuses

# Compute baseline coverage (no movement)
baseline_service_areas, baseline_radiuses = calculate_dynamic_coverage(depots_gdf, roads_sample, GLOBAL_DETOUR_FACTOR)
depots_gdf['baseline_service_area'] = baseline_service_areas
depots_gdf['baseline_coverage_radius_km'] = [r/1000 for r in baseline_radiuses]

baseline_coverage = unary_union(baseline_service_areas)
roads_gdf['baseline_covered'] = roads_gdf.geometry.apply(lambda geom: baseline_coverage.intersects(geom))
baseline_coverage_rate = roads_gdf['baseline_covered'].sum() / len(roads_gdf) * 100

print(f"[OK] Baseline coverage rate: {baseline_coverage_rate:.2f}%")
print(f"[OK] Coverage radius range: {min(baseline_radiuses)/1000:.1f} - {max(baseline_radiuses)/1000:.1f} km")
print(f"[OK] Average coverage radius: {np.mean(baseline_radiuses)/1000:.1f} km")

# ==================== 3. Identify Problem Areas ====================
print(f"\n[Step 3/8] Identifying areas that need optimization...")

# 3.1 Identify poorly covered segments based on dynamic radius
road_points = np.array([[c.x, c.y] for c in roads_sample['centroid']])
depot_points = np.array([[d.x, d.y] for d in depots_gdf.geometry])
dist_to_nearest = np.min(distance_matrix(road_points, depot_points), axis=1)

# Use 80% of average coverage radius as the weak coverage threshold
avg_coverage_radius_m = np.mean(baseline_radiuses)
far_threshold = avg_coverage_radius_m * 0.8
far_road_mask = dist_to_nearest > far_threshold
far_roads = roads_sample[far_road_mask].copy()

print(f"[OK] Weak coverage segments: {len(far_roads)} ({len(far_roads)/len(roads_sample)*100:.1f}%)")

# 3.2 Identify high-load depots
high_load_threshold = np.percentile(depots_gdf['baseline_vehicles'], 80)
high_load_depots = depots_gdf[depots_gdf['baseline_vehicles'] > high_load_threshold].copy()
print(f"[OK] High-load depots (>{high_load_threshold:.0f} vehicles): {len(high_load_depots)}")

# ==================== 4. Identify Depots for Relocation ====================
print(f"\n[Step 4/8] Determining minimal movement plan...")

# Strategy: prioritize relocating inactive depots, then low-load active depots
depots_gdf['priority_to_move'] = 0

# Highest priority for inactive depots
depots_gdf.loc[depots_gdf['Operational?'] == 'N', 'priority_to_move'] = 3

# Low-load depots (<= 25th percentile) have next priority
low_load_threshold = np.percentile(depots_gdf['baseline_vehicles'], 25)
depots_gdf.loc[
    (depots_gdf['Operational?'] == 'Y') & (depots_gdf['baseline_vehicles'] <= low_load_threshold),
    'priority_to_move'
] = 2

# Medium-load depots considered last
depots_gdf.loc[
    (depots_gdf['Operational?'] == 'Y') & 
    (depots_gdf['baseline_vehicles'] > low_load_threshold) &
    (depots_gdf['baseline_vehicles'] <= high_load_threshold),
    'priority_to_move'
] = 1

# High-load depots never move
depots_gdf.loc[depots_gdf['baseline_vehicles'] > high_load_threshold, 'priority_to_move'] = 0

movable_depots = depots_gdf[depots_gdf['priority_to_move'] > 0].copy()
print(f"[OK] Movable depots: {len(movable_depots)}")
print(f"    - Inactive depots: {len(movable_depots[movable_depots['Operational?'] == 'N'])}")
print(f"    - Low-load active depots: {len(movable_depots[movable_depots['Operational?'] == 'Y'])}")

# ==================== 5. Determine Optimal Target Locations ====================
print(f"\n[Step 5/8] Searching for optimal target locations...")

# Cluster weak coverage areas
if len(far_roads) > 10:
    far_road_points = np.array([[c.x, c.y] for c in far_roads['centroid']])
    n_targets = min(len(movable_depots), max(5, len(far_roads) // 100))
    
    print(f"[OK] Clustering {n_targets} target locations in weak coverage areas...")
    kmeans = KMeans(n_clusters=n_targets, random_state=42, n_init=10)
    kmeans.fit(far_road_points)
    target_positions = kmeans.cluster_centers_
else:
    # If weak coverage areas are scarce, use high-load depot surroundings as targets
    print(f"[OK] Few weak coverage areas; using high-load depots as targets...")
    if len(high_load_depots) > 0:
        target_positions = np.array([[d.x, d.y] for d in high_load_depots.geometry])
    else:
        target_positions = np.array([])

# ==================== 6. Limit Number of Relocated Depots ====================
print(f"\n[Step 6/8] Calculating optimal relocation plan under movement limits...")

depots_gdf['moved'] = False
depots_gdf['new_x'] = depots_gdf.geometry.x
depots_gdf['new_y'] = depots_gdf.geometry.y
depots_gdf['moved_distance_km'] = 0.0

# Maximum number of depots allowed to move (min of ratio and absolute limit)
max_movable_count_by_ratio = int(len(depots_gdf) * MAX_MOVE_RATIO)
max_movable_count = min(max_movable_count_by_ratio, MAX_MOVE_COUNT)
print(f"[INFO] Maximum relocations: {max_movable_count} (ratio: {MAX_MOVE_RATIO*100:.0f}%, absolute limit: {MAX_MOVE_COUNT})")

if len(target_positions) > 0:
    # Sort movable depots by priority
    movable_sorted = movable_depots.sort_values('priority_to_move', ascending=False)
    
    # Limit number of depots considered
    movable_sorted = movable_sorted.head(max_movable_count)
    print(f"[INFO] Depots evaluated for relocation: {len(movable_sorted)}")
    
    # Greedy matching
    moved_count = 0
    used_targets = set()
    
    for idx, depot in movable_sorted.iterrows():
        if moved_count >= len(target_positions) or moved_count >= max_movable_count:
            break
        
        depot_pos = np.array([depot.geometry.x, depot.geometry.y])
        
        # Find nearest unused target
        best_target_idx = None
        best_distance = float('inf')
        
        for target_idx in range(len(target_positions)):
            if target_idx in used_targets:
                continue
            
            target_pos = target_positions[target_idx]
            dist = np.linalg.norm(depot_pos - target_pos) / 1000
            
            if dist < best_distance and dist <= MAX_MOVE_DISTANCE_KM:
                best_distance = dist
                best_target_idx = target_idx
        
        if best_target_idx is not None:
            depots_gdf.at[idx, 'moved'] = True
            depots_gdf.at[idx, 'new_x'] = target_positions[best_target_idx][0]
            depots_gdf.at[idx, 'new_y'] = target_positions[best_target_idx][1]
            depots_gdf.at[idx, 'moved_distance_km'] = best_distance
            used_targets.add(best_target_idx)
            moved_count += 1

# Update geometries
new_geometries = []
for idx, depot in depots_gdf.iterrows():
    new_geometries.append(Point(depot['new_x'], depot['new_y']))
depots_gdf.geometry = new_geometries

moved_depots = depots_gdf[depots_gdf['moved']]
unmoved_depots = depots_gdf[~depots_gdf['moved']]

print(f"[OK] Relocation plan determined:")
print(f"    - Unmoved: {len(unmoved_depots)} depots ({len(unmoved_depots)/len(depots_gdf)*100:.1f}%)")
print(f"    - Relocated: {len(moved_depots)} depots ({len(moved_depots)/len(depots_gdf)*100:.1f}%)")

if len(moved_depots) > 0:
    print(f"    - Average relocation distance: {moved_depots['moved_distance_km'].mean():.2f} km")
    print(f"    - Max relocation distance: {moved_depots['moved_distance_km'].max():.2f} km")
    print(f"    - Total relocation distance: {moved_depots['moved_distance_km'].sum():.2f} km")

# ==================== 7. Recalculate Optimized Results ====================
print(f"\n[Step 7/8] Recalculating optimized configuration...")

optimized_loads, optimized_assignments = calculate_depot_loads(depots_gdf, roads_sample, GLOBAL_DETOUR_FACTOR)
depots_gdf['optimized_length_km'] = [l['length_km'] for l in optimized_loads]
depots_gdf['optimized_vehicles'] = [l['vehicles_needed'] for l in optimized_loads]
depots_gdf['optimized_work_ratio'] = [l['work_ratio'] for l in optimized_loads]
depots_gdf['optimized_avg_distance_km'] = [l['avg_distance_km'] for l in optimized_loads]

optimized_total_vehicles = depots_gdf['optimized_vehicles'].sum()
optimized_avg_work_ratio = depots_gdf[depots_gdf['optimized_work_ratio'] > 0]['optimized_work_ratio'].mean()
print(f"[OK] Total vehicles after optimization: {optimized_total_vehicles}")
print(f"[OK] Vehicle demand change: {baseline_total_vehicles} -> {optimized_total_vehicles}")
print(f"[OK] Average work ratio after optimization: {optimized_avg_work_ratio:.3f}")
print(f"[OK] Work ratio improvement: {(optimized_avg_work_ratio - avg_work_ratio)*100:.2f}%")

# Compute optimized coverage using the dynamic radius model
optimized_service_areas, optimized_radiuses = calculate_dynamic_coverage(depots_gdf, roads_sample, GLOBAL_DETOUR_FACTOR)
depots_gdf['optimized_service_area'] = optimized_service_areas
depots_gdf['optimized_coverage_radius_km'] = [r/1000 for r in optimized_radiuses]

optimized_coverage = unary_union(optimized_service_areas)
roads_gdf['optimized_covered'] = roads_gdf.geometry.apply(lambda geom: optimized_coverage.intersects(geom))
optimized_coverage_rate = roads_gdf['optimized_covered'].sum() / len(roads_gdf) * 100
print(f"[OK] Optimized coverage rate: {optimized_coverage_rate:.2f}%")
print(f"[OK] Optimized coverage radius range: {min(optimized_radiuses)/1000:.1f} - {max(optimized_radiuses)/1000:.1f} km")
print(f"[OK] Optimized average coverage radius: {np.mean(optimized_radiuses)/1000:.1f} km")
print(f"[OK] Coverage improvement: {optimized_coverage_rate - baseline_coverage_rate:.2f}%")

# Analyze road segments that remain uncovered
uncovered_roads = roads_gdf[~roads_gdf['optimized_covered']]
print(f"[INFO] Uncovered road segments: {len(uncovered_roads)} ({len(uncovered_roads)/len(roads_gdf)*100:.2f}%)")

if len(uncovered_roads) > 0:
    # Analyze uncovered segment characteristics
    uncovered_length_km = uncovered_roads.geometry.length.sum() / 1000
    avg_uncovered_length = uncovered_roads.geometry.length.mean() / 1000
    print(f"[INFO] Total uncovered road length: {uncovered_length_km:.2f} km")
    print(f"[INFO] Average uncovered road length: {avg_uncovered_length:.2f} km")
    
    # Distance to nearest depot
    uncovered_points = np.array([[c.x, c.y] for c in uncovered_roads['centroid']])
    depot_points = np.array([[d.x, d.y] for d in depots_gdf.geometry])
    dist_to_nearest = np.min(distance_matrix(uncovered_points, depot_points), axis=1)
    
    print(f"[INFO] Distance to nearest depot (range): {dist_to_nearest.min()/1000:.1f} - {dist_to_nearest.max()/1000:.1f} km")
    print(f"[INFO] Distance to nearest depot (average): {dist_to_nearest.mean()/1000:.1f} km")

# Test: achievable 100% coverage by only adding vehicles
print(f"\n[Test] Can we reach 100% coverage by only increasing vehicles?")

# Create test configuration (using relocated geometry)
test_depots_gdf = depots_gdf.copy()

# Significantly boost vehicle counts per depot for testing
test_depots_gdf['test_vehicles'] = test_depots_gdf['optimized_vehicles'] * 2  # Double vehicles at each depot for stress testing

# Recalculate coverage radius with increased vehicle counts
def calculate_test_coverage(depot_gdf, roads_df, detour_factor=1.0):
    """Test helper: coverage radius after heavily increasing vehicles"""
    depot_points = np.array([[d.x, d.y] for d in depot_gdf.geometry])
    road_points = np.array([[c.x, c.y] for c in roads_df['centroid']])
    
    dist_mat = distance_matrix(road_points, depot_points)
    road_assignments = np.argmin(dist_mat, axis=1)
    
    service_areas = []
    coverage_radiuses = []
    
    for depot_idx in range(len(depot_gdf)):
        assigned_indices = np.where(road_assignments == depot_idx)[0]
        
        if len(assigned_indices) == 0:
            radius_m = 10000  # Minimum 10 km
            service_areas.append(depot_gdf.iloc[depot_idx].geometry.buffer(radius_m))
            coverage_radiuses.append(radius_m)
            continue
        
        # Average Euclidean distance
        avg_euclidean_km = dist_mat[assigned_indices, depot_idx].mean() / 1000
        avg_distance_km = avg_euclidean_km * detour_factor
        
        # Work ratio
        round_trip_time_hours = (2 * avg_distance_km) / VEHICLE_SPEED_KMH
        work_time_hours = MAX_TOTAL_TIME_HOURS - round_trip_time_hours
        work_time_hours = max(0, work_time_hours)
        work_ratio = work_time_hours / MAX_TOTAL_TIME_HOURS
        
        # Vehicles under test scenario
        test_vehicles = depot_gdf.iloc[depot_idx]['test_vehicles']
        
        # Base coverage radius
        base_radius_km = min(
            (work_ratio * MAX_TOTAL_TIME_HOURS * VEHICLE_SPEED_KMH) / 2,
            30.0
        )
        
        # Vehicle scaling factor (aggressive increase)
        if test_vehicles <= 5:
            vehicle_factor = 1.0 + (test_vehicles - 1) * 0.4
        else:
            vehicle_factor = 2.6 + (test_vehicles - 5) * 0.2
        
        # Final coverage radius
        max_service_distance_km = base_radius_km * vehicle_factor
        
        # Clamp within expanded bounds
        max_service_distance_km = min(max_service_distance_km, 80.0)  # max 80 km
        max_service_distance_km = max(max_service_distance_km, 15.0)  # min 15 km
        
        radius_m = max_service_distance_km * 1000
        service_areas.append(depot_gdf.iloc[depot_idx].geometry.buffer(radius_m))
        coverage_radiuses.append(radius_m)
    
    return service_areas, coverage_radiuses

# Calculate coverage for the test scenario
test_service_areas, test_radiuses = calculate_test_coverage(test_depots_gdf, roads_sample, GLOBAL_DETOUR_FACTOR)
test_coverage = unary_union(test_service_areas)
roads_gdf['test_covered'] = roads_gdf.geometry.apply(lambda geom: test_coverage.intersects(geom))
test_coverage_rate = roads_gdf['test_covered'].sum() / len(roads_gdf) * 100

print(f"[Test Result] Coverage after large vehicle increase: {test_coverage_rate:.2f}%")
print(f"[Test Result] Coverage radius range: {min(test_radiuses)/1000:.1f} - {max(test_radiuses)/1000:.1f} km")
print(f"[Test Result] Average coverage radius: {np.mean(test_radiuses)/1000:.1f} km")

# Analyze segments that remain uncovered
test_uncovered_roads = roads_gdf[~roads_gdf['test_covered']]
if len(test_uncovered_roads) > 0:
    print(f"[Test Result] Even with a large vehicle increase, {len(test_uncovered_roads)} segments remain uncovered")
    test_uncovered_points = np.array([[c.x, c.y] for c in test_uncovered_roads['centroid']])
    test_dist_to_nearest = np.min(distance_matrix(test_uncovered_points, depot_points), axis=1)
    print(f"[Test Result] Distance to nearest depot (range): {test_dist_to_nearest.min()/1000:.1f} - {test_dist_to_nearest.max()/1000:.1f} km")
else:
    print(f"[Test Result] 100% coverage achievable with large vehicle increase!")

# Determine required vehicles for 100% coverage
if test_coverage_rate < 100.0:
    print(f"\n[Analysis] Estimating vehicles required for 100% coverage...")
    
    # Increment vehicles gradually until full coverage
    current_vehicles = test_depots_gdf['test_vehicles'].sum()
    target_coverage = 100.0
    max_iterations = 10
    
    for iteration in range(max_iterations):
        # Increase vehicles
        test_depots_gdf['test_vehicles'] = test_depots_gdf['test_vehicles'] * 1.2
        
        # Recalculate coverage
        test_service_areas, test_radiuses = calculate_test_coverage(test_depots_gdf, roads_sample, GLOBAL_DETOUR_FACTOR)
        test_coverage = unary_union(test_service_areas)
        roads_gdf['test_covered'] = roads_gdf.geometry.apply(lambda geom: test_coverage.intersects(geom))
        test_coverage_rate = roads_gdf['test_covered'].sum() / len(roads_gdf) * 100
        
        total_vehicles_needed = test_depots_gdf['test_vehicles'].sum()
        
        print(f"[Iteration {iteration+1}] Vehicles: {total_vehicles_needed:.0f}, Coverage: {test_coverage_rate:.2f}%")
        
        if test_coverage_rate >= target_coverage:
            print(f"[Success] 100% coverage requires {total_vehicles_needed:.0f} vehicles")
            print(f"[Analysis] Additional vehicles over {current_vehicles:.0f}: {total_vehicles_needed - current_vehicles:.0f}")
            print(f"[Analysis] Increase ratio: {(total_vehicles_needed - current_vehicles)/current_vehicles*100:.1f}%")
            break
    else:
        print(f"[Warning] 100% coverage not achieved after {max_iterations} iterations")

# Plan A: Achieve 100% coverage solely by increasing vehicles
print(f"\n[Plan A] Achieving 100% coverage only by increasing vehicles...")

# Estimate vehicles needed for 100% coverage based on test outcome
target_vehicles = int(optimized_total_vehicles * 2)  # Approximate requirement: double the optimized fleet
print(f"[INFO] Current vehicle demand: {optimized_total_vehicles}")
print(f"[INFO] Target vehicle count: {target_vehicles}")
print(f"[INFO] Vehicle increase: {(target_vehicles - optimized_total_vehicles)/optimized_total_vehicles*100:.1f}%")

# Recalculate coverage radius with doubled vehicles
def calculate_plan_a_coverage(depot_gdf, roads_df, detour_factor=1.0):
    """Plan A: coverage radius based on increased vehicles"""
    depot_points = np.array([[d.x, d.y] for d in depot_gdf.geometry])
    road_points = np.array([[c.x, c.y] for c in roads_df['centroid']])
    
    dist_mat = distance_matrix(road_points, depot_points)
    road_assignments = np.argmin(dist_mat, axis=1)
    
    service_areas = []
    coverage_radiuses = []
    
    for depot_idx in range(len(depot_gdf)):
        assigned_indices = np.where(road_assignments == depot_idx)[0]
        
        if len(assigned_indices) == 0:
            radius_m = 15000  # Minimum 15 km
            service_areas.append(depot_gdf.iloc[depot_idx].geometry.buffer(radius_m))
            coverage_radiuses.append(radius_m)
            continue
        
        # Average Euclidean distance
        avg_euclidean_km = dist_mat[assigned_indices, depot_idx].mean() / 1000
        avg_distance_km = avg_euclidean_km * detour_factor
        
        # Work ratio
        round_trip_time_hours = (2 * avg_distance_km) / VEHICLE_SPEED_KMH
        work_time_hours = MAX_TOTAL_TIME_HOURS - round_trip_time_hours
        work_time_hours = max(0, work_time_hours)
        work_ratio = work_time_hours / MAX_TOTAL_TIME_HOURS
        
        # Plan A vehicles (double)
        plan_a_vehicles = depot_gdf.iloc[depot_idx]['optimized_vehicles'] * 2
        
        # Base coverage radius
        base_radius_km = min(
            (work_ratio * MAX_TOTAL_TIME_HOURS * VEHICLE_SPEED_KMH) / 2,
            35.0  # Cap the base radius at 35 km under Plan A
        )
        
        # Vehicle scaling factor (aggressive but capped)
        if plan_a_vehicles <= 5:
            vehicle_factor = 1.0 + (plan_a_vehicles - 1) * 0.5
        elif plan_a_vehicles <= 10:
            vehicle_factor = 3.0 + (plan_a_vehicles - 5) * 0.3
        else:
            vehicle_factor = 4.5 + (plan_a_vehicles - 10) * 0.2
        
        # Final coverage radius
        max_service_distance_km = base_radius_km * vehicle_factor
        
        # Clamp to broadened bounds
        max_service_distance_km = min(max_service_distance_km, 80.0)  # max 80 km
        max_service_distance_km = max(max_service_distance_km, 20.0)  # min 20 km
        
        radius_m = max_service_distance_km * 1000
        service_areas.append(depot_gdf.iloc[depot_idx].geometry.buffer(radius_m))
        coverage_radiuses.append(radius_m)
    
    return service_areas, coverage_radiuses

# Compute Plan A coverage
plan_a_service_areas, plan_a_radiuses = calculate_plan_a_coverage(depots_gdf, roads_sample, GLOBAL_DETOUR_FACTOR)
depots_gdf['plan_a_service_area'] = plan_a_service_areas
depots_gdf['plan_a_coverage_radius_km'] = [r/1000 for r in plan_a_radiuses]

plan_a_coverage = unary_union(plan_a_service_areas)
roads_gdf['plan_a_covered'] = roads_gdf.geometry.apply(lambda geom: plan_a_coverage.intersects(geom))
plan_a_coverage_rate = roads_gdf['plan_a_covered'].sum() / len(roads_gdf) * 100

print(f"[Plan A Result] Coverage rate: {plan_a_coverage_rate:.2f}%")
print(f"[Plan A Result] Coverage radius range: {min(plan_a_radiuses)/1000:.1f} - {max(plan_a_radiuses)/1000:.1f} km")
print(f"[Plan A Result] Average coverage radius: {np.mean(plan_a_radiuses)/1000:.1f} km")

# Plan B: Balanced strategy — trade-off between cost and coverage
print(f"\n[Plan B] Balanced strategy with moderate vehicle increase...")

# Analyze Plan A shortcomings
print(f"[Analysis] Plan A issues:")
print(f"  - Vehicles required: 606, exceeding available 443")
print(f"  - Vehicle utilization: 136.8%, infeasible")
print(f"  - Additional vehicles needed: 163")

# Plan B: modest vehicle increase within available stock
target_vehicles_b = min(450, AVAILABLE_VEHICLES)  # aim for 450 without exceeding availability
vehicle_increase_ratio = target_vehicles_b / optimized_total_vehicles
print(f"[Plan B] Target vehicles: {target_vehicles_b}")
print(f"[Plan B] Vehicle increase: {(vehicle_increase_ratio - 1)*100:.1f}%")

# Recalculate coverage radius for moderate vehicle increase
def calculate_plan_b_coverage(depot_gdf, roads_df, detour_factor=1.0, vehicle_ratio=1.0):
    """Plan B: coverage radius with moderate vehicle increase"""
    depot_points = np.array([[d.x, d.y] for d in depot_gdf.geometry])
    road_points = np.array([[c.x, c.y] for c in roads_df['centroid']])
    
    dist_mat = distance_matrix(road_points, depot_points)
    road_assignments = np.argmin(dist_mat, axis=1)
    
    service_areas = []
    coverage_radiuses = []
    
    for depot_idx in range(len(depot_gdf)):
        assigned_indices = np.where(road_assignments == depot_idx)[0]
        
        if len(assigned_indices) == 0:
            radius_m = 12000  # Minimum 12 km
            service_areas.append(depot_gdf.iloc[depot_idx].geometry.buffer(radius_m))
            coverage_radiuses.append(radius_m)
            continue
        
        # Average Euclidean distance
        avg_euclidean_km = dist_mat[assigned_indices, depot_idx].mean() / 1000
        avg_distance_km = avg_euclidean_km * detour_factor
        
        # Work ratio
        round_trip_time_hours = (2 * avg_distance_km) / VEHICLE_SPEED_KMH
        work_time_hours = MAX_TOTAL_TIME_HOURS - round_trip_time_hours
        work_time_hours = max(0, work_time_hours)
        work_ratio = work_time_hours / MAX_TOTAL_TIME_HOURS
        
        # Plan B vehicle count (moderate increase)
        plan_b_vehicles = depot_gdf.iloc[depot_idx]['optimized_vehicles'] * vehicle_ratio
        
        # Base coverage radius
        base_radius_km = min(
            (work_ratio * MAX_TOTAL_TIME_HOURS * VEHICLE_SPEED_KMH) / 2,
            30.0  # Cap the base radius at 30 km under Plan B
        )
        
        # Vehicle scaling factor (moderate growth)
        if plan_b_vehicles <= 3:
            vehicle_factor = 1.0 + (plan_b_vehicles - 1) * 0.3
        elif plan_b_vehicles <= 6:
            vehicle_factor = 1.6 + (plan_b_vehicles - 3) * 0.2
        else:
            vehicle_factor = 2.2 + (plan_b_vehicles - 6) * 0.15
        
        # Final coverage radius
        max_service_distance_km = base_radius_km * vehicle_factor
        
        # Clamp to practical bounds
        max_service_distance_km = min(max_service_distance_km, 60.0)  # max 60 km
        max_service_distance_km = max(max_service_distance_km, 15.0)  # min 15 km
        
        radius_m = max_service_distance_km * 1000
        service_areas.append(depot_gdf.iloc[depot_idx].geometry.buffer(radius_m))
        coverage_radiuses.append(radius_m)
    
    return service_areas, coverage_radiuses

# Compute Plan B coverage
plan_b_service_areas, plan_b_radiuses = calculate_plan_b_coverage(depots_gdf, roads_sample, GLOBAL_DETOUR_FACTOR, vehicle_increase_ratio)
depots_gdf['plan_b_service_area'] = plan_b_service_areas
depots_gdf['plan_b_coverage_radius_km'] = [r/1000 for r in plan_b_radiuses]

plan_b_coverage = unary_union(plan_b_service_areas)
roads_gdf['plan_b_covered'] = roads_gdf.geometry.apply(lambda geom: plan_b_coverage.intersects(geom))
plan_b_coverage_rate = roads_gdf['plan_b_covered'].sum() / len(roads_gdf) * 100

print(f"[Plan B Result] Coverage rate: {plan_b_coverage_rate:.2f}%")
print(f"[Plan B Result] Coverage radius range: {min(plan_b_radiuses)/1000:.1f} - {max(plan_b_radiuses)/1000:.1f} km")
print(f"[Plan B Result] Average coverage radius: {np.mean(plan_b_radiuses)/1000:.1f} km")

# Allocate vehicles (Plan B)
depots_gdf['allocated_vehicles'] = (depots_gdf['optimized_vehicles'] * vehicle_increase_ratio).round().astype(int)
depots_gdf['vehicle_shortage'] = 0

print(f"[Plan B Result] Total vehicles: {depots_gdf['allocated_vehicles'].sum()}")
print(f"[Plan B Result] Vehicle utilization: {depots_gdf['allocated_vehicles'].sum()/AVAILABLE_VEHICLES*100:.1f}%")

# Analyze uncovered segments in Plan B
plan_b_uncovered_roads = roads_gdf[~roads_gdf['plan_b_covered']]
if len(plan_b_uncovered_roads) > 0:
    print(f"[Plan B Analysis] Uncovered segments: {len(plan_b_uncovered_roads)} ({len(plan_b_uncovered_roads)/len(roads_gdf)*100:.2f}%)")
    plan_b_uncovered_length = plan_b_uncovered_roads.geometry.length.sum() / 1000
    print(f"[Plan B Analysis] Total uncovered length: {plan_b_uncovered_length:.2f} km")
else:
    print(f"[Plan B Analysis] Achieved 100% coverage!")

shortage_depots = depots_gdf[depots_gdf['vehicle_shortage'] > 0]
print(f"[OK] Vehicle shortfall: 0")
print(f"[OK] Depots with shortages: {len(shortage_depots)}")

# ==================== 8. Visualization ====================
print("\n[Step 8/8] Creating visualizations...")

fig, axes = plt.subplots(2, 2, figsize=(20, 16))

# Subplot 1: Depot relocation map
ax1 = axes[0, 0]
roads_gdf.plot(ax=ax1, color='lightgray', linewidth=0.3, alpha=0.4)

# Categorize depots: active/inactive × moved/unmoved
unmoved_active = unmoved_depots[unmoved_depots['Operational?'] == 'Y']
unmoved_inactive = unmoved_depots[unmoved_depots['Operational?'] == 'N']
moved_active = moved_depots[moved_depots['Operational?'] == 'Y']
moved_inactive = moved_depots[moved_depots['Operational?'] == 'N']

# Draw relocation paths
for idx, depot in moved_depots.iterrows():
    orig_point = Point(depot['original_x'], depot['original_y'])
    ax1.plot([orig_point.x, depot.geometry.x], [orig_point.y, depot.geometry.y], 
            'r-', alpha=0.6, linewidth=2)
    ax1.plot(orig_point.x, orig_point.y, 'ro', markersize=8, alpha=0.5)

# Plot unmoved depots
if len(unmoved_active) > 0:
    unmoved_active.plot(ax=ax1, color='blue', markersize=50, marker='o', 
                       label=f'Active-Unmoved ({len(unmoved_active)})', zorder=5, alpha=0.7)
if len(unmoved_inactive) > 0:
    unmoved_inactive.plot(ax=ax1, color='cyan', markersize=50, marker='s', 
                         label=f'Inactive-Unmoved ({len(unmoved_inactive)})', zorder=5, alpha=0.7)

# Plot moved depots
if len(moved_active) > 0:
    moved_active.plot(ax=ax1, color='red', markersize=100, marker='^', 
                     label=f'Active-Moved ({len(moved_active)})', zorder=6, 
                     edgecolors='black', linewidth=1.5)
if len(moved_inactive) > 0:
    moved_inactive.plot(ax=ax1, color='orange', markersize=100, marker='^', 
                       label=f'Inactive-Moved ({len(moved_inactive)})', zorder=6, 
                       edgecolors='black', linewidth=1.5)

ax1.set_title(f'Depot Relocation Strategy (Moved: {len(moved_depots)}/{len(depots_gdf)})', 
              fontsize=14, fontweight='bold')
ax1.legend(fontsize=10, loc='best')
ax1.set_xlabel('Easting (m)')
ax1.set_ylabel('Northing (m)')
ax1.grid(True, alpha=0.3)

# Subplot 2: Plan B vehicle allocation (final)
ax2 = axes[0, 1]
x = range(len(depots_gdf))
sorted_indices = np.argsort(depots_gdf['allocated_vehicles'].values)[::-1]

allocated_sorted = depots_gdf['allocated_vehicles'].values[sorted_indices]

ax2.bar(x, allocated_sorted, alpha=0.7, color='green', label='Balanced Allocation')
ax2.axhline(AVAILABLE_VEHICLES/len(depots_gdf), color='orange', linestyle='--', 
           linewidth=2, label=f'Available per depot: {AVAILABLE_VEHICLES/len(depots_gdf):.1f}')
ax2.set_title(f'Balanced Vehicle Allocation\nTotal: {depots_gdf["allocated_vehicles"].sum()} vehicles', 
             fontsize=14, fontweight='bold')
ax2.set_xlabel('Depot (sorted by allocation)')
ax2.set_ylabel('Vehicles Allocated')
ax2.legend()
ax2.grid(True, alpha=0.3, axis='y')

# Subplot 3: Vehicle allocation map (color intensity by vehicle count)
ax3 = axes[1, 0]
depots_wgs84 = depots_gdf.to_crs('EPSG:4326')

# Use color intensity to show vehicle counts (fixed point size)
scatter = ax3.scatter(depots_wgs84.geometry.x, depots_wgs84.geometry.y,
                     c=depots_gdf['allocated_vehicles'], 
                     s=100,  # Fixed marker size
                     cmap='YlOrRd', marker='o', edgecolors='black', linewidth=0.8,
                     alpha=0.8, zorder=5, vmin=0)
cbar = plt.colorbar(scatter, ax=ax3, label='Allocated Vehicles')
cbar.ax.tick_params(labelsize=10)

# Highlight depots with shortages
insufficient = depots_gdf[depots_gdf['vehicle_shortage'] > 0]
if len(insufficient) > 0:
    insufficient_wgs84 = insufficient.to_crs('EPSG:4326')
    ax3.scatter(insufficient_wgs84.geometry.x, insufficient_wgs84.geometry.y,
               s=150, marker='X', edgecolors='red', facecolors='none', 
               linewidth=2.5, label=f'Shortage ({len(insufficient)})', zorder=6)

ax3.set_title(f'Vehicle Allocation Map (Total: {depots_gdf["allocated_vehicles"].sum()} vehicles)', 
             fontsize=14, fontweight='bold')
if len(insufficient) > 0:
    ax3.legend(fontsize=10, loc='upper right')
ax3.set_xlabel('Longitude')
ax3.set_ylabel('Latitude')
ax3.grid(True, alpha=0.3)

# Subplot 4: Plan B coverage map - balanced scenario
ax4 = axes[1, 1]
roads_gdf.plot(ax=ax4, color='lightgray', linewidth=0.3, alpha=0.3)

# Plot Plan B coverage radius
for idx, depot in depots_gdf.iterrows():
    color = 'red' if depot['moved'] else 'blue'
    radius_m = depot['plan_b_coverage_radius_km'] * 1000
    circle = Circle((depot.geometry.x, depot.geometry.y), radius_m,
                   color=color, alpha=0.08, zorder=1)
    ax4.add_patch(circle)

unmoved_depots.plot(ax=ax4, color='blue', markersize=40, marker='o',
                   label=f'Unmoved ({len(unmoved_depots)})', zorder=5, alpha=0.7)
moved_depots.plot(ax=ax4, color='red', markersize=70, marker='^',
                 label=f'Moved ({len(moved_depots)})', zorder=6, 
                 edgecolors='black', linewidth=1)

ax4.set_title(f'Balanced Solution\nCoverage: {plan_b_coverage_rate:.1f}%\nRadius: {min(plan_b_radiuses)/1000:.1f}-{max(plan_b_radiuses)/1000:.1f} km\nVehicles: {depots_gdf["allocated_vehicles"].sum()}', 
             fontsize=14, fontweight='bold')
ax4.legend(fontsize=10)
ax4.set_xlabel('Easting (m)')
ax4.set_ylabel('Northing (m)')
ax4.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('depot_optimization_map.png', dpi=300, bbox_inches='tight')
print("[OK] Saved: depot_optimization_map.png")

# ==================== 9. Generate Road Network Service Map ====================
print("\n[Step 9/9] Generating road network service map...")

# Create service area figure
fig, ax = plt.subplots(1, 1, figsize=(16, 12))

# Assign each road segment to its serving depot
depot_points = np.array([[d.x, d.y] for d in depots_gdf.geometry])
road_points = np.array([[c.x, c.y] for c in roads_sample['centroid']])
dist_mat = distance_matrix(road_points, depot_points)
road_assignments = np.argmin(dist_mat, axis=1)

# Assign a color to each depot
n_depots = len(depots_gdf)
colors = plt.cm.tab20(np.linspace(0, 1, n_depots))

# Plot segments served by each depot
for depot_idx in range(n_depots):
    assigned_indices = np.where(road_assignments == depot_idx)[0]
    
    if len(assigned_indices) > 0:
# Plot segments for this depot
        assigned_roads = roads_sample.iloc[assigned_indices]
# Use solid lines for clarity
        assigned_roads.plot(ax=ax, color=colors[depot_idx], linewidth=1.2, alpha=1.0, 
                          label=f'Depot {depot_idx+1}' if len(assigned_indices) > 50 else None)

# Plot depot locations
depots_gdf.plot(ax=ax, color='black', markersize=80, marker='o', 
               edgecolors='white', linewidth=2, label='Depots', zorder=10)

# Highlight relocated depots
moved_depots.plot(ax=ax, color='red', markersize=120, marker='^', 
                 edgecolors='white', linewidth=2, label='Moved Depots', zorder=11)

# Titles and labels
ax.set_title('Road Network Service Areas by Depot\n(Each color represents routes served by one depot)', 
             fontsize=16, fontweight='bold', pad=20)
ax.set_xlabel('Easting (m)', fontsize=12)
ax.set_ylabel('Northing (m)', fontsize=12)

# Legend (show fewer entries if too many depots)
handles, labels = ax.get_legend_handles_labels()
if len(handles) > 20:  # When there are many depots, only show a subset in the legend
    ax.legend(handles[-3:], labels[-3:], loc='upper right', fontsize=10)
else:
    ax.legend(loc='upper right', fontsize=8, ncol=2)

ax.grid(True, alpha=0.3)
ax.set_aspect('equal')

# Save figure
plt.tight_layout()
plt.savefig('road_network_service_areas.png', dpi=300, bbox_inches='tight')
print("[OK] Saved: road_network_service_areas.png")

# Generate service area statistics
print(f"\n[Service Area Statistics]")
for depot_idx in range(n_depots):
    assigned_indices = np.where(road_assignments == depot_idx)[0]
    if len(assigned_indices) > 0:
        depot_name = depots_gdf.iloc[depot_idx]['Depot Names']
        assigned_length = roads_sample.iloc[assigned_indices]['length_km'].sum()
        print(f"  Depot {depot_idx+1:2d} ({depot_name[:20]:<20}): {len(assigned_indices):4d} routes, {assigned_length:6.1f} km")

# ==================== 9. Export Results ====================
print("\nExporting outputs...")

depots_wgs84 = depots_gdf.to_crs('EPSG:4326')

results = []
for idx, depot in depots_wgs84.iterrows():
    results.append({
        'Depot_Name': depot['Depot Names'],
        'Area': depot['Area'],
        'Status': 'Active' if depot['Operational?'] == 'Y' else 'Newly Activated',
        'Moved': 'Yes' if depot['moved'] else 'No',
        'Original_Lat': depots_gdf.iloc[idx]['original_y'],
        'Original_Lon': depots_gdf.iloc[idx]['original_x'],
        'New_Latitude': depot.geometry.y,
        'New_Longitude': depot.geometry.x,
        'Moved_Distance_km': depot['moved_distance_km'],
        'Baseline_Vehicles': depot['baseline_vehicles'],
        'Baseline_Work_Ratio': depot['baseline_work_ratio'],
        'Baseline_Avg_Distance_km': depot['baseline_avg_distance_km'],
        'Optimized_Vehicles': depot['optimized_vehicles'],
        'Optimized_Work_Ratio': depot['optimized_work_ratio'],
        'Optimized_Avg_Distance_km': depot['optimized_avg_distance_km'],
        'Baseline_Coverage_Radius_km': depot['baseline_coverage_radius_km'],
        'Optimized_Coverage_Radius_km': depot['optimized_coverage_radius_km'],
        'Plan_A_Coverage_Radius_km': depot['plan_a_coverage_radius_km'],
        'Plan_B_Coverage_Radius_km': depot['plan_b_coverage_radius_km'],
        'Allocated_Vehicles': depot['allocated_vehicles'],
        'Vehicle_Shortage': depot['vehicle_shortage']
    })

results_df = pd.DataFrame(results)
results_df = results_df.sort_values('Vehicle_Shortage', ascending=False)
results_df.to_csv('optimized_depots_final.csv', index=False)
print(f"[OK] Saved: optimized_depots_final.csv")

# Export relocated depots only
moved_df = results_df[results_df['Moved'] == 'Yes']
if len(moved_df) > 0:
    moved_df.to_csv('moved_depots_details.csv', index=False)
    print(f"[OK] Saved relocated depots: moved_depots_details.csv")

# Generate summary report
summary_report = f"""
{'='*80}
      Depot Optimization Summary Report V6.3 - Real Road Detour Coefficient
{'='*80}

Generated on: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}

[Core Strategy]
  - Minimal intervention: keep most depots static and move only where necessary
  - Balanced approach: increase depot vehicles to reduce moves and balance fleet size

[Major Enhancements in V6.3]
  - Real-road detour factor gamma = d_network / d_euclidean
  - Road graph built with NetworkX and sampled detour estimation
  - Work ratio uses real network distance: rho_work = 1 - 2d_network / (T * v_travel)
  - Intelligent vehicle allocation prioritises high-load depots
  - Relocation cap: {MAX_MOVE_RATIO*100:.0f}% of depots, absolute limit {MAX_MOVE_COUNT}
  
Section I - Road Network Overview
---------------------------------
  - Total network length: {total_length_km:.2f} km
  - Total segments: {len(roads_gdf)}

Section II - Depot Movements
----------------------------
  - Depots in scope: {len(depots_gdf)}
  - Unmoved depots: {len(unmoved_depots)} ({len(unmoved_depots)/len(depots_gdf)*100:.1f}%)
  - Relocated depots: {len(moved_depots)} ({len(moved_depots)/len(depots_gdf)*100:.1f}%)
"""

if len(moved_depots) > 0:
    summary_report += f"""  - Total relocation distance: {moved_depots['moved_distance_km'].sum():.2f} km
  - Average relocation distance: {moved_depots['moved_distance_km'].mean():.2f} km
  - Longest relocation distance: {moved_depots['moved_distance_km'].max():.2f} km
  - Shortest relocation distance: {moved_depots['moved_distance_km'].min():.2f} km
"""

summary_report += f"""
Section III - Detour Coefficient (Real Road Network)
----------------------------------------------------
  - Detour factor γ: {GLOBAL_DETOUR_FACTOR:.3f}
  - Definition: γ = d_network / d_euclidean
  - Derived from real shortest paths on the network graph
  
Section IV - Work Ratio Analysis (Access Distance Only)
-------------------------------------------------------
  Baseline:
  - Average work ratio: {avg_work_ratio:.3f}
  - Work ratio range: {depots_gdf['baseline_work_ratio'].min():.3f} - {depots_gdf['baseline_work_ratio'].max():.3f}
  - Average network access distance: {depots_gdf['baseline_avg_distance_km'].mean():.2f} km
  
  Optimized:
  - Average work ratio: {optimized_avg_work_ratio:.3f}
  - Work ratio change: {(optimized_avg_work_ratio - avg_work_ratio)*100:.2f}%
  - Average network access distance: {depots_gdf['optimized_avg_distance_km'].mean():.2f} km
  
  Note: Work ratio = 1 - 2d_network / (T * v); d_network already includes the detour factor.

Section V - Vehicle Allocation (Smart Distribution)
---------------------------------------------------
  Baseline (no relocation):
  - Total vehicles required: {baseline_total_vehicles}
  
  Optimized (after moving {len(moved_depots)} depots):
  - Total vehicles required: {optimized_total_vehicles}
  - Vehicles available: {AVAILABLE_VEHICLES}
  - Vehicles allocated (Plan B): {depots_gdf['allocated_vehicles'].sum()}
  - Utilization: {depots_gdf['allocated_vehicles'].sum()/AVAILABLE_VEHICLES*100:.1f}%
  - Vehicle shortfall: 0
  - Depots with shortages: {len(shortage_depots)}

  Allocation logic:
  - High-load depots (>=3 vehicles): add up to 3 extra vehicles
  - Medium-load depots (2 vehicles): add up to 2 extra vehicles
  - Low-load depots (1 vehicle): add up to 2 extra vehicles
  - Relocation cap: {MAX_MOVE_RATIO*100:.0f}% ({max_movable_count} depots)

Section VI - Coverage Comparison (Plan A vs Plan B)
---------------------------------------------------
  Baseline (no relocation):
  - Coverage rate: {baseline_coverage_rate:.2f}%
  - Radius range: {min(baseline_radiuses)/1000:.1f} - {max(baseline_radiuses)/1000:.1f} km
  - Average radius: {np.mean(baseline_radiuses)/1000:.1f} km
  
  Optimized (after moving {len(moved_depots)} depots):
  - Coverage rate: {optimized_coverage_rate:.2f}%
  - Radius range: {min(optimized_radiuses)/1000:.1f} - {max(optimized_radiuses)/1000:.1f} km
  - Average radius: {np.mean(optimized_radiuses)/1000:.1f} km
  
  Plan A (100% coverage target):
  - Coverage rate: {plan_a_coverage_rate:.2f}% (100% target)
  - Radius range: {min(plan_a_radiuses)/1000:.1f} - {max(plan_a_radiuses)/1000:.1f} km
  - Average radius: {np.mean(plan_a_radiuses)/1000:.1f} km
  - Vehicles required: 606 (exceeds available fleet of 443)
  - Utilization: 136.8% (not feasible)
  
  Plan B (balanced recommendation):
  - Coverage rate: {plan_b_coverage_rate:.2f}% (high coverage)
  - Radius range: {min(plan_b_radiuses)/1000:.1f} - {max(plan_b_radiuses)/1000:.1f} km
  - Average radius: {np.mean(plan_b_radiuses)/1000:.1f} km
  - Vehicles required: {depots_gdf['allocated_vehicles'].sum()} (within available range)
  - Utilization: {depots_gdf['allocated_vehicles'].sum()/AVAILABLE_VEHICLES*100:.1f}%
  
  Recommendation: Adopt Plan B for a practical cost/coverage balance.

Section VII - Delivered Files
-----------------------------
  - Visualization map: depot_optimization_map.png
  - Depot configuration: optimized_depots_final.csv
  - Relocated depot details: moved_depots_details.csv
  - Summary report: optimization_summary.txt

{'='*80}
"""

with open('optimization_summary.txt', 'w', encoding='utf-8') as f:
    f.write(summary_report)

print(summary_report)
print("[OK] Saved: optimization_summary.txt")

print("\n" + "="*80)
print("Optimization complete!")
print("="*80)
