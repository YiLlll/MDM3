import geopandas as gpd
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import networkx as nx
from shapely.geometry import Point, LineString
from pathlib import Path
import random
import warnings
from collections import defaultdict
import pickle
from datetime import datetime

warnings.filterwarnings("ignore", category=UserWarning, module="matplotlib")

# =====================
# CONFIGURATION (edit these)
# =====================
SCRIPT_DIR = Path(__file__).parent
GEOJSON_PATH = SCRIPT_DIR / "england_complete_road_network.geojson"
DEPOT_CSV_PATH = SCRIPT_DIR / "depot_details.csv"

# performance & behaviour
AVERAGE_SPEED_KMH = 80
BUFFER_METERS = 1000

# ACO parameters
ACO_ENABLED = True
N_ANTS = 12            # ants per vehicle
N_ITERATIONS = 100      # iterations per vehicle's pheromone update
ALPHA = 0.3
BETA = 6.0
RHO = 0.1
STEPS = 5000

# multi-vehicle
VEHICLES_PER_DEPOT = 6   # number of vehicles starting from each depot

# simplification tolerance (meters in projected CRS)
SIMPLIFY_TOLERANCE = 200

# max iterations for connecting subnetworks (safety)
MAX_CONNECT_ITER = 100000

# Cache files (so you don't reconnect every run)
GRAPH_CACHE_PATH = SCRIPT_DIR / "connected_network_full.pkl"     # saves (G, bridge_gdf)
SAVE_CONNECTED_GEOJSON = SCRIPT_DIR / "connected_bridges_full.geojson"  # optional bridge geojson

# =====================
# SUBNETWORK REBUILD CONTROL
# =====================
FORCE_RECONNECT_SUBNETWORKS = False   # True = always rebuild subnetworks

# =====================
# HELPERS
# =====================
def travel_time_hours(distance_m):
    return distance_m / 1000.0 / AVERAGE_SPEED_KMH

def euclid(a, b):
    return np.hypot(a[0] - b[0], a[1] - b[1])

# =====================
# LOAD DATA
# =====================
def load_routes_and_depots(geojson_path, depot_csv):
    print(f"Loading routes from: {geojson_path.name}")
    routes_gdf = gpd.read_file(geojson_path).to_crs(epsg=3857)

    # simplify if large dataset (improves connectivity and performance)
    if len(routes_gdf) > 1000:
        print(f"Simplifying geometries (tolerance={SIMPLIFY_TOLERANCE}m) to improve connectivity...")
        routes_gdf["geometry"] = routes_gdf["geometry"].simplify(SIMPLIFY_TOLERANCE, preserve_topology=True)
        routes_gdf = routes_gdf[~routes_gdf["geometry"].is_empty].copy()
        print(f"{len(routes_gdf)} features remain after simplification")

        # preview simplified geometry
        fig, ax = plt.subplots(figsize=(10, 8))
        routes_gdf.plot(ax=ax, color="steelblue", linewidth=0.8, alpha=0.8)
        ax.set_title(f"Preview: Simplified Route Geometry ({SIMPLIFY_TOLERANCE} m tolerance)")
        ax.set_aspect("equal")
        plt.tight_layout()
        plt.show()

    depot_df = pd.read_csv(depot_csv)
    depot_df = depot_df[(depot_df["Latitude"] != 0) & (depot_df["Longitude"] != 0)]
    depot_gdf = gpd.GeoDataFrame(
        depot_df, geometry=gpd.points_from_xy(depot_df["Longitude"], depot_df["Latitude"]), crs="EPSG:4326"
    ).to_crs(epsg=3857)

    # use a small buffer so depots near the routes are included
    route_union = routes_gdf.buffer(BUFFER_METERS).union_all()
    depot_gdf = depot_gdf[depot_gdf.intersects(route_union)]

    print(f"Loaded {len(routes_gdf)} route features")
    print(f"Loaded {len(depot_gdf)} nearby depots")
    return routes_gdf, depot_gdf

# =====================
# BUILD GRAPH
# =====================
def build_graph_from_routes(gdf):
    print("Building network graph from routes...")
    G = nx.Graph()
    for _, row in gdf.iterrows():
        geom = row.geometry
        if geom is None:
            continue

        geom_type = getattr(geom, "geom_type", None)
        if geom_type == "LineString":
            lines = [geom]
        elif geom_type == "MultiLineString":
            lines = list(geom.geoms)
        else:
            continue

        for line in lines:
            coords = list(line.coords)
            for i in range(len(coords) - 1):
                u, v = coords[i], coords[i + 1]
                d = euclid(u, v)
                if G.has_edge(u, v):
                    if d < G[u][v].get("weight", np.inf):
                        G[u][v]["weight"] = d
                else:
                    G.add_edge(u, v, weight=d)

    # store node positions as attribute for easy access
    pos_map = {n: n for n in G.nodes()}
    nx.set_node_attributes(G, pos_map, "pos")

    print(f"Graph built with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges")
    return G

# =====================
# CONNECT SUBNETWORKS
# =====================
def connect_subnetworks(G, max_k=10):
    """
    Connects all subnetworks by iteratively linking the closest pair of subnetworks
    until only one remains. New connecting edges are added to the graph.
    """
    from scipy.spatial import cKDTree

    print("\nConnecting subnetworks...")

    comps = list(nx.connected_components(G))
    print(f"Initial subnetworks: {len(comps)}")

    if len(comps) <= 1:
        return G, gpd.GeoDataFrame(columns=["geometry"], crs="EPSG:3857")

    added_edges = []
    iteration = 0

    while len(comps) > 1 and iteration < MAX_CONNECT_ITER:
        centroids = [np.mean(np.array(list(c)), axis=0) for c in comps]
        tree = cKDTree(centroids)

        dists, idxs = tree.query(centroids, k=min(max_k, len(centroids)))

        min_dist = float("inf")
        best_pair = None

        for i, comp_a in enumerate(comps):
            for j in np.atleast_1d(idxs[i]):
                if i == j or j >= len(comps):
                    continue
                comp_b = comps[j]
                a_nodes = np.array(list(comp_a))
                b_nodes = np.array(list(comp_b))

                tree_b = cKDTree(b_nodes)
                dist, idx_b = tree_b.query(a_nodes)
                min_idx = np.argmin(dist)
                d = dist[min_idx]
                if d < min_dist:
                    min_dist = d
                    best_pair = (tuple(a_nodes[min_idx]), tuple(b_nodes[idx_b[min_idx]]))

        if best_pair:
            u, v = best_pair
            G.add_edge(u, v, weight=min_dist, artificial=True)
            added_edges.append(LineString([u, v]))
            comps = list(nx.connected_components(G))
            iteration += 1
            if iteration % 50 == 0 or iteration <= 5:
                print(f"  Connected 2 subnetworks -> now {len(comps)} remain (added edge {min_dist:.1f} m)")
        else:
            print("No suitable pair found — breaking.")
            break

    bridge_gdf = gpd.GeoDataFrame(geometry=added_edges, crs="EPSG:3857")
    comps_final = list(nx.connected_components(G))
    print(f"✅ All subnetworks connected (or stopped). Final total: {len(comps_final)}")
    print(f"Added {len(added_edges)} bridge edges. Iterations: {iteration}")

    # Plot combined network
    fig, ax = plt.subplots(figsize=(10, 10))
    for u, v in G.edges():
        line = LineString([u, v])
        ax.plot(*line.xy, color="lightgray", linewidth=0.8, alpha=0.6)
    if len(added_edges) > 0:
        bridge_gdf.plot(ax=ax, color="red", linewidth=2, label="Added Connections")
    plt.title("Connected Road Network (Bridged Subnetworks)")
    plt.legend()
    plt.axis("equal")
    plt.show()

    return G, bridge_gdf

# =====================
# PLOT CONNECTED NETWORK (original routes + added connectors)
# =====================
def plot_connected_network(routes_gdf, bridge_gdf):
    fig, ax = plt.subplots(figsize=(12, 10))
    routes_gdf.plot(ax=ax, color="lightgray", linewidth=1.0, alpha=0.8, label="Original Routes")
    if not bridge_gdf.empty:
        bridge_gdf.plot(ax=ax, color="red", linewidth=2.0, label="Added Connections", zorder=5)
    ax.set_title("Final Integrated Network (Original + Added Connections)")
    ax.legend()
    ax.set_aspect("equal")
    plt.tight_layout()
    plt.show()

# =====================
# PLOT SUBNETWORKS (multi-colour)
# =====================
def plot_graph_subnetworks(G, depot_nodes=None, title="Road Network Subnetworks"):
    comps = list(nx.connected_components(G.to_undirected()))
    print(f"\nDetected {len(comps)} subnetworks")

    num = len(comps)
    cmap = plt.cm.get_cmap("tab20", num if num <= 20 else num)
    plt.figure(figsize=(10, 10))

    for idx, comp in enumerate(comps):
        color = cmap(idx % cmap.N)
        sub = G.subgraph(comp)
        for u, v in sub.edges():
            plt.plot([u[0], v[0]], [u[1], v[1]], color=color, linewidth=0.8, alpha=0.9)
        xs, ys = zip(*sub.nodes())
        plt.scatter(xs, ys, s=6, color=color, alpha=0.9)

    if depot_nodes:
        dx, dy = zip(*depot_nodes)
        plt.scatter(dx, dy, color="blue", s=60, zorder=6, label="Depots")

    plt.title(title)
    plt.axis("equal")
    plt.tight_layout()
    plt.legend()
    plt.show()

# =====================
# CONNECT DEPOTS -> GRAPH
# =====================
def connect_depots_to_graph(G, depot_gdf):
    depot_nodes = []
    for _, depot in depot_gdf.iterrows():
        depot_point = (depot.geometry.x, depot.geometry.y)
        nearest_node = min(G.nodes, key=lambda n: euclid((n[0], n[1]), depot_point))
        dist = euclid(nearest_node, depot_point)
        if depot_point not in G:
            G.add_node(depot_point)
            nx.set_node_attributes(G, {depot_point: depot_point}, "pos")
        G.add_edge(depot_point, nearest_node, weight=dist)
        depot_nodes.append(depot_point)
    print(f"Connected {len(depot_nodes)} depots to network.")
    return depot_nodes

# =====================
# PREPARE ORIENTED EDGES & ADJACENCY
# =====================
def make_oriented_edges(G):
    oriented_edges = []
    edge_weight = {}
    for (u, v, data) in G.edges(data=True):
        w = data.get("weight", euclid(u, v))
        oriented_edges.append((u, v))
        edge_weight[(u, v)] = w
        oriented_edges.append((v, u))
        edge_weight[(v, u)] = w
    start_map = defaultdict(list)
    for idx, (a, b) in enumerate(oriented_edges):
        start_map[a].append(idx)
    return oriented_edges, start_map, edge_weight

# =====================
# MULTI-VEHICLE, MULTI-DEPOT ACO (MEMORY-EFFICIENT)
# =====================
def multi_depot_multi_vehicle_aco(
    G, depot_nodes, oriented_edges, start_map, edge_weight,
    vehicles_per_depot=2, n_ants=12, n_iter=20,
    alpha=0.5, beta=5.0, rho=0.1,
    start_radius=8, teleport_max_dist_m=50000
):
    """
    Multi-depot multi-vehicle ACO with progressive CSV saving.
    Each depot's results are appended to a shared output file as the algorithm runs.
    """
    epsilon = 1e-12
    oriented_index_map = {edge: i for i, edge in enumerate(oriented_edges)}
    total_edges = len(oriented_edges)

    global AVERAGE_SPEED_KMH

    # File setup for progressive saving
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    csv_path = Path(__file__).parent / f"ACO_summary_progress_{timestamp}.csv"

    # Undirected edge mapping (unique coverage)
    undirected_weight = {}
    oriented_to_undirected = {}
    for i, (u, v) in enumerate(oriented_edges):
        key = tuple(sorted((u, v)))
        oriented_to_undirected[i] = key
        if key not in undirected_weight:
            undirected_weight[key] = edge_weight.get((u, v), euclid(u, v))

    def nearby_edges(G, depot_node, radius):
        visited_nodes = {depot_node}
        queue = [(depot_node, 0)]
        nearby = []
        while queue:
            node, dist = queue.pop(0)
            if dist < radius:
                for nb in G.neighbors(node):
                    if nb not in visited_nodes:
                        visited_nodes.add(nb)
                        queue.append((nb, dist + 1))
                    for e in [(node, nb), (nb, node)]:
                        if e in oriented_index_map:
                            nearby.append(oriented_index_map[e])
        return list(dict.fromkeys(nearby))

    global_tau = {i: 1.0 for i in range(total_edges)}
    global_covered_oriented = set()
    global_covered_undirected = set()
    results = []

    for depot_idx, depot_node in enumerate(depot_nodes):
        print(f"\n--- Depot {depot_idx + 1}/{len(depot_nodes)} ---")

        # determine start edges (oriented indices)
        if start_map and depot_node in start_map and start_map[depot_node]:
            start_edges = list(start_map[depot_node])
        else:
            start_edges = nearby_edges(G, depot_node, start_radius)
        if not start_edges:
            nearest_edge = min(oriented_edges, key=lambda e: euclid(e[0], depot_node))
            start_edges = [oriented_index_map[nearest_edge]]

        depot_covered_oriented = set()
        depot_covered_undirected = set()
        vehicle_routes = []

        for vehicle_num in range(vehicles_per_depot):
            print(f"\n🚚 Vehicle {vehicle_num+1} starting from Depot {depot_idx+1}")
            vehicle_oriented_set = set()
            vehicle_undirected_set = set()

            for iteration in range(1, n_iter + 1):
                ant_routes = []
                ant_scores = []

                for ant in range(n_ants):
                    current_idx = random.choice(start_edges)
                    route = [current_idx]
                    visited = set(route)
                    steps = 0

                    while True:
                        steps += 1
                        if steps > STEPS:
                            break

                        current_end = oriented_edges[current_idx][1]
                        if current_end not in G:
                            break

                        neighbors = list(G.neighbors(current_end))
                        candidates = []
                        for nb in neighbors:
                            for e in [(current_end, nb), (nb, current_end)]:
                                if e in oriented_index_map:
                                    e_idx = oriented_index_map[e]
                                    if e_idx not in visited or e_idx not in global_covered_oriented:
                                        candidates.append(e_idx)

                        if not candidates:
                            uncovered_edges = [e for e in range(total_edges) if e not in global_covered_oriented]
                            if not uncovered_edges:
                                break
                            min_dist = float("inf")
                            chosen_uncovered = None
                            for e in uncovered_edges:
                                start_node = oriented_edges[e][0]
                                d = euclid(current_end, start_node)
                                if d < min_dist:
                                    min_dist = d
                                    chosen_uncovered = e
                            if chosen_uncovered is not None and min_dist <= teleport_max_dist_m:
                                current_idx = chosen_uncovered
                                route.append(current_idx)
                                visited.add(current_idx)
                                continue
                            else:
                                break

                        phs = np.array([global_tau[c] for c in candidates], dtype=float)
                        lengths = np.array([edge_weight.get(oriented_edges[c], euclid(*oriented_edges[c])) for c in candidates], dtype=float)
                        heur = 1.0 / np.maximum(lengths, epsilon)
                        unvisited_boost = np.array([50.0 if c not in global_covered_oriented else 0.2 for c in candidates])
                        probs_raw = (phs ** alpha) * (heur ** beta) * unvisited_boost
                        total_prob = probs_raw.sum()

                        if total_prob <= 0 or np.isnan(total_prob):
                            chosen = random.choice(candidates)
                        else:
                            probs = probs_raw / total_prob
                            probs = np.maximum(probs, 0)
                            s = probs.sum()
                            if s <= 0 or np.isnan(s):
                                chosen = random.choice(candidates)
                            else:
                                probs = probs / s
                                chosen = np.random.choice(candidates, p=probs)

                        route.append(chosen)
                        visited.add(chosen)
                        current_idx = chosen
                        if len(route) > 5000:
                            break

                    total_distance_m = sum(edge_weight.get(oriented_edges[e], euclid(*oriented_edges[e])) for e in route)
                    unique_undirected_keys = set(oriented_to_undirected[e] for e in route)
                    unique_distance_m = sum(undirected_weight[k] for k in unique_undirected_keys)
                    ant_routes.append({
                        "route_oriented": route,
                        "total_distance_m": total_distance_m,
                        "unique_undirected_keys": unique_undirected_keys,
                        "unique_distance_m": unique_distance_m
                    })
                    ant_scores.append(total_distance_m)

                if ant_scores:
                    best_idx = int(np.argmax(ant_scores))
                    ant_best = ant_routes[best_idx]
                    ant_best_route_oriented = ant_best["route_oriented"]

                    for e_idx in ant_best_route_oriented:
                        global_tau[e_idx] = (1 - rho) * global_tau.get(e_idx, 1.0) + rho * ant_scores[best_idx]

                    vehicle_oriented_set.update(ant_best_route_oriented)
                    vehicle_undirected_set.update(oriented_to_undirected[e] for e in ant_best_route_oriented)
                    depot_covered_oriented.update(ant_best_route_oriented)
                    depot_covered_undirected.update(oriented_to_undirected[e] for e in ant_best_route_oriented)
                    global_covered_oriented.update(ant_best_route_oriented)
                    global_covered_undirected.update(oriented_to_undirected[e] for e in ant_best_route_oriented)

                if iteration % max(1, n_iter // 10) == 0:
                    print(f"  Iter {iteration:03d}: global (oriented) coverage {len(global_covered_oriented)}/{total_edges} "
                          f" | global (unique undirected) {len(global_covered_undirected)}")

                if len(global_covered_oriented) >= total_edges:
                    break

            vehicle_unique_distance_m = sum(undirected_weight[k] for k in vehicle_undirected_set)
            vehicle_routes.append((vehicle_num, list(vehicle_undirected_set), vehicle_unique_distance_m, len(vehicle_undirected_set)))

        total_unique_distance_depot_m = sum(undirected_weight[k] for k in depot_covered_undirected)
        total_unique_distance_km = total_unique_distance_depot_m / 1000.0
        avg_unique_distance_per_vehicle_m = total_unique_distance_depot_m / vehicles_per_depot
        avg_time_per_vehicle_h = (total_unique_distance_km / vehicles_per_depot) / AVERAGE_SPEED_KMH

        results.append((depot_idx, vehicle_routes, len(global_covered_oriented),
                        total_edges, total_unique_distance_depot_m, avg_time_per_vehicle_h,
                        avg_unique_distance_per_vehicle_m))

        print(f"🏁 Depot {depot_idx+1} done: Unique coverage {total_unique_distance_km:.2f} km total, "
              f"~{avg_unique_distance_per_vehicle_m/1000:.2f} km/vehicle")

        # --- Progressive CSV writing ---
        depot_records = []
        for vnum, vehicle_edges, dist_m, n_edges in vehicle_routes:
            depot_records.append({
                "Level": "Vehicle",
                "Depot": depot_idx + 1,
                "Vehicle": vnum + 1,
                "Unique_Edges": n_edges,
                "Unique_Distance_km": dist_m / 1000.0,
                "Total_Unique_Distance_km": "",
                "Avg_Time_per_Vehicle_h": "",
                "Avg_Distance_per_Vehicle_km": ""
            })
        depot_records.append({
            "Level": "Depot",
            "Depot": depot_idx + 1,
            "Vehicle": "",
            "Unique_Edges": len(depot_covered_undirected),
            "Unique_Distance_km": "",
            "Total_Unique_Distance_km": total_unique_distance_depot_m / 1000.0,
            "Avg_Time_per_Vehicle_h": avg_time_per_vehicle_h,
            "Avg_Distance_per_Vehicle_km": avg_unique_distance_per_vehicle_m / 1000.0
        })

        df_new = pd.DataFrame(depot_records)
        if not csv_path.exists():
            df_new.to_csv(csv_path, index=False)
        else:
            df_new.to_csv(csv_path, index=False, mode="a", header=False)
        print(f"💾 Saved progress for Depot {depot_idx + 1} to {csv_path}")

        if len(global_covered_oriented) >= total_edges:
            break

    print(f"\n✅ FINAL GLOBAL COVERAGE: {len(global_covered_oriented)}/{total_edges}")
    print(f"✅ FINAL UNIQUE UNDIRECTED COVERAGE: {len(global_covered_undirected)}")
    print(f"📁 All progress saved to {csv_path}")
    return results

# =====================
# PLOTTING
# =====================
def plot_initial_map(routes_gdf, depot_gdf):
    fig, ax = plt.subplots(figsize=(12, 10))
    routes_gdf.plot(ax=ax, color="lightgray", linewidth=1.2, alpha=0.9, zorder=1)
    if not depot_gdf.empty:
        op = depot_gdf[depot_gdf.get("Operational?", pd.Series()) == "Y"]
        non_op = depot_gdf[depot_gdf.get("Operational?", pd.Series()) == "N"]
        if not op.empty:
            op.plot(ax=ax, color="green", markersize=50, edgecolor="black", label="Operational", zorder=3)
        if not non_op.empty:
            non_op.plot(ax=ax, color="red", markersize=50, edgecolor="black", label="Non-Operational", zorder=3)
    ax.set_aspect("equal")
    ax.set_title("Initial Road Network and Depots")
    plt.legend()
    plt.tight_layout()
    plt.show()

def plot_results_multi(routes_gdf, depot_gdf, results, depot_nodes):
    """
    Plot multi-depot ACO results.
    Draws vehicle routes and depot numbers.
    (CSV saving now handled progressively during ACO.)
    """
    fig, ax = plt.subplots(figsize=(12, 10))
    routes_gdf.plot(ax=ax, color="lightgray", linewidth=1.2, alpha=0.8, zorder=1)
    cmap = plt.colormaps.get_cmap("tab10")

    # Plot depots with numbering
    if not depot_gdf.empty:
        depot_gdf.plot(ax=ax, color="orange", markersize=70, marker="*", edgecolor="black", zorder=4)
        for depot_idx, depot_node in enumerate(depot_nodes):
            node_x, node_y = depot_node
            depot_gdf["dist_tmp_for_label"] = depot_gdf.geometry.distance(Point(node_x, node_y))
            nearest = depot_gdf.loc[depot_gdf["dist_tmp_for_label"].idxmin()]
            geom = nearest.geometry
            ax.text(geom.x + 1000, geom.y + 1000, str(depot_idx + 1),
                    fontsize=10, color="black", weight="bold", zorder=6)
        depot_gdf.drop(columns=["dist_tmp_for_label"], inplace=True, errors="ignore")

    # Draw vehicle routes
    for ridx, entry in enumerate(results):
        (depot_idx, vehicle_routes, _, _, _, _, _) = entry
        color = cmap(depot_idx % cmap.N)
        for vnum, vehicle_edges, _, _ in vehicle_routes:
            for u, v in vehicle_edges:
                ax.plot([u[0], v[0]], [u[1], v[1]], color=color, alpha=0.7, linewidth=2.2, zorder=3)

    ax.set_aspect("equal")
    ax.set_title("Multi-Depot Multi-Vehicle ACO Results (Unique Edge Distances)")
    plt.tight_layout()
    plt.show()

# =====================
# MAIN
# =====================
def main():
    routes_gdf, depot_gdf = load_routes_and_depots(GEOJSON_PATH, DEPOT_CSV_PATH)
    print("\nPlotting initial map (before ACO)...")
    plot_initial_map(routes_gdf, depot_gdf)

    G = build_graph_from_routes(routes_gdf)

    # -------------------------
    # Show subnetworks (each in different colour)
    # -------------------------
    plot_graph_subnetworks(
        G,
        depot_nodes=[(p.geometry.x, p.geometry.y) for _, p in depot_gdf.iterrows()],
        title="Network Subnetworks (before connection)"
    )

    # -------------------------
    # Connect all subnetworks (load cache or rebuild)
    # -------------------------
    bridge_gdf = gpd.GeoDataFrame(geometry=[], crs="EPSG:3857")

    rebuild = False
    if FORCE_RECONNECT_SUBNETWORKS:
        print("\nFORCE_RECONNECT_SUBNETWORKS=True → rebuilding subnetworks from scratch.")
        rebuild = True
    elif not GRAPH_CACHE_PATH.exists():
        print("\nNo existing cache found → building subnetworks.")
        rebuild = True
    else:
        print(f"\nLoading connected graph from cache: {GRAPH_CACHE_PATH}")
        try:
            with open(GRAPH_CACHE_PATH, "rb") as f:
                G, bridge_gdf = pickle.load(f)
            print("✅ Loaded cached connected graph successfully.")
        except Exception as e:
            print(f"⚠️ Cache load failed ({e}). Rebuilding subnetworks instead.")
            rebuild = True

    if rebuild:
        G, bridge_gdf = connect_subnetworks(G)
        try:
            with open(GRAPH_CACHE_PATH, "wb") as f:
                pickle.dump((G, bridge_gdf), f)
            print(f"✅ Saved connected graph to {GRAPH_CACHE_PATH}")
        except Exception as e:
            print(f"⚠️ Could not save connected graph cache: {e}")

    # Optionally save bridge GeoJSON
    try:
        if not bridge_gdf.empty:
            bridge_gdf.to_file(SAVE_CONNECTED_GEOJSON, driver="GeoJSON")
            print(f"✅ Saved added bridge edges to {SAVE_CONNECTED_GEOJSON}")
    except Exception as e:
        print(f"⚠️ Could not save bridge GeoJSON: {e}")

    # -------------------------
    # Plot unified connected network
    # -------------------------
    fig, ax = plt.subplots(figsize=(10, 8))
    routes_gdf.plot(ax=ax, color="lightgray", linewidth=0.8, alpha=0.6)
    if not bridge_gdf.empty:
        bridge_gdf.plot(ax=ax, color="red", linewidth=1.2, label="Added Bridge Edges", zorder=5)
    plt.title("Unified Network After Connecting Subnetworks")
    plt.axis("equal")
    plt.legend()
    plt.tight_layout()
    plt.show()

    # -------------------------
    # Connect depots & run ACO
    # -------------------------
    depot_nodes = connect_depots_to_graph(G, depot_gdf)
    oriented_edges, start_map, edge_weight = make_oriented_edges(G)

    if ACO_ENABLED and len(depot_nodes) > 0:
        results = multi_depot_multi_vehicle_aco(
            G, depot_nodes,
            oriented_edges, start_map, edge_weight,
            vehicles_per_depot=VEHICLES_PER_DEPOT,
            n_ants=N_ANTS, n_iter=N_ITERATIONS,

        )
        print("\nACO finished for all depots.")
        plot_results_multi(routes_gdf, depot_gdf, results, depot_nodes)
    else:
        print("ACO disabled or no depots found; only initial map shown.")


if __name__ == "__main__":
    main()