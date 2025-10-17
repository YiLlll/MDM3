import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import pandas as pd
import numpy as np
import networkx as nx
from shapely.geometry import Point, LineString
from pathlib import Path
import random
import warnings
from collections import defaultdict
from scipy.spatial import cKDTree
import os
import pickle

warnings.filterwarnings("ignore", category=UserWarning, module="matplotlib")

# =====================
# CONFIGURATION (edit these)
# =====================
SCRIPT_DIR = Path(__file__).parent
GEOJSON_PATH = SCRIPT_DIR / "area6_combined_routes.geojson"
DEPOT_CSV_PATH = SCRIPT_DIR / "depot_details.csv"

# performance & behavior
AVERAGE_SPEED_KMH = 80
TIME_LIMIT_HOURS = 2.0
BUFFER_METERS = 100

# ACO parameters
ACO_ENABLED = True
N_ANTS = 20            # ants per vehicle
N_ITERATIONS = 70      # iterations per vehicle's pheromone update
ALPHA = 1.5
BETA = 3.0
RHO = 0.5

# multi-vehicle
VEHICLES_PER_DEPOT = 4   # number of vehicles starting from each depot

# node sampling for ACO (limits work set size)
SAMPLED_NODE_COUNT = 2000  # number of sample nodes used for candidate moves (keeps memory low)

# simplification tolerance (meters in projected CRS)
SIMPLIFY_TOLERANCE = 1000

# max iterations for connecting subnetworks (safety)
MAX_CONNECT_ITER = 10000

# Cache files (so you don't reconnect every run)
GRAPH_CACHE_PATH = SCRIPT_DIR / "connected_network.pkl"     # saves (G, bridge_gdf)
SAVE_CONNECTED_GEOJSON = SCRIPT_DIR / "connected_bridges.geojson"  # optional bridge geojson

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
    print(f"\nDetected {len(comps)} subnetworks:")
    for i, comp in enumerate(comps, start=1):
        print(f"  Subnetwork {i}: {len(comp)} nodes")

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
def multi_depot_multi_vehicle_aco(G, depot_nodes, oriented_edges, start_map, edge_weight,
                                  vehicles_per_depot=2, n_ants=8, n_iter=15, time_limit_h=2.0):
    results = []
    epsilon = 1e-9
    oriented_index_map = {edge: i for i, edge in enumerate(oriented_edges)}

    for depot_idx, depot_node in enumerate(depot_nodes):
        print(f"\n--- Depot {depot_idx + 1}/{len(depot_nodes)} ---")
        start_edges = start_map.get(depot_node, [])
        if not start_edges:
            nearest_edge = min(oriented_edges, key=lambda e: euclid(e[0], depot_node))
            start_edges = [oriented_index_map[nearest_edge]]

        covered_edges = set()
        vehicle_routes = []
        base_tau = {i: 1.0 for i in range(len(oriented_edges))}

        for vehicle_num in range(vehicles_per_depot):
            best_vehicle_route = None
            best_vehicle_score = -1
            best_vehicle_time = None
            tau = base_tau.copy()

            for iteration in range(n_iter):
                ant_routes, ant_scores, ant_times = [], [], []

                for ant in range(n_ants):
                    current_idx = random.choice(start_edges)
                    route = [current_idx]
                    visited = {current_idx}
                    time_used = 0.0

                    while True:
                        current_end = oriented_edges[current_idx][1]
                        if current_end not in G:
                            break

                        neighbors = list(G.neighbors(current_end))
                        candidates = []
                        for nb in neighbors:
                            edge = (current_end, nb)
                            if edge in oriented_index_map:
                                e_idx = oriented_index_map[edge]
                                if e_idx not in visited:
                                    candidates.append(e_idx)

                        if not candidates:
                            break

                        phs = np.array([tau[c] for c in candidates], dtype=float)
                        lengths = np.array([edge_weight[oriented_edges[c]] for c in candidates], dtype=float)
                        heur = 1.0 / (lengths + epsilon)
                        uncovered_bonus = np.array([1.5 if c not in covered_edges else 1.0 for c in candidates])
                        probs_raw = (phs ** ALPHA) * (heur ** BETA) * uncovered_bonus
                        s = probs_raw.sum()

                        if s <= 0 or np.isnan(s):
                            break
                        probs = probs_raw / s
                        probs = probs / probs.sum()
                        if not np.isclose(probs.sum(), 1.0):
                            probs = np.ones(len(candidates)) / len(candidates)

                        next_idx = np.random.choice(candidates, p=probs)
                        edge_m = edge_weight[oriented_edges[next_idx]]
                        t_edge = travel_time_hours(edge_m)

                        next_end = oriented_edges[next_idx][1]
                        try:
                            ret_m = nx.shortest_path_length(G, source=next_end, target=depot_node, weight="weight")
                        except nx.NetworkXNoPath:
                            ret_m = float("inf")
                        t_return = travel_time_hours(ret_m)

                        if time_used + t_edge + t_return > time_limit_h:
                            break

                        route.append(next_idx)
                        visited.add(next_idx)
                        time_used += t_edge
                        current_idx = next_idx

                    # Reward exploring new edges efficiently, penalize underusing time
                    new_edges = len([e for e in route if e not in covered_edges])
                    time_ratio = min(1.0, time_used / time_limit_h)

                    # Balanced scoring: prioritize edge coverage and effective use of time
                    score = new_edges * (1.0 - 0.5 * time_ratio)

                    ant_routes.append(route)
                    ant_scores.append(score)
                    ant_times.append(time_used)

                if ant_scores:
                    iter_best_idx = int(np.argmax(ant_scores))
                    iter_best_score = ant_scores[iter_best_idx]
                    iter_best_route = ant_routes[iter_best_idx]
                    iter_best_time = ant_times[iter_best_idx]

                    for e_idx in iter_best_route:
                        tau[e_idx] += iter_best_score / (1.0 + iter_best_time)

                    if iter_best_score > best_vehicle_score:
                        best_vehicle_score = iter_best_score
                        best_vehicle_route = iter_best_route[:]
                        best_vehicle_time = iter_best_time

                if iteration % max(1, n_iter // 5) == 0:
                    print(f"    Iter {iteration+1}/{n_iter} – Vehicle {vehicle_num+1}: "
                          f"best score {best_vehicle_score:.2f}, time {best_vehicle_time or 0:.2f}h")

            if best_vehicle_route:
                newly_covered = [e for e in best_vehicle_route if e not in covered_edges]
                covered_edges.update(newly_covered)
                vehicle_routes.append((vehicle_num, best_vehicle_route, best_vehicle_time, len(newly_covered)))
                print(f"  Vehicle {vehicle_num+1}: covered {len(newly_covered)} new edges, time {best_vehicle_time:.2f}h")
            else:
                vehicle_routes.append((vehicle_num, [], 0.0, 0))
                print(f"  Vehicle {vehicle_num+1}: no feasible route found")

        total_possible = len(oriented_edges)
        results.append((depot_idx, vehicle_routes, len(covered_edges), total_possible))
        print(f"Depot {depot_idx+1} summary: covered {len(covered_edges)}/{total_possible} oriented edges")

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

def plot_results_multi(routes_gdf, depot_gdf, oriented_edges, results):
    fig, ax = plt.subplots(figsize=(12, 10))
    routes_gdf.plot(ax=ax, color="lightgray", linewidth=1.2, alpha=0.8, zorder=1)

    # plot depots
    if not depot_gdf.empty:
        depot_gdf.plot(ax=ax, color="orange", markersize=70, marker="*", edgecolor="black", zorder=4, label="Depots")

    cmap = plt.cm.get_cmap("tab10")
    for ridx, (depot_idx, vehicle_routes, covered, total) in enumerate(results):
        color = cmap(ridx % 10)
        for vnum, route, time_used, new_count in vehicle_routes:
            if not route:
                continue
            # plot sequence of oriented edges as segments
            for e_idx in route:
                a, b = oriented_edges[e_idx]
                ax.plot([a[0], b[0]], [a[1], b[1]], color=color, linewidth=2.5, alpha=0.9, zorder=3)
        # label
        ax.text(0.01, 0.98 - ridx*0.03, f"Depot {depot_idx+1}: covered {covered} oriented edges", transform=ax.transAxes, color=color, fontsize=9)

    ax.set_aspect("equal")
    ax.set_title("Multi-Depot Multi-Vehicle ACO Results")
    plt.legend()
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
    # Show subnetworks (each in different colour) and list counts
    # -------------------------
    plot_graph_subnetworks(
        G,
        depot_nodes=[(p.geometry.x, p.geometry.y) for _, p in depot_gdf.iterrows()],
        title="Network Subnetworks (before connection)"
    )

    # -------------------------
    # Connect all subnetworks together (bridge shortest gaps)
    # -------------------------
    # Load from cache if available, otherwise connect and save
    bridge_gdf = gpd.GeoDataFrame(geometry=[], crs="EPSG:3857")
    if GRAPH_CACHE_PATH.exists():
        try:
            print(f"Loading connected graph from cache: {GRAPH_CACHE_PATH}")
            with open(GRAPH_CACHE_PATH, "rb") as f:
                G, bridge_gdf = pickle.load(f)
            print("Loaded cached connected graph.")
        except Exception as e:
            print(f"Failed to load cache ({e}), will rebuild bridges.")
            G, bridge_gdf = connect_subnetworks(G)
            # save
            with open(GRAPH_CACHE_PATH, "wb") as f:
                pickle.dump((G, bridge_gdf), f)
            print(f"Saved connected graph to {GRAPH_CACHE_PATH}")
    else:
        G, bridge_gdf = connect_subnetworks(G)
        # save
        try:
            with open(GRAPH_CACHE_PATH, "wb") as f:
                pickle.dump((G, bridge_gdf), f)
            print(f"Saved connected graph to {GRAPH_CACHE_PATH}")
        except Exception as e:
            print(f"Warning: could not save connected graph cache: {e}")

    # optionally save bridge GeoJSON for reuse in GIS
    try:
        if not bridge_gdf.empty:
            bridge_gdf.to_file(SAVE_CONNECTED_GEOJSON, driver="GeoJSON")
            print(f"Saved added bridge edges to {SAVE_CONNECTED_GEOJSON}")
    except Exception as e:
        print(f"Could not save bridge geojson: {e}")

    # Plot final fully connected network
    fig, ax = plt.subplots(figsize=(10, 8))
    # Original routes (gray)
    routes_gdf.plot(ax=ax, color="lightgray", linewidth=0.8, alpha=0.6)
    # Bridge edges (red)
    if not bridge_gdf.empty:
        bridge_gdf.plot(ax=ax, color="red", linewidth=1.2, label="Added Bridge Edges", zorder=5)
    plt.title("Unified Network After Connecting Subnetworks")
    plt.axis("equal")
    plt.legend()
    plt.tight_layout()
    plt.show()

    depot_nodes = connect_depots_to_graph(G, depot_gdf)  # adds depot nodes and edges to G

    # build oriented edges and adjacency
    oriented_edges, start_map, edge_weight = make_oriented_edges(G)

    if ACO_ENABLED and len(depot_nodes) > 0:
        results = multi_depot_multi_vehicle_aco(
            G, depot_nodes,
            oriented_edges, start_map, edge_weight,
            vehicles_per_depot=VEHICLES_PER_DEPOT,
            n_ants=N_ANTS, n_iter=N_ITERATIONS, time_limit_h=TIME_LIMIT_HOURS
        )
        print("\nACO finished for all depots.")
        plot_results_multi(routes_gdf, depot_gdf, oriented_edges, results)
    else:
        print("ACO disabled or no depots found; only initial map shown.")

if __name__ == "__main__":
    main()
