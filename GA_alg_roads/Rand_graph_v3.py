import geopandas as gpd
import networkx as nx
from shapely.geometry import LineString
import os
import numpy as np
import random
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree
from collections import deque

# ================================================================
# 1. Load Weighted Graph
# ================================================================
def load_connected_graph(processed_path="area6_combined_routes.geojson",
                         original_path="roads.geojson.json"):
    if os.path.exists(processed_path):
        print(f"✅ Found processed file: {processed_path}")
        gdf = gpd.read_file(processed_path)
    else:
        print(f"⚙️ Building processed file from {original_path} ...")
        gdf = gpd.read_file(original_path)

        # Filter to Southwest UK region
        minx, miny, maxx, maxy = -6.0, 50.0, -2.75, 52.0
        subset = gdf.cx[minx:maxx, miny:maxy]

        # Convert CRS to British National Grid (EPSG:27700)
        subset = subset.to_crs(epsg=27700)
        subset.to_file(processed_path, driver="GeoJSON")
        print(f"💾 Saved processed network to {processed_path}")
        gdf = subset

    G = nx.Graph()
    for _, row in gdf.iterrows():
        geom = row.geometry
        if isinstance(geom, LineString):
            coords = list(geom.coords)
            for i in range(len(coords) - 1):
                p1, p2 = coords[i], coords[i + 1]
                G.add_edge(tuple(map(float, p1)), tuple(map(float, p2)),
                           weight=LineString([p1, p2]).length)
    print(f"✅ Graph ready: {len(G.nodes)} nodes, {len(G.edges)} edges.")
    depot_nodes = []  # empty for now
    return G, depot_nodes

# ================================================================
# 2. Plot Graph
# ================================================================
def plot_original_graph(G):
    pos = {n: (n[0], n[1]) for n in G.nodes}
    plt.figure(figsize=(7, 6))
    nx.draw(G, pos, node_color="lightblue", node_size=20, edge_color="gray", with_labels=False)
    plt.title("Loaded Road Network (GeoJSON)")
    plt.axis("off")
    plt.show()
    return pos

# ================================================================
# 3. Snap depots to nearest nodes
# ================================================================
def snap_to_graph_nodes(G, depots):
    nodes = np.array(list(G.nodes()))
    tree = cKDTree(nodes)
    snapped = []
    for d in depots:
        dist, idx = tree.query(d)
        snapped.append(tuple(nodes[idx]))
    return snapped

# ================================================================
# 4. Generate routes with full Dijkstra-style traversal
# ================================================================
def generate_routes(G, depot_nodes, D_max):
    depot_routes = {d: [] for d in depot_nodes}
    traversed_edges = set()

    for depot in depot_nodes:
        remaining_edges = set(tuple(sorted(e)) for e in G.edges())

        while remaining_edges:
            route = []
            visited_nodes = set([depot])
            # Queue: (current_node, remaining_distance)
            queue = deque([(depot, D_max)])

            while queue:
                node, dist_left = queue.popleft()
                for neighbor in G.neighbors(node):
                    edge = tuple(sorted((node, neighbor)))
                    w = G[node][neighbor]['weight']
                    if edge in remaining_edges and w <= dist_left:
                        route.append(edge)
                        traversed_edges.add(edge)
                        remaining_edges.remove(edge)
                        queue.append((neighbor, dist_left - w))

            if route:
                depot_routes[depot].append(route)
            else:
                break

    return depot_routes, traversed_edges

# ================================================================
# 5. Fitness Function
# ================================================================
def fitness(G, depot_routes, D_max):
    total_edges = set(tuple(sorted(e)) for e in G.edges())
    covered_edges = set()
    vehicle_distances = []

    for depot, routes in depot_routes.items():
        for edges in routes:
            dist = sum(G[u][v]['weight'] for u, v in edges if G.has_edge(u, v))
            vehicle_distances.append(dist)
            for e in edges:
                if G.has_edge(*e):
                    covered_edges.add(tuple(sorted(e)))
            if dist > D_max:
                return 0.0

    coverage_ratio = len(covered_edges) / len(total_edges)
    if coverage_ratio < 1.0:
        return 0.5 * coverage_ratio

    n_vehicles = len(vehicle_distances)
    avg_D = np.mean(vehicle_distances) if vehicle_distances else 0
    normalized_utilization = avg_D / D_max
    score = 1.0 + 0.8 * normalized_utilization - 0.2 * (n_vehicles / len(G.nodes()))
    return max(score, 0.0)

# ================================================================
# 6. GA Initialization and Operators
# ================================================================
def initialize_population(G, num_depots, pop_size):
    nodes = list(G.nodes())
    population = []
    for _ in range(pop_size):
        sample = random.sample(nodes, num_depots)
        population.append([tuple(map(float, n)) for n in sample])
    return population

def crossover(p1, p2, G):
    p1 = [tuple(map(float, n)) for n in p1]
    p2 = [tuple(map(float, n)) for n in p2]
    child = list(set(p1[:len(p1)//2] + p2[len(p2)//2:]))
    nodes = list(G.nodes())
    while len(child) < len(p1):
        n = random.choice(nodes)
        if n not in child:
            child.append(tuple(map(float, n)))
    return child[:len(p1)]

def mutate(chromosome, mutation_rate, G):
    if random.random() < mutation_rate:
        available_nodes = [n for n in G.nodes() if n not in chromosome]
        if available_nodes:
            idx = random.randint(0, len(chromosome)-1)
            chromosome[idx] = tuple(map(float, random.choice(available_nodes)))
    return chromosome

def select_parents(population, scores):
    idx = np.argsort(scores)[-2:]
    return population[idx[0]], population[idx[1]]

def evaluate_population(G, population, D_max):
    scores, coverages, vehicles = [], [], []

    for depots in population:
        depots = snap_to_graph_nodes(G, depots)
        depot_routes, traversed_edges = generate_routes(G, depots, D_max)
        score = fitness(G, depot_routes, D_max)
        coverage = (len(traversed_edges) / len(G.edges())) * 100
        total_vehicles = sum(len(routes) for routes in depot_routes.values())
        scores.append(score)
        coverages.append(coverage)
        vehicles.append(total_vehicles)

    return scores, coverages, vehicles

# ================================================================
# 7. GA Main
# ================================================================
def genetic_algorithm(G, num_depots=3, pop_size=10, generations=10, D_max=5000, mutation_rate=0.1):
    population = initialize_population(G, num_depots, pop_size)
    best_score = -1
    best_solution = None
    best_depot_routes = None
    best_coverage = 0.0

    for gen in range(generations):
        # Evaluate population
        scores, coverages, vehicles = evaluate_population(G, population, D_max)

        # --- Select the best individual based on coverage first, then fitness ---
        max_cov = max(coverages)
        indices_max_cov = [i for i, cov in enumerate(coverages) if cov == max_cov]

        if len(indices_max_cov) > 1:
            # Tie-breaker: pick the one with highest fitness among max coverage
            best_idx = indices_max_cov[np.argmax([scores[i] for i in indices_max_cov])]
        else:
            best_idx = indices_max_cov[0]

        gen_best_score = scores[best_idx]
        gen_best_cov = coverages[best_idx]
        gen_best = population[best_idx]

        # --- Update overall best solution ---
        if gen_best_cov > best_coverage or (gen_best_cov == best_coverage and gen_best_score > best_score):
            best_score = gen_best_score
            best_solution = gen_best
            best_depot_routes, traversed_edges = generate_routes(G, snap_to_graph_nodes(G, best_solution), D_max)
            best_coverage = gen_best_cov

        print(f"Gen {gen+1}: Fit={gen_best_score:.3f}, Cov={gen_best_cov:.2f}%, Veh={vehicles[best_idx]} Depots={best_solution}")

        # --- Create next generation ---
        new_population = []
        for _ in range(pop_size // 2):
            p1, p2 = select_parents(np.array(population), np.array(scores))
            child1 = mutate(crossover(p1, p2, G), mutation_rate, G)
            child2 = mutate(crossover(p2, p1, G), mutation_rate, G)
            new_population.extend([child1, child2])
        population = new_population

    # --- Final output ---
    print("\n=== Best Overall Solution ===")
    print(f"Depots: {[tuple(map(float, d)) for d in best_solution]}")
    print(f"Fitness: {best_score:.3f}")
    print(f"Coverage: {best_coverage:.2f}%")
    return best_solution, best_depot_routes


# ================================================================
# 8. Plot Results
# ================================================================
def plot_graph_with_colored_edges(G, depot_nodes, depot_routes, pos):
    nx.draw(G, pos, node_color="lightgray", node_size=10, edge_color="lightgray", alpha=0.5)
    cmap = plt.cm.get_cmap('tab10')
    depot_colors = {depot: cmap(i % 10) for i, depot in enumerate(depot_nodes)}

    for depot, routes in depot_routes.items():
        color = depot_colors[depot]
        edges_to_draw = []
        for vehicle_edges in routes:
            for e in vehicle_edges:
                if G.has_edge(*e):
                    edges_to_draw.append(e)
        nx.draw_networkx_edges(G, pos, edgelist=edges_to_draw, edge_color=color, width=2.5)
        nx.draw_networkx_nodes(G, pos, nodelist=[depot], node_color=color, node_size=50, edgecolors="black")

    plt.title("Optimized Depot Routes (GA on Real Network)")
    plt.axis("off")
    plt.show()

# ================================================================
# 9. Run Example
# ================================================================
if __name__ == "__main__":
    G, depot_nodes = load_connected_graph("area6_combined_routes.geojson", "roads.geojson.json")
    pos = {n: (n[0], n[1]) for n in G.nodes}
    plot_original_graph(G)

    D_max = 80000  # meters
    best_depots, best_depot_routes = genetic_algorithm(G, num_depots=2, pop_size=15, generations=200, D_max=D_max)
    plot_graph_with_colored_edges(G, best_depots, best_depot_routes, pos)
