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
def load_connected_graph(processed_path="processed_connected_network_full.geojson",
                         original_path="roads.geojson.json"):
    if os.path.exists(processed_path):
        print(f"✅ Found processed file: {processed_path}")
        gdf = gpd.read_file(processed_path)
    else:
        print(f"⚙️ Building processed file from {original_path} ...")
        gdf = gpd.read_file(original_path)

        # Filter to Southwest UK region
        minx, miny, maxx, maxy = 0.0, 51.51, 1.8, 53.2
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
# 4. Generate routes with distance tracking
# ================================================================
def generate_routes(G, depot_nodes, D_max, max_vehicles_per_depot=None):
    depot_routes = {d: [] for d in depot_nodes}
    depot_distances = {d: [] for d in depot_nodes}  # distances per depot
    traversed_edges = set()

    for depot in depot_nodes:
        remaining_edges = set(tuple(sorted(e)) for e in G.edges())  # reset for each depot
        while remaining_edges:
            route = []
            dist_used = 0.0  # track distance used for this vehicle
            visited_nodes = set([depot])
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
                        dist_used += w  # add edge distance
                        queue.append((neighbor, dist_left - w))

            if route:
                depot_routes[depot].append(route)
                depot_distances[depot].append(dist_used)  # save distance per vehicle
            else:
                break

            # Stop if maximum vehicles per depot is reached
            if max_vehicles_per_depot and len(depot_routes[depot]) >= max_vehicles_per_depot:
                break

    # Vehicle count for each depot is now just len(depot_routes[depot])
    return depot_routes, traversed_edges, depot_distances


# ================================================================
# 5. Fitness Function
# ================================================================
def fitness(G, depot_routes, D_max):
    total_edges = set(tuple(sorted(e)) for e in G.edges())
    covered_edges = set()
    vehicle_distances = []

    # --- Collect coverage and distances ---
    for depot, routes in depot_routes.items():
        for edges in routes:
            dist = sum(G[u][v]['weight'] for u, v in edges if G.has_edge(u, v))
            vehicle_distances.append(dist)
            for e in edges:
                if G.has_edge(*e):
                    covered_edges.add(tuple(sorted(e)))

    # --- Coverage calculation ---
    coverage_ratio = len(covered_edges) / len(total_edges)

    # Always prioritize coverage first
    # If coverage < 1, return that directly as fitness (scaled)
    if coverage_ratio < 1.0:
        return coverage_ratio  # prioritize improving coverage before efficiency

    # --- If full coverage achieved, reward efficiency and balance ---
    n_vehicles = len(vehicle_distances)
    avg_D = np.mean(vehicle_distances) if vehicle_distances else 0
    normalized_utilization = avg_D / D_max  # may exceed 1.0, that’s fine

    # Efficiency + vehicle usage bonus (applied only when full coverage)
    score = 1.0 + 0.8 * normalized_utilization - 0.2 * (n_vehicles / len(G.nodes()))

    # Add a strong base reward for full coverage
    return max(coverage_ratio + 0.5 * score, 0.0)

# ================================================================
# Save Best Depot Locations to File (in Lat/Lon)
# ================================================================
def save_best_solution(G, best_solution, fitness_score, coverage, total_vehicles, num_depots):
    import geopandas as gpd
    from shapely.geometry import Point

    filename = f"best_solution_{num_depots}_depots.txt"

    prev_fit = None
    prev_cov = None

    # === Check previous solution ===
    if os.path.exists(filename):
        with open(filename, "r") as f:
            lines = f.readlines()
            for line in lines:
                if line.startswith("Fitness:"):
                    prev_fit = float(line.split(":")[1].strip())
                elif line.startswith("Coverage:"):
                    prev_cov = float(line.split(":")[1].replace("%", "").strip())

        # === Improved comparison logic ===
        if prev_fit is not None and prev_cov is not None:
            if (fitness_score < prev_fit) or (coverage < prev_cov):
                print(f"⚠️ Existing best solution is better "
                      f"(Fitness {prev_fit:.3f}, Coverage {prev_cov:.2f}%) — not overwriting {filename}")
                return

    # === Convert depot locations to lat/lon ===
    gdf = gpd.GeoDataFrame(geometry=[Point(xy) for xy in best_solution], crs="EPSG:27700")
    gdf = gdf.to_crs(epsg=4326)
    latlon_coords = [(round(p.x, 6), round(p.y, 6)) for p in gdf.geometry]

    # === Write new best ===
    with open(filename, "w") as f:
        f.write(f"Best Solution for {num_depots} Depots\n")
        f.write(f"Fitness: {fitness_score:.3f}\n")
        f.write(f"Coverage: {coverage:.2f}%\n")
        f.write(f"Total Vehicles: {total_vehicles}\n\n")

        f.write("Depot Locations (British National Grid - meters):\n")
        for d in best_solution:
            f.write(f"  {tuple(map(float, d))}\n")

        f.write("\nDepot Locations (Latitude, Longitude):\n")
        for ll in latlon_coords:
            f.write(f"  {ll}\n")

    print(f"💾 Saved new best solution for {num_depots} depots → {filename}")
    print(f"🟢 Fitness: {fitness_score:.3f}, Coverage: {coverage:.2f}%")

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
        depot_routes, traversed_edges, depot_distances = generate_routes(G, depots, D_max)
        score = fitness(G, depot_routes, D_max)
        coverage = (len(traversed_edges) / len(G.edges())) * 100
        # Vehicle count = sum of number of routes per depot
        total_vehicles = sum(len(routes) for routes in depot_routes.values())
        scores.append(score)
        coverages.append(coverage)
        vehicles.append(total_vehicles)

    return scores, coverages, vehicles

# ================================================================
# 7. GA Main
# ================================================================
def genetic_algorithm(G, num_depots=3, pop_size=10, generations=10, D_max=5000, mutation_rate=0.1):
    """
    Run the GA to optimize depot locations and routes.
    Outputs per-depot vehicle distances and correct vehicle counts.
    Also records coverage per generation for plotting.
    """
    population = initialize_population(G, num_depots, pop_size)
    best_score = -1
    best_solution = None
    best_depot_routes = None
    best_depot_distances = None
    best_coverage = 0.0

    # ✅ Track coverage each generation
    coverage_history = []

    for gen in range(generations):
        scores, coverages, vehicles = evaluate_population(G, population, D_max)

        max_cov = max(coverages)
        indices_max_cov = [i for i, cov in enumerate(coverages) if cov == max_cov]

        if len(indices_max_cov) > 1:
            best_idx = indices_max_cov[np.argmax([scores[i] for i in indices_max_cov])]
        else:
            best_idx = indices_max_cov[0]

        gen_best_score = scores[best_idx]
        gen_best_cov = coverages[best_idx]
        gen_best = population[best_idx]

        # ✅ Record coverage for this generation
        coverage_history.append(gen_best_cov)

        if gen_best_cov > best_coverage or (gen_best_cov == best_coverage and gen_best_score > best_score):
            best_score = gen_best_score
            best_solution = gen_best
            best_depot_routes, traversed_edges, best_depot_distances = generate_routes(
                G, snap_to_graph_nodes(G, best_solution), D_max
            )
            best_coverage = gen_best_cov

        print(f"Gen {gen+1}: Fit={gen_best_score:.3f}, Cov={gen_best_cov:.2f}%, Veh={vehicles[best_idx]} Depots={best_solution}")

        # Generate next population
        new_population = []
        for _ in range(pop_size // 2):
            p1, p2 = select_parents(np.array(population), np.array(scores))
            child1 = mutate(crossover(p1, p2, G), mutation_rate, G)
            child2 = mutate(crossover(p2, p1, G), mutation_rate, G)
            new_population.extend([child1, child2])
        population = new_population

    # Correct vehicle count using number of routes per depot
    total_vehicles = sum(len(routes) for routes in best_depot_routes.values())

    print("\n=== Best Overall Solution ===")
    print(f"Depots: {[tuple(map(float, d)) for d in best_solution]}")
    print(f"Fitness: {best_score:.3f}")
    print(f"Coverage: {best_coverage:.2f}%")
    print(f"Total Vehicles Used: {total_vehicles}")

    # Save best solution to file
    save_best_solution(G, best_solution, best_score, best_coverage, total_vehicles, num_depots)
    # ✅ Plot coverage history
    plt.figure(figsize=(8, 5))  # ensure it's a new independent figure
    plt.scatter(range(1, len(coverage_history) + 1), coverage_history, c='blue', label='Coverage % per generation')
    plt.plot(range(1, len(coverage_history) + 1), coverage_history, linestyle='--', alpha=0.7)
    plt.xlabel("Generation")
    plt.ylabel("Coverage (%)")
    plt.title("Coverage Improvement Over Generations")
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend()
    plt.show(block=False)  # 👈 show but don’t block program

    
    return best_solution, best_depot_routes, total_vehicles, best_depot_distances



    

# ================================================================
# 8. Plot Results
# ================================================================
def plot_graph_with_coloured_edges(G, depot_nodes, depot_routes, pos):
    nx.draw(G, pos, node_color="lightgray", node_size=10, edge_color="lightgray", alpha=0.5)
    cmap = plt.cm.get_cmap('tab20', 20)
    
    vehicle_count = 0
    for i, depot in enumerate(depot_nodes):
        vehicle_routes = depot_routes[depot]
        for vehicle_edges in vehicle_routes:
            visited_nodes = set()
            for u, v in vehicle_edges:
                visited_nodes.add(u)
                visited_nodes.add(v)
            
            color = cmap(vehicle_count % 20)
            edges_to_draw = [e for e in vehicle_edges if G.has_edge(*e)]
            nx.draw_networkx_edges(G, pos, edgelist=edges_to_draw, edge_color=color, width=2.5)
            nx.draw_networkx_nodes(G, pos, nodelist=list(visited_nodes),
                                   node_color=[color]*len(visited_nodes),
                                   node_size=50, edgecolors="black")
            
            nx.draw_networkx_labels(G, pos, labels={depot: f"V{vehicle_count+1}"}, font_size=10, font_color='red')
            vehicle_count += 1
        
        nx.draw_networkx_nodes(G, pos, nodelist=[depot], node_color='black', node_size=80, edgecolors="yellow")

    plt.title("Optimized Depot Routes with Vehicle Numbers")
    plt.axis("off")
    plt.show()



# ================================================================
# 9. Run Example
# ================================================================
if __name__ == "__main__":
    G, depot_nodes = load_connected_graph("processed_connected_network_full.geojson", "roads.geojson.json")
    pos = {n: (n[0], n[1]) for n in G.nodes}
    plot_original_graph(G)

    D_max = 33333  # meters
    best_depots, best_depot_routes, total_vehicles, best_depot_distances = genetic_algorithm(
        G, num_depots=7, pop_size=15, generations=100, D_max=D_max)
    print(f"Vehicles used in final solution: {total_vehicles}")
    plot_graph_with_coloured_edges(G, best_depots, best_depot_routes, pos)

