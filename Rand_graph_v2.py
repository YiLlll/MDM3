import networkx as nx
import random
import matplotlib.pyplot as plt
import numpy as np

# ================================================================
# 1. Generate a connected random weighted graph
# ================================================================
def generate_random_weighted_graph(num_nodes=8, edge_prob=0.4, weight_range=(1, 10)):
    G = nx.Graph()
    G.add_nodes_from(range(num_nodes))
    for i in range(num_nodes):
        for j in range(i + 1, num_nodes):
            if random.random() < edge_prob:
                G.add_edge(i, j, weight=random.randint(*weight_range))
    for node in G.nodes():
        if G.degree(node) == 0:
            other = random.choice([n for n in G.nodes() if n != node])
            G.add_edge(node, other, weight=random.randint(*weight_range))
    while not nx.is_connected(G):
        components = list(nx.connected_components(G))
        comp_a = random.choice(components)
        comp_b = random.choice([c for c in components if c != comp_a])
        u = random.choice(list(comp_a))
        v = random.choice(list(comp_b))
        G.add_edge(u, v, weight=random.randint(*weight_range))
    return G

# ================================================================
# 2. Plot original graph
# ================================================================
def plot_original_graph(G):
    pos = nx.spring_layout(G, seed=42)
    plt.figure(figsize=(6, 5))
    nx.draw(G, pos, with_labels=True, node_color="lightblue", node_size=400)
    edge_labels = nx.get_edge_attributes(G, 'weight')
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)
    plt.title("Original Random Weighted Graph")
    plt.axis("off")
    plt.show()
    return pos

# ================================================================
# 3. Generate routes with vehicle distance limits
# ================================================================
def generate_routes(G, depot_nodes, D_max):
    depot_routes = {d: [] for d in depot_nodes}
    depot_weights = {d: 0 for d in depot_nodes}
    vehicles_per_depot = {d: 0 for d in depot_nodes}
    traversed_edges = set()  # edges actually traveled

    for depot in depot_nodes:
        remaining_edges = set(tuple(sorted(e)) for e in G.edges())
        vehicle_routes = []

        while remaining_edges:
            current_route = []
            remaining_distance = D_max
            current_position = depot
            progress = False

            for u, v in list(remaining_edges):
                edge_wt = G[u][v]['weight']

                # Edge must be reachable from current vehicle
                if edge_wt <= remaining_distance and (current_position == u or current_position == v):
                    current_route.append((u, v))
                    traversed_edges.add(tuple(sorted((u, v))))
                    remaining_distance -= edge_wt
                    current_position = v if current_position == u else u
                    remaining_edges.remove(tuple(sorted((u, v))))
                    progress = True

            if current_route:
                vehicle_routes.append(current_route)
                vehicles_per_depot[depot] += 1

            if not progress:
                # No more reachable edges from this depot
                break

        depot_routes[depot] = vehicle_routes

    return depot_routes, depot_weights, vehicles_per_depot, traversed_edges

# ================================================================
# 4. Fitness function
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
# 5. GA components
# ================================================================
def initialize_population(G, num_depots, pop_size):
    nodes = list(G.nodes())
    return [random.sample(nodes, num_depots) for _ in range(pop_size)]

def crossover(p1, p2, G):
    child = list(set(p1[:len(p1)//2] + p2[len(p2)//2:]))
    nodes = list(G.nodes())
    while len(child) < len(p1):
        n = random.choice(nodes)
        if n not in child:
            child.append(n)
    return child[:len(p1)]

def mutate(chromosome, mutation_rate, G):
    if random.random() < mutation_rate:
        available_nodes = [n for n in G.nodes() if n not in chromosome]
        if available_nodes:
            idx = random.randint(0, len(chromosome)-1)
            chromosome[idx] = random.choice(available_nodes)
    return chromosome

def select_parents(population, scores):
    idx = np.argsort(scores)[-2:]
    return population[idx[0]], population[idx[1]]

# ================================================================
# 6. Evaluate population
# ================================================================
def evaluate_population(G, population, D_max):
    scores, coverages, vehicles, per_depot_vehicles, per_depot_weights, avg_weights, max_weights = [], [], [], [], [], [], []

    for depots in population:
        depot_routes, depot_weights, vehicles_pd, traversed_edges = generate_routes(G, depots, D_max)
        score = fitness(G, depot_routes, D_max)

        coverage = (len(traversed_edges) / len(G.edges())) * 100
        total_vehicles = sum(vehicles_pd.values())

        all_vehicle_distances = []
        for routes in depot_routes.values():
            for vehicle_edges in routes:
                dist = sum(G[u][v]['weight'] for u, v in vehicle_edges if G.has_edge(u, v))
                all_vehicle_distances.append(dist)

        avg_weight = sum(depot_weights.values()) / total_vehicles if total_vehicles > 0 else 0
        max_vehicle_weight = max(all_vehicle_distances) if all_vehicle_distances else 0

        scores.append(score)
        coverages.append(coverage)
        vehicles.append(total_vehicles)
        per_depot_vehicles.append([vehicles_pd[d] for d in depots])
        per_depot_weights.append([depot_weights[d] for d in depots])
        avg_weights.append(avg_weight)
        max_weights.append(max_vehicle_weight)

    return scores, coverages, vehicles, per_depot_vehicles, per_depot_weights, avg_weights, max_weights

# ================================================================
# 7. GA execution
# ================================================================
def genetic_algorithm(G, num_depots=3, pop_size=10, generations=20, D_max=40, mutation_rate=0.1):
    population = initialize_population(G, num_depots, pop_size)
    best_score = -1
    best_solution = None
    best_depot_routes = None

    for gen in range(generations):
        scores, coverages, vehicles, per_depot_vehicles, per_depot_weights, avg_weights, max_weights = evaluate_population(G, population, D_max)
        gen_best_idx = np.argmax(scores)
        gen_best_score = scores[gen_best_idx]
        gen_best = population[gen_best_idx]

        if gen_best_score > best_score:
            best_score = gen_best_score
            best_solution = gen_best
            best_depot_routes, _, _, _ = generate_routes(G, best_solution, D_max)

        print(f"Gen {gen+1}: Fit={gen_best_score:.3f}, Cov={coverages[gen_best_idx]:.2f}%, "
              f"Veh={vehicles[gen_best_idx]}, PerDepotVeh={per_depot_vehicles[gen_best_idx]}, "
              f"PerDepotWt={per_depot_weights[gen_best_idx]}, "
              f"AvgWt/Veh={avg_weights[gen_best_idx]:.2f}, MaxVehWt={max_weights[gen_best_idx]:.2f}, "
              f"Depots={gen_best}")

        new_population = []
        for _ in range(pop_size // 2):
            p1, p2 = select_parents(np.array(population), np.array(scores))
            child1 = mutate(crossover(p1, p2, G), mutation_rate, G)
            child2 = mutate(crossover(p2, p1, G), mutation_rate, G)
            new_population.extend([child1, child2])
        population = new_population

    # Final best solution
    best_solution = [int(d) for d in best_solution]
    depot_routes, depot_weights, vehicles_pd, traversed_edges = generate_routes(G, best_solution, D_max)
    total_vehicles = sum(vehicles_pd.values())
    all_vehicle_distances = [sum(G[u][v]['weight'] for u,v in vehicle_edges if G.has_edge(u,v))
                             for routes in depot_routes.values() for vehicle_edges in routes]
    avg_distance = np.mean(all_vehicle_distances) if all_vehicle_distances else 0
    max_distance = max(all_vehicle_distances) if all_vehicle_distances else 0
    coverage = (len(traversed_edges) / len(G.edges())) * 100

    print("\n=== Best Overall Solution ===")
    print(f"Depots: {best_solution}")
    print(f"Fitness Score: {best_score:.3f}")
    print(f"Coverage: {coverage:.2f}%")
    print(f"Total Vehicles: {total_vehicles}")
    print(f"Vehicles per Depot: {[vehicles_pd[d] for d in best_solution]}")
    print(f"Distance per Depot: {[depot_weights[d] for d in best_solution]}")
    print(f"Average Distance per Vehicle: {avg_distance:.2f}")
    print(f"Maximum Distance of Any Vehicle: {max_distance:.2f}")

    return best_solution, depot_routes

# ================================================================
# 8. Plot graph with depot nodes red, edges colored by depot
# ================================================================
def plot_graph_with_colored_edges(G, depot_nodes, depot_routes, pos):
    nx.draw_networkx_nodes(G, pos, nodelist=[n for n in G.nodes if n not in depot_nodes],
                           node_color="lightblue", node_size=400)
    nx.draw_networkx_nodes(G, pos, nodelist=depot_nodes, node_color="red", node_size=500)
    nx.draw_networkx_edges(G, pos, edgelist=G.edges(), edge_color='lightgray', width=2, alpha=0.5)

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

    nx.draw_networkx_labels(G, pos, font_size=8)
    edge_labels = nx.get_edge_attributes(G, 'weight')
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)
    plt.title("Graph with Depot Routes Colored")
    plt.axis("off")
    plt.show()

# ================================================================
# 9. Run example
# ================================================================
if __name__ == "__main__":
    G = generate_random_weighted_graph(num_nodes=6, edge_prob=0.3, weight_range=(1, 15))
    pos = plot_original_graph(G)

    D_max = 20
    best_depots, best_depot_routes = genetic_algorithm(G, num_depots=1, pop_size=12, generations=15, D_max=D_max)
    plot_graph_with_colored_edges(G, best_depots, best_depot_routes, pos)
