import networkx as nx
import random
import matplotlib.pyplot as plt
import numpy as np

# === CONSTANTS ===
ANT_MAX_DISTANCE = 30

# === GRAPH GENERATION ===
def generate_random_weighted_graph(num_nodes=8, edge_prob=0.4, weight_range=(1, 10)):
    G = nx.Graph()
    G.add_nodes_from(range(num_nodes))

    # Add random edges
    for i in range(num_nodes):
        for j in range(i + 1, num_nodes):
            if random.random() < edge_prob:
                G.add_edge(i, j, weight=random.randint(*weight_range))

    # Ensure each node has at least one edge
    for node in G.nodes():
        if G.degree(node) == 0:
            other = random.choice([n for n in G.nodes() if n != node])
            G.add_edge(node, other, weight=random.randint(*weight_range))

    # Ensure full connectivity
    while not nx.is_connected(G):
        comps = list(nx.connected_components(G))
        c1, c2 = random.sample(comps, 2)
        u, v = random.choice(list(c1)), random.choice(list(c2))
        G.add_edge(u, v, weight=random.randint(*weight_range))

    return G

# === FUNCTION: Add depots on edges ===
def add_depots_on_edges(G, depot_specs, original_pos):
    """
    depot_specs: [(u, v, pos_fraction), ...]
    original_pos: positions of original nodes
    returns: new graph with added depot nodes and their positions
    """
    G2 = G.copy()
    next_depot_id = max(G2.nodes()) + 1
    pos2 = original_pos.copy()

    depot_nodes = []
    for (u, v, pos) in depot_specs:
        if not G2.has_edge(u, v):
            continue  # invalid edge
        w = G2[u][v]["weight"]
        G2.remove_edge(u, v)

        # Create depot node
        depot_id = next_depot_id
        next_depot_id += 1

        G2.add_node(depot_id, depot=True)

        # Split edge
        w1 = max(w * pos, 0.001)    
        w2 = max(w * (1 - pos), 0.001)
        
        G2.add_edge(u, depot_id, weight=w1)
        G2.add_edge(depot_id, v, weight=w2)

        # Position depot node along the original edge
        x1, y1 = original_pos[u]
        x2, y2 = original_pos[v]
        depot_x = x1 + pos * (x2 - x1)
        depot_y = y1 + pos * (y2 - y1)
        pos2[depot_id] = (depot_x, depot_y)

        depot_nodes.append(depot_id)

    return G2, depot_nodes, pos2

# === ANT COLONY SIMULATION ===
def ant_colony_cover(G, nests, ants_per_nest, max_distance=ANT_MAX_DISTANCE,
                     pher_decay=0.9, pher_deposit=2.0, steps=1000, seed=None):
    if seed is not None:
        random.seed(seed)
    
    pheromone = {frozenset((u, v)): 1.0 for u, v in G.edges()}
    visited_edges = set()
    edge_owner = {}

    ants = []
    for depot_idx, (nest, num_ants) in enumerate(zip(nests, ants_per_nest)):
        for _ in range(num_ants):
            ants.append({"pos": nest, "remaining": max_distance, "path": [], "owner": depot_idx})

    for step in range(steps):
        for ant in ants:
            if ant["remaining"] <= 0:
                continue

            neighbors = list(G.neighbors(ant["pos"]))
            if not neighbors:
                continue

            valid_choices = []
            for n in neighbors:
                dist = G[ant["pos"]][n]["weight"]
                if ant["remaining"] >= dist and dist > 0:
                    edge = frozenset((ant["pos"], n))
                    p = pheromone[edge]
                    desirability = (1.0 / max(dist, 1e-6)) * (1.0 / max(p, 1e-6))
                    valid_choices.append((n, edge, dist, desirability))
            if not valid_choices:
                continue

            total_des = sum(d[3] for d in valid_choices)
            probs = [d[3] / total_des for d in valid_choices]
            next_node, edge, dist, _ = random.choices(valid_choices, weights=probs, k=1)[0]

            ant["remaining"] -= dist
            ant["path"].append(edge)
            ant["pos"] = next_node
            visited_edges.add(edge)
            if edge not in edge_owner:
                edge_owner[edge] = ant["owner"]
            pheromone[edge] += pher_deposit

        for e in pheromone:
            pheromone[e] = max(pheromone[e] * pher_decay, 0.1)

        if len(visited_edges) == len(G.edges()):
            print(f"  All edges covered at step {step}")
            break

    # Calculate both coverage metrics
    total_edges = len(G.edges())
    covered_edges = len(visited_edges)
    edge_count_coverage = 100 * covered_edges / total_edges
    
    # Calculate weight coverage
    total_weight = sum(data["weight"] for _, _, data in G.edges(data=True))
    covered_weight = sum(G[u][v]["weight"] for u, v in G.edges() if frozenset((u, v)) in visited_edges)
    weight_coverage = 100 * covered_weight / total_weight if total_weight > 0 else 0

    return visited_edges, edge_count_coverage, weight_coverage, covered_weight, total_weight, edge_owner

# === COST FUNCTION ===
def evaluate_solution(G, depot_specs, ants_per_depot, original_pos,  # ← ADD original_pos parameter
                      max_distance=ANT_MAX_DISTANCE, w_ants=1.0, w_travel=0.3, w_cov=5.0, evaluation_seed=42):
    
    # Create new graph with depot nodes added USING THE SAME LAYOUT
    G2, depots, _ = add_depots_on_edges(G, depot_specs, original_pos)  # ← Use consistent layout

    print(f"  Evaluating: {len(depots)} depots, {sum(ants_per_depot)} ants")
    visited_edges, edge_coverage, weight_coverage, covered_weight, total_weight, _ = ant_colony_cover(
        G2, nests=depots, ants_per_nest=ants_per_depot, max_distance=max_distance, seed=evaluation_seed
    )
    total_ants = sum(ants_per_depot)
    avg_travel = covered_weight / total_ants if total_ants > 0 else 0

    # PRIORITIZE WEIGHT COVERAGE
    if weight_coverage >= 100:
        cost = w_ants * total_ants + w_travel * avg_travel
    else:
        penalty = (100 - weight_coverage) * w_cov * 10
        cost = penalty + w_ants * total_ants + w_travel * avg_travel

    print(f"  Result: Edge={edge_coverage:.1f}%, Weight={weight_coverage:.1f}%, Cost={cost:.2f}")
    return cost, edge_coverage, weight_coverage, total_ants

# === GENETIC ALGORITHM ===
def initialize_population(G, num_depots=3, pop_size=10):
    edges = list(G.edges())
    population = []
    for _ in range(pop_size):
        depot_specs = []
        for _ in range(num_depots):
            u, v = random.choice(edges)
            pos = random.random()
            depot_specs.append((u, v, pos))
        ants = [random.randint(2, 6) for _ in range(num_depots)]
        population.append({"depot_specs": depot_specs, "ants": ants})
    return population

def crossover(p1, p2):
    depots = []
    for d1, d2 in zip(p1["depot_specs"], p2["depot_specs"]):
        chosen = random.choice([d1, d2])
        if random.random() < 0.5:
            chosen = (chosen[0], chosen[1], min(1, max(0, chosen[2] + np.random.normal(0, 0.1))))
        depots.append(chosen)
    ants = [random.choice([a1, a2]) for a1, a2 in zip(p1["ants"], p2["ants"])]
    return {"depot_specs": depots, "ants": ants}

def mutate(ind, G, rate=0.3):
    edges = list(G.edges())
    if random.random() < rate:
        i = random.randrange(len(ind["depot_specs"]))
        u, v = random.choice(edges)
        pos = random.random()
        ind["depot_specs"][i] = (u, v, pos)
    if random.random() < rate:
        i = random.randrange(len(ind["ants"]))
        ind["ants"][i] = max(1, ind["ants"][i] + random.choice([-1, 1]))
    return ind

def optimize_depots_ga(G, original_pos, num_depots=3, generations=20, pop_size=10):  # ← ADD original_pos
    population = initialize_population(G, num_depots, pop_size)
    best_ind, best_cost = None, float("inf")
    best_coverage_results = None

    for gen in range(generations):
        scored = []
        for ind in population:
            cost, edge_coverage, weight_coverage, total_ants = evaluate_solution(
                G, ind["depot_specs"], ind["ants"], original_pos  # ← Pass original_pos
            )
            scored.append((ind, cost, edge_coverage, weight_coverage, total_ants))
            if cost < best_cost:
                best_cost, best_ind = cost, ind.copy()
                best_coverage_results = (edge_coverage, weight_coverage, total_ants)

        scored.sort(key=lambda x: x[1])
        elites = [scored[0][0]]
        new_pop = elites.copy()

        while len(new_pop) < pop_size:
            parents = random.sample(scored[:5], 2)
            child = crossover(parents[0][0], parents[1][0])
            child = mutate(child, G)
            new_pop.append(child)
        population = new_pop

        print(f"Gen {gen+1}: Best Cost={scored[0][1]:.2f}, "
              f"Edge Coverage={scored[0][2]:.1f}%, "
              f"Weight Coverage={scored[0][3]:.1f}%, "
              f"Ants={scored[0][4]}")

    print("\n=== FINAL BEST ===")
    print(f"Cost: {best_cost:.2f}")
    print(f"Depots: {best_ind['depot_specs']}")
    print(f"Ants per depot: {best_ind['ants']}")
    print(f"Total ants used: {sum(best_ind['ants'])}")
    
    if best_coverage_results:
        edge_cov, weight_cov, ants_used = best_coverage_results
        print(f"Best Achieved During GA: Edge Coverage={edge_cov:.1f}%, Weight Coverage={weight_cov:.1f}%")
    
    return best_ind, best_coverage_results

# === VISUALIZATION ===
def draw_graph(G, pos, title):
    plt.figure(figsize=(10, 8))
    nx.draw(G, pos, with_labels=True, node_color="skyblue", edge_color="gray",
            width=2, node_size=800, edgecolors="black")
    labels = nx.get_edge_attributes(G, "weight")
    for edge, weight in labels.items():
        x = (pos[edge[0]][0] + pos[edge[1]][0]) / 2
        y = (pos[edge[0]][1] + pos[edge[1]][1]) / 2
        plt.text(x, y, f"{weight}", fontsize=8, 
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.7),
                horizontalalignment='center', verticalalignment='center')
    plt.title(title)
    plt.axis("off")
    plt.show()

def draw_final_graph(G, pos, edge_owner, depots):
    plt.figure(figsize=(10, 8))
    colors = plt.cm.tab10(np.linspace(0, 1, len(depots)))
    edge_colors = []
    edge_widths = []
    
    for u, v in G.edges():
        edge = frozenset((u, v))
        if edge in edge_owner:
            edge_colors.append(colors[edge_owner[edge]])
            edge_widths.append(2.5)
        else:
            edge_colors.append("gray")
            edge_widths.append(1.5)
    
    node_colors = []
    node_sizes = []
    for node in G.nodes():
        if node in depots:
            node_colors.append("red")
            node_sizes.append(600)
        else:
            node_colors.append("skyblue")
            node_sizes.append(800)
    
    nx.draw(G, pos, with_labels=True, node_color=node_colors, edge_color=edge_colors,
            width=edge_widths, node_size=node_sizes, edgecolors="black")
    
    edge_labels = nx.get_edge_attributes(G, "weight")
    for edge, weight in edge_labels.items():
        x = (pos[edge[0]][0] + pos[edge[1]][0]) / 2
        y = (pos[edge[0]][1] + pos[edge[1]][1]) / 2
        plt.text(x, y, f"{weight}", fontsize=8, 
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.7),
                horizontalalignment='center', verticalalignment='center')
    
    plt.title("Final Graph with Depot Coverage")
    plt.axis("off")
    plt.show()

# === MAIN ===
if __name__ == "__main__":
    random.seed(42)
    np.random.seed(42)
    
    G = generate_random_weighted_graph(num_nodes=20, edge_prob=0.5, weight_range=(5, 20))
    
    # Generate and save the original layout ONCE
    original_pos = nx.spring_layout(G, seed=42)
    
    draw_graph(G, original_pos, "Initial Weighted Graph")

    # Pass the original layout to GA
    best_solution, ga_coverage_results = optimize_depots_ga(G, original_pos, num_depots=15, generations=30, pop_size=15)

    # Build final graph using the SAME original layout
    G2, depots, final_pos = add_depots_on_edges(G, best_solution["depot_specs"], original_pos)
    
    # Final run with same parameters
    visited_edges, edge_coverage, weight_coverage, covered_weight, total_weight, edge_owner = ant_colony_cover(
        G2, nests=depots, ants_per_nest=best_solution["ants"], 
        max_distance=ANT_MAX_DISTANCE,
        seed=42
    )

    print(f"\n=== FINAL RESULTS ===")
    print(f"Edge Coverage: {edge_coverage:.1f}% ({len(visited_edges)}/{len(G2.edges())} edges)")
    print(f"Weight Coverage: {weight_coverage:.1f}% ({covered_weight:.1f}/{total_weight:.1f} weight units)")
    print(f"Depot nodes: {depots}")
    
    if ga_coverage_results:
        ga_edge_cov, ga_weight_cov, _ = ga_coverage_results
        total_ants_final = sum(best_solution["ants"])
        avg_weight_per_ant = covered_weight / total_ants_final if total_ants_final > 0 else 0
        
        print(f"\n=== COMPARISON WITH GA EVALUATION ===")
        print(f"GA Evaluation:  Edge Coverage={ga_edge_cov:.1f}%, Weight Coverage={ga_weight_cov:.1f}%, " 
              f"Average weight per ant={covered_weight / sum(best_solution['ants']):.2f}")
        print(f"Final Run:      Edge Coverage={edge_coverage:.1f}%, Weight Coverage={weight_coverage:.1f}%, "
              f"Average weight per ant={avg_weight_per_ant:.2f}")
    
    
    draw_final_graph(G2, final_pos, edge_owner, depots)