import numpy as np
import random
import matplotlib.pyplot as plt
import networkx as nx
from matplotlib.lines import Line2D


# ----------------------------
# Grid-Based Road Network Setup with depot at center
# ----------------------------
def generate_grid_network(grid_size=5, n_customers=10, seed=42):
    """Generates a grid network with the depot at the center."""
    random.seed(seed)

    # Create a 2D grid graph
    G = nx.grid_2d_graph(grid_size, grid_size)
    G = nx.convert_node_labels_to_integers(G)
    for u, v in G.edges():
        G.edges[u, v]['weight'] = 1

    # Depot is center node
    center_coord = (grid_size // 2, grid_size // 2)
    coord_to_node = {(i % grid_size, i // grid_size): i for i in G.nodes()}
    depot = coord_to_node[center_coord]

    # Pick customers randomly from remaining nodes
    nodes = list(G.nodes())
    nodes.remove(depot)
    customers = random.sample(nodes, n_customers)

    return G, depot, customers


def compute_distance_matrix(G, nodes):
    """Compute shortest path distances only between selected nodes (depot + customers)"""
    n = len(nodes)
    D = np.zeros((n, n))
    node_index = {node: idx for idx, node in enumerate(nodes)}
    for i, node_i in enumerate(nodes):
        lengths = nx.single_source_dijkstra_path_length(G, node_i, weight='weight')
        for node_j, dist in lengths.items():
            if node_j in node_index:
                j = node_index[node_j]
                D[i, j] = dist
    return D


# ----------------------------
# Cost Calculation
# ----------------------------
def route_cost(routes, D):
    total = 0
    for r in routes:
        for i in range(len(r) - 1):
            total += D[int(r[i]), int(r[i + 1])]
    return total


# ----------------------------
# ACO for CVRP
# ----------------------------
class ACO_CVRP:
    def __init__(self, D, vehicle_capacity, n_ants=10, alpha=1.0, beta=5.0, rho=0.1, iterations=100):
        self.D = D
        self.n = len(D)
        self.vehicle_capacity = vehicle_capacity
        self.n_ants = n_ants
        self.alpha = alpha
        self.beta = beta
        self.rho = rho
        self.iterations = iterations
        self.pheromone = np.ones((self.n, self.n))
        self.best_cost = float("inf")
        self.best_routes = None

    def run(self):
        for it in range(self.iterations):
            all_solutions = []
            all_costs = []
            for ant in range(self.n_ants):
                routes = self.construct_solution()
                cost = route_cost(routes, self.D)
                all_solutions.append(routes)
                all_costs.append(cost)
                if cost < self.best_cost:
                    self.best_cost = cost
                    self.best_routes = routes
            self.update_pheromone(all_solutions, all_costs)
            if it % 10 == 0:
                print(f"Iteration {it}: Best = {self.best_cost:.3f}")
        return self.best_routes, self.best_cost

    def construct_solution(self):
        unvisited = list(range(1, self.n))  # exclude depot
        routes = []
        while unvisited:
            route = [0]
            capacity_used = 0
            while unvisited and capacity_used < self.vehicle_capacity:
                i = route[-1]
                probs = []
                candidates = []
                for j in unvisited:
                    if self.D[int(i)][int(j)] == 0:
                        continue
                    tau = self.pheromone[int(i)][int(j)] ** self.alpha
                    eta = (1 / self.D[int(i)][int(j)]) ** self.beta
                    probs.append(tau * eta)
                    candidates.append(j)
                if not candidates:
                    break
                probs = np.array(probs)
                probs /= probs.sum()
                next_customer = int(np.random.choice(candidates, p=probs))
                route.append(next_customer)
                unvisited.remove(next_customer)
                capacity_used += 1
            route.append(0)
            routes.append(route)
        return routes

    def update_pheromone(self, solutions, costs):
        self.pheromone *= (1 - self.rho)
        for routes, cost in zip(solutions, costs):
            for r in routes:
                for i in range(len(r) - 1):
                    self.pheromone[int(r[i])][int(r[i + 1])] += 1.0 / cost


# ----------------------------
# Plotting function with robust legend, depot-centered, clean axes
# ----------------------------
def plot_grid_routes_with_legend(G, node_mapping, depot_idx, customer_indices, routes, grid_size):
    pos = {n: (n % grid_size, n // grid_size) for n in G.nodes()}
    plt.figure(figsize=(7, 7))

    # Grid limits
    depot_x, depot_y = pos[node_mapping[depot_idx]]
    half_range = grid_size // 2 + 1
    x_min, x_max = depot_x - half_range, depot_x + half_range
    y_min, y_max = depot_y - half_range, depot_y + half_range
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)

    # Draw all grid edges
    for u, v in G.edges():
        x = [pos[u][0], pos[v][0]]
        y = [pos[u][1], pos[v][1]]
        plt.plot(x, y, color='lightgrey', linewidth=1)

    # Plot depot and customers
    plt.scatter(depot_x, depot_y, c='red', marker='*', s=400)
    cust_x = [pos[node_mapping[c]][0] for c in customer_indices]
    cust_y = [pos[node_mapping[c]][1] for c in customer_indices]
    plt.scatter(cust_x, cust_y, c='black', marker='o', s=100)

    # Plot routes
    cmap = plt.colormaps['tab20']
    route_colors = []
    for idx, r in enumerate(routes):
        color = cmap(idx / len(routes))
        route_colors.append(color)
        x = [pos[node_mapping[int(i)]][0] for i in r]
        y = [pos[node_mapping[int(i)]][1] for i in r]
        plt.plot(x, y, color=color, linewidth=2)
        plt.scatter(x, y, color=color, s=50)

    # Build manual legend
    legend_elements = [
        Line2D([0], [0], marker='*', color='w', label='Depot', markerfacecolor='red', markersize=15),
        Line2D([0], [0], marker='o', color='w', label='Customer', markerfacecolor='black', markersize=10)
    ]
    for idx, color in enumerate(route_colors):
        legend_elements.append(Line2D([0], [0], color=color, lw=2, label=f'Vehicle {idx + 1}'))
    plt.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(1.3, 1))

    # Remove axis numbers and borders
    plt.xticks([])
    plt.yticks([])
    for spine in plt.gca().spines.values():
        spine.set_visible(False)

    # Bold solid cross lines through depot, limited to grid edges
    plt.plot([x_min, x_max], [depot_y, depot_y], color='black', linewidth=2)
    plt.plot([depot_x, depot_x], [y_min, y_max], color='black', linewidth=2)

    plt.grid(True)
    plt.tight_layout()
    plt.show()


# ----------------------------
# Run Example with all iteration parameters
# ----------------------------
# Ant Colony Optimisation (ACO) Parameters:
#
# n_ants  → number of candidate solutions generated per iteration
#            - higher = more exploration but slower runtime
#            - lower = faster but may miss good solutions
#            - typical: 5 – 30
#
# alpha   → influence of pheromone trail (self.pheromone[i][j] ** alpha)
#            - higher = stronger memory of past solutions
#            - lower = more random exploration
#            - typical: 1 – 2
#
# beta    → influence of heuristic ( (1 / distance) ** beta )
#            - higher = greedier for short distances
#            - lower = less biased by distance
#            - typical: 2 – 5
#
# rho     → pheromone evaporation rate (self.pheromone *= (1 - rho))
#            - higher = forgets old info faster, encourages exploration
#            - lower = remembers old info longer, risks premature convergence
#            - typical: 0.3 – 0.7
#
# iterations → number of optimisation loops
#              - higher = more refinement, slower runtime
#              - typical: 50 – 500 depending on problem size

if __name__ == "__main__":
    grid_size = 100
    n_customers = 150
    vehicle_capacity = 20
    n_ants = 10 # Solutions per Iterations
    iterations = 50
    alpha = 1.0 # How much it prefers 'good' paths
    beta = 5.0 # How strongly it prioritises closer nodes
    rho = 0.1 # Decay of paths being 'good'

    # Generate grid network
    G, depot, customers = generate_grid_network(grid_size, n_customers)

    # Node mapping: relabeled ACO indices -> original grid node labels
    selected_nodes = [depot] + customers
    node_mapping = {idx: node for idx, node in enumerate(selected_nodes)}
    depot_idx = 0
    customer_indices = list(range(1, len(selected_nodes)))

    # Compute distance matrix
    D = compute_distance_matrix(G, selected_nodes)

    # Solve CVRP
    solver = ACO_CVRP(D, vehicle_capacity, n_ants=n_ants, alpha=alpha, beta=beta, rho=rho, iterations=iterations)
    best_routes, best_cost = solver.run()

    # Print results
    print("\nBest Routes:")
    for idx, r in enumerate(best_routes):
        print(f" Vehicle {idx + 1}: {r}")
    print("Total Cost:", best_cost)

    # Plot routes with depot-centered grid and clean layout
    plot_grid_routes_with_legend(G, node_mapping, depot_idx, customer_indices, best_routes, grid_size)
