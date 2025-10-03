import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

# =====================================================
# Create Grid Graph (road network)
# =====================================================
def create_grid_graph(n):
    G = nx.grid_2d_graph(n, n)  # square grid
    mapping = {node: i for i, node in enumerate(G.nodes())}
    G = nx.relabel_nodes(G, mapping)
    pos = {mapping[node]: node for node in nx.grid_2d_graph(n, n).nodes()}
    return G, pos

# =====================================================
# Compute Distance Matrix
# =====================================================
def compute_distance_matrix(G):
    n = len(G.nodes())
    D = np.zeros((n, n))
    for i in G.nodes():
        lengths = nx.single_source_shortest_path_length(G, i)
        for j in G.nodes():
            if j in lengths:
                D[i, j] = lengths[j]
            else:
                D[i, j] = np.inf
    return D

# =====================================================
# Ant Colony Optimization for VRP with multiple depots
# =====================================================
class AntColonyVRP:
    def __init__(self, D, depot_indices, customer_indices, n_vehicles,
                 n_ants=10, alpha=1, beta=3, rho=0.5, iterations=100):
        """
        Parameters:
        - D (2D array): Distance matrix
        - depot_indices (list): List of depot node indices
        - customer_indices (list): List of customer node indices
        - n_vehicles (int): Number of vehicles available
        - n_ants (int): Number of ants (solutions explored per iteration)
        - alpha (float): Influence of pheromone trails
        - beta (float): Influence of distance (heuristic desirability)
        - rho (float): Pheromone evaporation rate
        - iterations (int): Number of iterations to run
        """
        self.D = D
        self.depot_indices = depot_indices
        self.customer_indices = customer_indices
        self.n_vehicles = n_vehicles
        self.n_ants = n_ants
        self.alpha = alpha
        self.beta = beta
        self.rho = rho
        self.iterations = iterations
        self.n_nodes = len(D)
        self.pheromone = np.ones((self.n_nodes, self.n_nodes))

    def run(self):
        best_routes = None
        best_cost = np.inf

        for it in range(self.iterations):
            all_routes = []
            all_costs = []

            for ant in range(self.n_ants):
                routes, cost = self.construct_solution()
                all_routes.append(routes)
                all_costs.append(cost)

            # Update pheromone
            self.pheromone *= (1 - self.rho)
            for routes, cost in zip(all_routes, all_costs):
                for r in routes:
                    for i in range(len(r) - 1):
                        self.pheromone[r[i], r[i + 1]] += 1.0 / cost

            # Track best solution
            min_cost = min(all_costs)
            if min_cost < best_cost:
                best_cost = min_cost
                best_routes = all_routes[np.argmin(all_costs)]

            if it % 10 == 0:
                print(f"Iteration {it}: Best = {best_cost:.3f}")

        return best_routes, best_cost

    def construct_solution(self):
        routes = []
        customers = self.customer_indices.copy()
        np.random.shuffle(customers)

        # Split customers among vehicles
        chunk_size = max(1, len(customers) // self.n_vehicles)
        vehicle_chunks = [customers[i:i + chunk_size] for i in range(0, len(customers), chunk_size)]

        for v, chunk in enumerate(vehicle_chunks):
            if not chunk:
                continue

            # Randomly assign a depot for this vehicle
            depot = np.random.choice(self.depot_indices)
            route = [depot]

            while chunk:
                current = route[-1]
                probs = []
                for c in chunk:
                    tau = self.pheromone[current, c] ** self.alpha
                    eta = (1.0 / (self.D[current, c] + 1e-6)) ** self.beta
                    probs.append(tau * eta)
                probs = np.array(probs)
                probs /= probs.sum()
                next_customer = np.random.choice(chunk, p=probs)
                route.append(next_customer)
                chunk.remove(next_customer)

            route.append(depot)
            routes.append(route)

        cost = sum(self.route_cost(r) for r in routes)
        return routes, cost

    def route_cost(self, route):
        return sum(self.D[route[i], route[i + 1]] for i in range(len(route) - 1))


# =====================================================
# Plotting Function
# =====================================================
def plot_grid_routes(G, pos, depot_indices, customer_indices, routes):
    plt.figure(figsize=(8, 8))

    # Draw base grid
    nx.draw(G, pos, node_size=50, node_color='lightgrey', with_labels=False, edge_color="lightgrey")

    # Draw depots
    nx.draw_networkx_nodes(G, pos, nodelist=depot_indices, node_size=500,
                           node_color='red', node_shape='*', label='Depots')

    # Draw customers
    nx.draw_networkx_nodes(G, pos, nodelist=customer_indices, node_size=200,
                           node_color='blue', label='Customers')

    # Draw routes with unique colors
    colors = plt.colormaps['tab20'].colors
    for i, r in enumerate(routes):
        x = [pos[int(n)][0] for n in r]
        y = [pos[int(n)][1] for n in r]
        plt.plot(x, y, color=colors[i % len(colors)], linewidth=2, label=f'Vehicle {i + 1}')

    # Legend outside the plot, deduplicated
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys(), loc='upper left', bbox_to_anchor=(1, 1))

    plt.axis('off')
    plt.subplots_adjust(right=0.8)  # leave space for legend
    plt.show()


# =====================================================
# Example Run
# =====================================================
if __name__ == "__main__":
    grid_size = 75
    n_customers = 75
    n_vehicles = 10
    n_depots = 3
    n_iterations = 20  # Number of iterations for ACO

    # Create road network
    G, pos = create_grid_graph(grid_size)
    D = compute_distance_matrix(G)

    # Randomly choose depots
    all_nodes = list(G.nodes())
    np.random.shuffle(all_nodes)
    depot_indices = all_nodes[:n_depots]

    # Randomly choose customers
    remaining_nodes = list(set(all_nodes) - set(depot_indices))
    np.random.shuffle(remaining_nodes)
    customer_indices = remaining_nodes[:n_customers]

    # Run ACO
    solver = AntColonyVRP(D, depot_indices, customer_indices, n_vehicles,
                           n_ants=10, alpha=1, beta=3, rho=0.5, iterations=n_iterations)
    best_routes, best_cost = solver.run()

    print("\nBest Routes:")
    for i, r in enumerate(best_routes):
        print(f" Vehicle {i + 1}: {r}")
    print(f"Total Cost: {best_cost:.1f}")

    # Plot
    plot_grid_routes(G, pos, depot_indices, customer_indices, best_routes)
