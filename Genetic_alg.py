import numpy as np
import matplotlib.pyplot as plt
import random
from sklearn.cluster import KMeans

# --- Step 1: Generate random points ---
def generate_random_points(n_points=200, x_range=(-10,10), y_range=(-10,10), seed=42):
    np.random.seed(seed)
    x = np.random.uniform(x_range[0], x_range[1], n_points)
    y = np.random.uniform(y_range[0], y_range[1], n_points)
    return np.column_stack((x, y))

data = generate_random_points(n_points=200)

# --- Step 2: GA fitness (manual inertia) ---
def fitness_ga(centroids, data):
    """ Compute sum of squared distances to nearest centroid (like K-Means inertia) """
    distances = np.linalg.norm(data[:, None, :] - centroids[None, :, :], axis=2)
    labels = np.argmin(distances, axis=1)
    inertia = np.sum((data - centroids[labels])**2)
    return inertia

# --- Step 3: GA operators ---
def crossover(parent1, parent2):
    alpha = np.random.rand(*parent1.shape)
    return alpha*parent1 + (1-alpha)*parent2

def mutate(centroids, mutation_rate=0.5, mutation_strength=3.0):
    new_centroids = centroids.copy()
    for i in range(len(centroids)):
        if np.random.rand() < mutation_rate:
            new_centroids[i] += np.random.normal(0, mutation_strength, size=2)
    return new_centroids

# --- Step 4: Genetic Algorithm ---
def genetic_kmeans_ga_only(data, n_clusters=4, pop_size=20, generations=1000, plot_every=50):
    # Initialize population randomly
    x_min, x_max = data[:,0].min(), data[:,0].max()
    y_min, y_max = data[:,1].min(), data[:,1].max()
    population = [np.column_stack((np.random.uniform(x_min, x_max, n_clusters),
                                   np.random.uniform(y_min, y_max, n_clusters))) for _ in range(pop_size)]
    
    best_cost = float('inf')
    best_centroids = None
    
    for gen in range(1, generations+1):
        fitness_values = []
        for centroids in population:
            cost = fitness_ga(centroids, data)
            fitness_values.append(cost)
            if cost < best_cost:
                best_cost = cost
                best_centroids = centroids.copy()
        
        # Sort by fitness
        sorted_idx = np.argsort(fitness_values)
        population = [population[i] for i in sorted_idx]
        
        # Elitism: keep top 1
        new_population = [population[0]]
        
        # Generate offspring
        while len(new_population) < pop_size:
            parents = random.sample(population[:10], 2)
            child = crossover(parents[0], parents[1])
            child = mutate(child)
            new_population.append(child)
        
        population = new_population
        
        # Plot progress
        if gen % plot_every == 0 or gen == generations:
            plt.figure(figsize=(6,6))
            plt.scatter(data[:,0], data[:,1], s=30, alpha=0.6, label='Points')
            plt.scatter(best_centroids[:,0], best_centroids[:,1], color='red', marker='X', s=150, label='GA Centroids')
            plt.title(f'Generation {gen} | Best cost: {best_cost:.2f}')
            plt.legend()
            plt.grid(True)
            plt.show()
    
    return best_centroids, best_cost

# --- Step 5: Run GA ---
best_centroids, best_cost = genetic_kmeans_ga_only(data, n_clusters=4, pop_size=20, generations=500, plot_every=50)
print("Best GA cost:", best_cost)

# --- Step 6: Refine with K-Means using GA centroids as initialization ---
kmeans = KMeans(n_clusters=4, init=best_centroids, n_init=1, max_iter=300, random_state=42)
kmeans.fit(data)
print("K-Means cost after GA initialization:", kmeans.inertia_)

# Optional: plot K-Means final result
plt.figure(figsize=(6,6))
plt.scatter(data[:,0], data[:,1], s=30, alpha=0.6, label='Points')
plt.scatter(kmeans.cluster_centers_[:,0], kmeans.cluster_centers_[:,1], color='green', marker='X', s=150, label='K-Means Centroids')
plt.title(f'K-Means after GA | Cost: {kmeans.inertia_:.2f}')
plt.legend()
plt.grid(True)
plt.show()
