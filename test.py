import numpy as np
import pandas as pd
from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans
from sklearn.svm import SVC
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from scipy.spatial.distance import cdist
from itertools import product
from scipy.stats import pearsonr

# --- Utility Functions ---

def normalize(df):
    return (df - df.min()) / (df.max() - df.min())

# --- Objective Functions ---
def asw_func(data, memberships):
    # Calculate the average silhouette width (ASW).
    return silhouette_score(data, memberships)

def cca_func(x, y):
    # Canonical correlation analysis-based objective.
    # Sample rows to align dimensions
    if x.shape[0] > y.shape[0]:
        x = x.iloc[np.random.choice(x.shape[0], y.shape[0], replace=False)]
    elif y.shape[0] > x.shape[0]:
        y = y[np.random.choice(y.shape[0], x.shape[0], replace=False)]

    # Compute covariance matrix
    cov_matrix = np.cov(x.T, y.T)
    diag_cov_x = np.diag(np.cov(x, rowvar=False))
    diag_cov_y = np.diag(np.cov(y, rowvar=False))
    return np.sum(cov_matrix) / (np.sqrt(np.sum(diag_cov_x) * np.sum(diag_cov_y)))


def vr_func(x, y):
    # Variance ratio objective.
    var_x = np.var(x, axis=0).sum()
    var_y = np.var(y, axis=0).sum()
    return min(var_x / var_y, var_y / var_x)

# --- Missing Value Generation ---
def generate_missing_values(df, pattern='simple', mechanism='overall', ratio=0.1):
    df_missing = df.copy()
    n_missing = int(ratio * df.size)

    if pattern == 'simple':
        for _ in range(n_missing):
            row_idx = np.random.randint(0, df_missing.shape[0])
            col_idx = np.random.randint(0, df_missing.shape[1])
            df_missing.iat[row_idx, col_idx] = np.nan

    elif pattern == 'medium':
        for _ in range(int(n_missing / 2)):
            row_idx = np.random.randint(0, df_missing.shape[0])
            col_indices = np.random.choice(df_missing.shape[1], 2, replace=False)
            for col_idx in col_indices:
                df_missing.iat[row_idx, col_idx] = np.nan

    elif pattern == 'complex':
        for _ in range(int(n_missing / 3)):
            row_indices = np.random.choice(df_missing.shape[0], 3, replace=False)
            col_idx = np.random.randint(0, df_missing.shape[1])
            for row_idx in row_indices:
                df_missing.iat[row_idx, col_idx] = np.nan

    return df_missing

# --- Genetic Algorithm Framework ---
def initialize_population(data, n_clusters=10, pop_size=20):
    population = []
    for _ in range(pop_size):
        centers = data.sample(n_clusters).to_numpy()
        m = np.random.uniform(1.5, 5.0)
        kernel = np.random.choice(['linear', 'rbf', 'sigmoid', 'poly'])
        c = np.random.uniform(0.1, 10.0)
        gamma = np.random.uniform(0.01, 1.0)
        population.append({'centers': centers, 'm': m, 'kernel': kernel, 'c': c, 'gamma': gamma})
    return population

def pareto_front(results):
    pareto = []
    for i, p in enumerate(results):
        dominated = False
        for j, q in enumerate(results):
            if i != j and all(q[k] >= p[k] for k in p) and any(q[k] > p[k] for k in p):
                dominated = True
                break
        if not dominated:
            pareto.append(p)
    return pareto

def crossover(parent1, parent2):
    # Perform crossover between two parents.
    child = parent1.copy()
    for key in ['centers', 'm', 'kernel', 'c', 'gamma']:
        if np.random.rand() > 0.5:
            child[key] = parent2[key]
    return child

def mutate(individual):
    #  Mutate an individual.
    mutation_rate = 0.1
    if np.random.rand() < mutation_rate:
        individual['m'] = np.random.uniform(1.5, 5.0)
    if np.random.rand() < mutation_rate:
        individual['kernel'] = np.random.choice(['linear', 'rbf', 'sigmoid', 'poly'])
    if np.random.rand() < mutation_rate:
        individual['c'] = np.random.uniform(0.1, 10.0)
    if np.random.rand() < mutation_rate:
        individual['gamma'] = np.random.uniform(0.01, 1.0)
    return individual

def track_errors_and_correlations(data, results):
    metrics = pd.DataFrame(results)
    
    def safe_pearsonr(x, y):
        if np.all(x == x[0]) or np.all(y == y[0]):  # Check if array is constant
            return np.nan
        return pearsonr(x, y)[0]
    
    correlations = {
        'asw_cca': safe_pearsonr(metrics['asw'], metrics['cca']),
        'asw_vr': safe_pearsonr(metrics['asw'], metrics['vr']),
        'cca_vr': safe_pearsonr(metrics['cca'], metrics['vr']),
    }
    return correlations


from sklearn.preprocessing import LabelEncoder

def fitness_function(data, population, train_x, train_y, test_x, test_y):
    # Ensure target variables are categorical
    le = LabelEncoder()
    train_y = le.fit_transform(train_y)
    test_y = le.transform(test_y)
    
    results = []
    for individual in population:
        centers = individual['centers']
        memberships = np.argmin(cdist(data, centers), axis=1)
        asw = asw_func(data, memberships)
        cca = cca_func(data, centers)
        vr = vr_func(data, centers)

        # SVM training and evaluation
        svm_model = SVC(kernel=individual['kernel'], C=individual['c'], gamma=individual['gamma'])
        svm_model.fit(train_x, train_y)
        train_acc = svm_model.score(train_x, train_y)
        test_acc = svm_model.score(test_x, test_y)

        results.append({'asw': asw, 'cca': cca, 'vr': vr, 'train_acc': train_acc, 'test_acc': test_acc})
    return results

def genetic_algorithm(data, n_clusters=10, pop_size=20, max_generations=50):
    population = initialize_population(data, n_clusters, pop_size)
    
    # Ensure target column exists
    target_col = data.columns[-1]
    train_x, test_x, train_y, test_y = train_test_split(data.iloc[:, :-1], data[target_col], test_size=0.5, random_state=1)

    for generation in range(max_generations):
        fitness = fitness_function(data, population, train_x, train_y, test_x, test_y)
        pareto = pareto_front(fitness)
        correlations = track_errors_and_correlations(data, fitness)
        print(f"Generation {generation}: Pareto Front Size = {len(pareto)}")
        print(f"Correlations: {correlations}")

        # Select individuals for next generation
        selected = np.random.choice(population, size=len(population)//2, replace=False)
        offspring = []

        # Crossover
        for parent1, parent2 in zip(selected[::2], selected[1::2]):
            child1 = crossover(parent1, parent2)
            child2 = crossover(parent2, parent1)
            offspring.extend([child1, child2])

        # Mutation
        offspring = [mutate(ind) for ind in offspring]

        # Update population
        population = list(selected) + offspring

def genetic_algorithm(data, n_clusters=10, pop_size=20, max_generations=50, patience=5):
    population = initialize_population(data, n_clusters, pop_size)
    train_x, test_x, train_y, test_y = train_test_split(data.iloc[:, :-1], data.iloc[:, -1], test_size=0.5, random_state=1)
    
    no_improvement = 0
    last_pareto_size = 0

    for generation in range(max_generations):
        fitness = fitness_function(data, population, train_x, train_y, test_x, test_y)
        pareto = pareto_front(fitness)
        correlations = track_errors_and_correlations(data, fitness)
        
        print(f"Generation {generation}: Pareto Front Size = {len(pareto)}")
        print(f"Correlations: {correlations}")
        
        # Check for early stopping
        if len(pareto) == last_pareto_size:
            no_improvement += 1
        else:
            no_improvement = 0
            last_pareto_size = len(pareto)
        
        if no_improvement >= patience:
            print(f"Early stopping at generation {generation} due to no improvement.")
            break
        
        # Select individuals for next generation
        selected = np.random.choice(population, size=len(population)//2, replace=False)
        offspring = []

        # Crossover
        for parent1, parent2 in zip(selected[::2], selected[1::2]):
            child1 = crossover(parent1, parent2)
            child2 = crossover(parent2, parent1)
            offspring.extend([child1, child2])

        # Mutation
        offspring = [mutate(ind) for ind in offspring]

        # Update population
        population = list(selected) + offspring


# --- Example Usage ---
from sklearn.datasets import load_iris
iris = load_iris()
iris_df = pd.DataFrame(iris.data, columns=iris.feature_names)
iris_df['target'] = iris.target
data = iris_df.apply(normalize)

# Run Genetic Algorithm
genetic_algorithm(data)



