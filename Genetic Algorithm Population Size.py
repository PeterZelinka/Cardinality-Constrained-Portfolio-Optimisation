import numpy as np
import pandas as pd

class DataSet:
    def __init__(self, Asset_File, N, K, epsilon=0.01, delta=1.0):
        """Loads a dataset and divides its contents into variables """
        self.Asset_File = Asset_File
        self.N = N # Total number of assets in a dataset
        self.K = K # Total number of assets in a solution
        self.epsilon = epsilon # Min investment
        self.delta = delta # Max investment
        self.number_of_stocks = 0
        self.returns_deviations = []
        self.correlations = []
        self.covariance = np.nan
        temp_li_1 = []
        temp_li_2 = []

        # Splitting rows based on what they contain
        with open('Datasets/{}'.format(Asset_File), newline='') as datafile:
            for row in datafile:
                if len(row.split()) == 1: # if row is len of 1 it will be number of assets
                    for x in row.split(' '):
                        if x == '':
                            continue
                        self.number_of_stocks = (int(x))
                elif len(row.split()) == 2: # if row is len of 2 it will be the assets return and standard deviation
                    for x in row.split(' '):
                        if x == '':
                            continue
                        self.returns_deviations.append(float(x))
                elif len(row.split()) == 3: # if row is len of 3 it will be the correlation between assets
                    for x in row.split(' '):
                        if x == '':
                            continue
                        self.correlations.append(float(x))

                # Variable for storing standard deviations of returns
            for i, z in zip(self.returns_deviations[0::2], self.returns_deviations[1::2]):
                temp_li_1.append([i, z])
            self.returns_deviations = temp_li_1

            # Variable for storing correlations between asset
            zeros = np.zeros((int(self.number_of_stocks), int(self.number_of_stocks)))
            for x, y, z in zip(self.correlations[0::3], self.correlations[1::3], self.correlations[2::3]):
                temp_li_2.append([x, y, z])
                zeros[int(x)-1][int(y)-1] = z
            self.correlations = temp_li_2

            # Creates a matrix of  returns and deviations
            self.returns_deviations=np.array(self.returns_deviations)

            # Splitting the data into variables needed for calculation
            self.deviations = self.returns_deviations[:, 1]
            self.mu = self.returns_deviations[:, 0]
            self.covariance = zeros*self.deviations*self.deviations.reshape((self.deviations.shape[0],1))
            self.sigma = self.covariance+ self.covariance.T - np.diag(self.covariance.diagonal()) # Fills in the second part of the covariance matrix

            # Making sure constraints on minimum and maximum investments are met
            if K * epsilon > 1.0:
                print("Minimum investment is too large")
                raise ValueError
            if K * delta < 1.0:
                print("Maximum investment is too small")
                raise ValueError

            self.F = 1.0 - K * epsilon

class Population():
    def __init__(self, size):
        """Population of solutions"""
        self.Population_size = size #Changed through iterations
        self.population_weights = [] # Weights of the population
        self.population_assets = [] # A list containing assets of individuals in the population
        self.fitness = []
        self.population_proportions = []
        self.best_fitness = 0 # Best f
        self.best_proportions = 0 # Best proportions by which each asset is in an individual
        self.best_weights = 0
        self.best_assets = 0
        self.best_covariance = 0
        self.best_return = 0
        self.Obj1 = []
        self.Obj2 = []

    def check_valid_solution(self, weights, proportions, assets, data):
        """Checks whether a solution is valid given constraints"""
            # Checking whether correct number of solutions has been picked
        if np.sum(weights >= data.epsilon) != K:
            raise ValueError("More than " + str(K) + " assets selected (" + str(np.sum(weights > 0.0)) + ") in solution: " + str(weights))
            # Checking whether number and size of proportions is correct
        if np.any(proportions > 1) or np.any(proportions < 0) or len(proportions) != K:
            raise ValueError("The values of proportions are not valid: " + str(proportions))
            # Checking whether proportions sum up to 1
        elif not np.isclose(weights.sum(), 1):
            raise ValueError("Proportions don't sum up to 1 (" + str(weights.sum()) + ") in solution: " + str(weights))
            # Checking whether maximum investment amount has not been exceeded
        elif np.any(weights > data.delta):
            raise ValueError("There's at least one proportion larger than delta: " + str(weights))
            # Checking for duplicate assets in a solution
        elif len(np.unique(assets)) != len(assets):
            raise ValueError("Duplicated assets in the portfolio: " + str(assets))


    def create_Population(Population, Lambda, l, data):
        """Initializes random population of solutions"""
        for i in range(Population.Population_size):
            #Initializing individuals in the popuplation
            R = np.random.permutation(N)[:K]
            # Random weights of the 10 assets
            s = np.random.rand(K)
            # Initializes weights
            w = np.zeros(N)
            # Initialized to make sure that the weights sum to 1
            L = s.sum()
            # Making sure that the random weights sum up to 1 given min investment
            w_temp = data.epsilon + s * data.F / L
            # Making sure the highest investment is met
            is_too_large = (w_temp > data.delta)
            # If an investment would be too large the loop would stop
            while is_too_large.sum() > 0:
                # Reversing logic
                is_not_too_large = np.logical_not(is_too_large)
                # Sum of weights
                L = s[is_not_too_large].sum()
                # Calculates temporary F value
                F_temp = 1.0 - (data.epsilon * is_not_too_large.sum() + data.delta * is_too_large.sum())
                # Adding minimal investment and making sure the actual weights sum to 1 given min investment
                w_temp = data.epsilon + s * F_temp / L
                # Implementing Max investment amount
                w_temp[is_too_large] = data.delta
                # Checking for invesments that are too large
                is_too_large = (w_temp > data.delta)


            w[:] = 0
            w[R] = w_temp # Actual weights
            s = w_temp - data.epsilon # Investment proportions

            # Checking whether our solution is valid
            Population.check_valid_solution(w, s, R, data)

            # Adding valid solution to our population
            Population.population_proportions.append(s)
            Population.population_weights.append(w)
            Population.population_assets.append(R.tolist())

            # Calculating fitness of the population
        for i in Population.population_weights:
            obj1 = np.sum((i * i.reshape((i.shape[0], 1))) * data.sigma)
            obj2 = np.sum(i * data.mu)
            f = Lambda[l] * obj1 - (1 - Lambda[l]) * obj2
            Population.fitness.append(f)
            Population.Obj1.append(obj1) # Covariance
            Population.Obj2.append(obj2) # Expected return


    def Genetic_Algorithm(Population, Lambda, l, data):
            """Applies the logic of genetic algorithm to the whole population"""
            if Population.Population_size == 1: # If population has only 4 individual we cannot select 4 at random
                picked_individuals = np.random.permutation(Population.Population_size)[:4].tolist()*4
            else:
                # Selecting 4 different individuals from the population
                picked_individuals = np.random.permutation(Population.Population_size)[:4].tolist() # Selecting 4 non-reccuring individuals from the population

            # Initializing child of the selected individuals
            child_assets = []
            child_proportions = []
            child_weights = np.zeros(N)
            l = 0


            #Pool_1
            pair_1_assets = [Population.population_assets[picked_individuals[0]], Population.population_assets[picked_individuals[1]]]
            pair_1_fitness = [Population.fitness[picked_individuals[0]], Population.fitness[picked_individuals[1]]]
            pair_1_proportions = [Population.population_proportions[picked_individuals[0]], Population.population_proportions[picked_individuals[1]]]

            # Pool_2
            pair_2_assets = [Population.population_assets[picked_individuals[2]], Population.population_assets[picked_individuals[3]]]
            pair_2_fitness = [Population.fitness[picked_individuals[2]], Population.fitness[picked_individuals[3]]]
            pair_2_proportions = [Population.population_proportions[picked_individuals[2]], Population.population_proportions[picked_individuals[3]]]

            # Selecting parents for the uniform crossover
            parent_1_assets = pair_1_assets[pair_1_fitness.index(min(pair_1_fitness))]
            parent_1_proportions = pair_1_proportions[pair_1_fitness.index(min(pair_1_fitness))]

            parent_2_assets = pair_2_assets[pair_2_fitness.index(min(pair_2_fitness))]
            parent_2_proportions = pair_2_proportions[pair_2_fitness.index(min(pair_2_fitness))]

            # Looking for same assets in parents and inputting them into child
            common_assets = []
            for i in parent_1_assets:
                if i in parent_2_assets:
                    common_assets.append(i)
            child_assets += common_assets

            # Finding out what are the indexes of those assets in parents
            indexes_1 = []
            indexes_2 = []
            for i in common_assets:
                indexes_1.append(parent_1_assets.index(i))
                indexes_2.append(parent_2_assets.index(i))

            # Adding the proportions of same assets to child with 50% chance
            for m, h in zip(indexes_1, indexes_2):
                rand_1 = np.random.rand()
                if rand_1 > 0.5:
                    child_proportions.append(parent_1_proportions[m])
                else:
                    child_proportions.append(parent_2_proportions[h])

            # Creating new lists with assets that each parent don't have in common
            temp_parent_1_assets = []
            temp_parent_2_assets = []
            for m, h in zip(parent_1_assets, parent_2_assets):
                temp_parent_1_assets.append(m)
                temp_parent_2_assets.append(h)

            for i in common_assets:
                if i in temp_parent_1_assets:
                    temp_parent_1_assets.remove(i)

            for i in common_assets:
                if i in temp_parent_2_assets:
                    temp_parent_2_assets.remove(i)

            # Adding other assets and their corresponding proportions to the child
            for m, h in zip(temp_parent_1_assets, temp_parent_2_assets):
                rand_2 = np.random.rand()
                if rand_2 > 0.5:
                    child_assets.append(m)
                    child_proportions.append(parent_1_proportions[parent_1_assets.index(m)])
                else:
                    child_assets.append(h)
                    child_proportions.append(parent_2_proportions[parent_2_assets.index(h)])

            # Creating A*
            # A* is a set of assets that are in the parents, but are not in the child (together with their associated values)
            parent_minus_child_assets = []
            parent_minus_child_proportions = []
            for m, h in zip(parent_1_assets, parent_2_assets):
                if m not in child_assets:
                    parent_minus_child_assets.append(m)
                    parent_minus_child_proportions.append(parent_1_proportions[parent_1_assets.index(m)])
                if h not in child_assets:
                    parent_minus_child_assets.append(h)
                    parent_minus_child_proportions.append(parent_2_proportions[parent_2_assets.index(h)])

            # Assets that can be potentially added to the child in case parent_minus_child assets (A*) are empty
            other_assets = np.random.permutation(N).tolist()
            for i in other_assets:
                if i in child_assets:
                    other_assets.remove(i)

            # Mutation
            mutated_asset = np.random.choice(child_proportions)
            rand_3 = np.random.rand()
            if rand_3 > 0.5:
                child_proportions[child_proportions.index(mutated_asset)] = (0.9 * (data.epsilon + mutated_asset) - data.epsilon)  # m=1
            else:
                child_proportions[child_proportions.index(mutated_asset)] = (1.1 * (data.epsilon + mutated_asset) - data.epsilon)  # m=2
            mutated_child_proportions = child_proportions

            # Making sure the child does not have two identical assets
            for i in child_assets:
                if child_assets.count(i) > 1:
                    mutated_child_proportions.remove(mutated_child_proportions[child_assets.index(i)])
                    child_assets.remove(i)

            # Making sure all child proportion are between 0 and 1 (if not they get excluded)
            for i in mutated_child_proportions:
                if i < 0 or i > 1:
                    child_assets.remove(child_assets[mutated_child_proportions.index(i)])
                    mutated_child_proportions.remove(i)

            # Ensure that child has exactly 10 assets and proportions
            while len(child_assets) > data.K and len(mutated_child_proportions) > data.K:
                child_assets.remove(child_assets.index(min(mutated_child_proportions)))
                mutated_child_proportions.remove(min(mutated_child_proportions))

                # Add assets from A* to child
            while len(child_assets) < data.K and len(mutated_child_proportions) < data.K:
                if len(parent_minus_child_assets) != 0:
                    rand_4 = np.random.choice(parent_minus_child_assets)
                    child_assets.append(rand_4)
                    mutated_child_proportions.append(parent_minus_child_proportions[parent_minus_child_assets.index(rand_4)])
                    parent_minus_child_proportions.remove(parent_minus_child_proportions[parent_minus_child_assets.index(rand_4)])
                    parent_minus_child_assets.remove(rand_4)
                    for i in mutated_child_proportions:
                        if i < 0 or i > 1:
                            child_assets.remove(child_assets[mutated_child_proportions.index(i)])
                            mutated_child_proportions.remove(i)
                    for i in child_assets:
                        if child_assets.count(i) > 1:
                            mutated_child_proportions.remove(mutated_child_proportions[child_assets.index(i)])
                            child_assets.remove(i)
                else: #In case A* is empty
                    rand_5=np.random.choice(other_assets)
                    child_assets.append(rand_5)
                    other_assets.remove(rand_5)
                    mutated_child_proportions.append(0)
                    for i in mutated_child_proportions:
                        if i < 0 or i > 1:
                            child_assets.remove(child_assets[mutated_child_proportions.index(i)])
                            mutated_child_proportions.remove(i)
                    for i in child_assets:
                        if child_assets.count(i) > 1:
                            mutated_child_proportions.remove(mutated_child_proportions[child_assets.index(i)])
                            child_assets.remove(i)

            # Given large amount of iterations and randomness all child proportions could be 0 hence set 1 at random to 0.01
            # Does not influence the overall result as it ist immediately replaced by a stronger individual
            if sum(mutated_child_proportions) == 0:
                mutated_child_proportions[mutated_child_proportions.index(np.random.choice(mutated_child_proportions))]=0.01

            # Evaluating child
            mutated_child_proportions = np.array(mutated_child_proportions)
            L = mutated_child_proportions.sum()
            w_temp = data.epsilon + mutated_child_proportions * data.F / L
            is_too_large = (w_temp > data.delta)
            while is_too_large.sum() > 0:
                is_not_too_large = np.logical_not(is_too_large)
                L = mutated_child_proportions[is_not_too_large].sum()
                F_temp = 1.0 - (data.epsilon * is_not_too_large.sum() + data.delta * is_too_large.sum())
                w_temp = data.epsilon + mutated_child_proportions * F_temp / L
                w_temp[is_too_large] = data.delta
                is_too_large = (w_temp > data.delta)

            # Assigning weights to child
            child_weights[:] = 0
            child_weights[child_assets] = w_temp
            mutated_child_proportions = w_temp - data.epsilon

            # Calculating child fitness
            obj1 = np.sum((child_weights * child_weights.reshape((child_weights.shape[0], 1))) * data.sigma)
            obj2 = np.sum(child_weights * data.mu)
            child_fitness = Lambda[l] * obj1 - (1 - Lambda[l]) * obj2

            # Checking whether child is valid
            Population.check_valid_solution(child_weights, mutated_child_proportions, child_assets, data)

            # Substituting child into the population and removing the weakest member
            index_worst_member = np.argmax(Population.fitness)
            Population.fitness[index_worst_member] = child_fitness
            Population.population_proportions[index_worst_member] = mutated_child_proportions
            Population.population_weights[index_worst_member] = child_weights
            Population.population_assets[index_worst_member] = child_assets
            Population.Obj1[index_worst_member] = obj1
            Population.Obj2[index_worst_member] = obj2

            # Finding the best member of the population
            index_best_member = np.argmin(Population.fitness)
            Population.best_fitness = Population.fitness[index_best_member]
            Population.best_proportions = Population.population_proportions[index_best_member]
            Population.best_weights = Population.population_weights[index_best_member]
            Population.best_assets = Population.population_assets[index_best_member]
            Population.best_covariance = Population.Obj1[index_best_member]
            Population.best_return = Population.Obj2[index_best_member]

            return Population.best_fitness, Population.best_proportions, Population.best_assets, Population.best_weights, Population.best_covariance, Population.best_return

# Iterating through data files
stock_lengths = [31,85,89,98,225]
asset_files = ['assets1.txt', 'assets2.txt', 'assets3.txt', 'assets4.txt', 'assets5.txt']
for n, file in zip(stock_lengths, asset_files):
    # Population sizes used in iteration
    population_sizes = [1, 10, 50, 100, 200, 500]
    # Calculating the values for different population sizes
    for pop_size in population_sizes:
        l = 0
        N = n # Total number of assets in data file
        Nvalues = [N]
        Asset_File = file
        K = 10 # Number of assets to include in the portfolio

        Lambda = np.array([0.5]) # Fixed lambda for this calculation

        # Initializing variables for collecting data on different lambdas
        Results_fitness = []
        Results_weights = []
        Results_assets = []
        Results_proportions = []
        Results_Covariances = []
        Results_Returns = []

        # Initializing the dataset
        dataset = DataSet(Asset_File, N, K)

        nevals = 0 # Counter for the number of iterations
        maxEvals = 1000*N # Solution evaluations per run

        # 30 runs for 30 different random seeds
        Runs = 30
        while nevals < Runs:
            # Changing the random seed 30 times
            seed = nevals + 12345
            np.random.seed(seed)

            # Initializing population
            population = Population(pop_size)
            population.create_Population(Lambda, l, dataset)

            for i in range(maxEvals):
                population.Genetic_Algorithm(Lambda, l, dataset)

            # Collecting results
            Results_fitness.append(population.best_fitness)
            Results_weights.append(population.best_weights)
            Results_assets.append(population.best_assets)
            Results_Covariances.append(population.best_covariance)
            Results_Returns.append(population.best_return)

            print(nevals) # Tracking current iteration
            nevals += 1

        Results_fitness = np.array(Results_fitness)
        Results_Returns = np.array(Results_Returns)
        Results_Covariances = np.array(Results_Covariances)

        # Statistics about f values
        f_stats = [Results_fitness.min(), Results_fitness.max(), Results_fitness.mean(), Results_fitness.std()]
        # Statistics about returns
        r_stats = [Results_Returns.min(), Results_Returns.max(), Results_Returns.mean(), Results_Returns.std()]
        # Statistics of the covariances
        cov_stats = [Results_Covariances.min(), Results_Covariances.max(), Results_Covariances.mean(), Results_Covariances.std()]

        # Statistical values about the F, Cov and R
        stats = pd.DataFrame(f_stats)
        stats[1] = r_stats
        stats[2] = cov_stats
        stats.columns = ['F value stats', 'Return stats', 'Covariance stats']

        # Results for the 30 seeds
        results = pd.DataFrame(Results_fitness)
        results[1] = Results_Returns
        results[2] = Results_Covariances
        results.columns = ['F values', 'Returns', 'Covariances']

        # Weights of the best portfolios for the 30 seeds
        weights = pd.DataFrame(Results_weights, columns=list(range(1, N+1)))

        # Indexes of  Assets used in each of the best portfolios for the 30 seeds
        col_names = ['asset_{}'.format(i) for i in range(1, 11)]
        assets = pd.DataFrame(Results_assets, columns=col_names)

        # Creating CSV files for further analysis
        df_results = pd.concat([results, assets, weights], axis=1)
        stats.to_csv('stats_GA_p={}_{}.csv'.format(str(pop_size),file[:-4]), index=False)
        df_results.to_csv('results_GA_p={}_{}.csv'.format(str(pop_size),file[:-4]), index=False)
