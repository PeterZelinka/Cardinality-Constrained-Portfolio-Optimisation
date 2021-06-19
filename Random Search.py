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
        with open(self.Asset_File, newline='') as datafile:
            for row in datafile:
                if len(row.split()) == 1: # if row is len of 1 it will be number of assets
                    for x in row.split(' '):
                        if x == '':
                            continue
                        self.number_of_stocks=(int(x))
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

            # Variable for storing correlations between assets
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
            self.covariance = zeros * self.deviations * self.deviations.reshape((self.deviations.shape[0], 1))
            self.sigma = self.covariance + self.covariance.T - np.diag(self.covariance.diagonal()) #Fills in the second part of the covariance matrix

            # Making sure constraints on minimum and maximum investments are met
            if K * epsilon > 1.0:
                print("Minimum investment is too large")
                raise ValueError
            if K * delta < 1.0:
                print("Maximum investment is too small")
                raise ValueError

            self.F = 1.0 - K * epsilon

class Solution:
    def __init__(self, N, K):
        """Initializes a solution"""
        # Initializing random attributes of a solution
        self.Q = np.random.permutation(N)[:K]
        self.s = np.random.rand(K)
        self.w = np.zeros(N)
        self.obj1 = np.nan
        self.obj2 = np.nan

def check_valid_solution(solution, dataset):
    """Checks whether a solution is valid given constraints"""
    w = solution.w
        # Checking whether correct number of solutions has been picked
    if np.sum(w >= dataset.epsilon) != K:
        raise ValueError("More than " + str(K) + " assets selected (" + str(np.sum(w > 0.0)) + ") in solution: " + str(w))
        # Checking whether number and size of proportions is correct
    elif np.any(solution.s > 1) or np.any(solution.s < 0) or len(solution.s) != K:
        raise ValueError("The values of solution.s are not valid: " + str(solution.s))
        # Checking whether proportions sum up to 1
    elif not np.isclose(w.sum(), 1):
        raise ValueError("Proportions don't sum up to 1 (" + str(w.sum()) + ") in solution: " + str(w))
        # Checking whether maximum investment amount has not been exceeded
    elif np.any(w > dataset.delta):
        raise ValueError("There's at least one proportion larger than delta: " + str(w))
        # Checking for duplicate assets in a solution
    elif len(np.unique(solution.Q)) != len(solution.Q):
        raise ValueError("Duplicated assets in the portfolio: " + str(w))


def evaluate(solution, dataset, l, Lambda, best_value_found, best_solutions):
    """ Creates a solution - calculates its covariance, expected return and f """
    improved = False
    # Initializing weights
    w = solution.w
    # Initialzed to make sure weights sum to 1 in the next step
    L = solution.s.sum()
    # Calculating weights from random numbers to sum to 1
    w_temp = dataset.epsilon + solution.s * dataset.F / L
    # Making sure the highest investment is met
    is_too_large = (w_temp > dataset.delta)
    # If an investment would be too large the loop would stop
    while is_too_large.sum() > 0:
        # Reversing logic
        is_not_too_large = np.logical_not(is_too_large)
        # Sum of weights
        L = solution.s[is_not_too_large].sum()
        # Temporary f value
        F_temp = 1.0 - (dataset.epsilon * is_not_too_large.sum() + dataset.delta * is_too_large.sum())
        # Calculating acutal weights to sum to 1 (adding minimal investmet)
        w_temp = dataset.epsilon + solution.s * F_temp / L
        # Implementing Max investment amount
        w_temp[is_too_large] = dataset.delta
        # Checking for invesments that are too large
        is_too_large = (w_temp > dataset.delta)

    w[:] = 0 
    w[solution.Q] = w_temp # Actual weights
    solution.s = w_temp - dataset.epsilon # Investment proportions

    # Checks whether a solution is valid given constraints
    check_valid_solution(solution, dataset)

    # Calculates covariance for a solution
    solution.obj1 = np.sum((w * w.reshape((w.shape[0], 1))) * dataset.sigma)

    # Calculates expected return for a solution
    solution.obj2 = np.sum(w * dataset.mu)

    # Calculate f
    f = Lambda[l] * solution.obj1 - (1 - Lambda[l]) * solution.obj2

    # Replace current solution with new solution if new is better
    if f < best_value_found[l]:
        improved = True
        best_value_found[l] = f
        best_solutions.append(solution)


    return improved, best_value_found[-1]


def RandomSearch(maxEvals, Lambda):
    """Calculates solutions based on the logic of the Random Search algorithm"""
    # An array of weights to weight the two objectives.
    if Lambda == 0.0:
        Lambda = np.array([Lambda])
        best_value_found = np.array([0.0])
    # Best value found for each weight.
    else:
        Lambda = np.array([Lambda])
        best_value_found = np.array(Lambda * [np.inf])

    # List of best solutions ever found.
    best_solutions = []

    nevals = 0 # Counter for the number of iterations
    l = 0
    # Generate and evaluate a new solution until maximum solution evaluations not reached
    while nevals < maxEvals:
        s = Solution(N,K)
        improved, f = evaluate(s, dataset, l, Lambda, best_value_found, best_solutions)
        nevals += 1

    # Collecting information on the best solution
    cov.append(best_solutions[-1].obj1)
    r.append(best_solutions[-1].obj2)
    assets.append(best_solutions[-1].Q)
    weights.append(best_solutions[-1].w)

    return s, f

# Iterating through data files
stock_lengths = [31,85,89,98,225]
asset_files = ['assets1.txt', 'assets2.txt', 'assets3.txt', 'assets4.txt', 'assets5.txt']
for n, file in zip(stock_lengths, asset_files):
    N= n # Total number of assets in data file
    Nvalues = [N]
    K = 10 # Number of assets to include in the portfolio
    E = 50 # Number of different lambda values

    # Initializing variables for collecting data on different lambdas
    Asset_File = file
    cov = []
    r = []
    weights = []
    assets = []
    fvalues = np.empty(E)
    lambvalues = np.empty(E)

    # Initializing the dataset
    dataset = DataSet(Asset_File, N, K, epsilon=0.01,)


    maxEvals = 1000 * N  # Maximum solution evaluations

    # Sets a random seed for solution repeatability
    seed = 12345
    np.random.seed(seed)

    # Iterating through different values of lambda
    for e in range(1, E + 1):
        Lambda = np.array([(e - 1) / (E - 1)]) # 50 lambda values equally spaced from 0 to 1
        s, f = RandomSearch(maxEvals, Lambda[0])
        print("N={0}, Lambda = {1}, f = {2}".format(N, Lambda[0], f))
        fvalues[e-1] = f
        lambvalues[e-1] = Lambda[0]

        # Tracking which lambda values is being currently calculated
        print(e)
    print("N={0}, mean = {1}, sd = {2}, min = {3}, max = {4}, lamb={5}".format(N, fvalues.mean(), fvalues.std(), fvalues.min(), fvalues.max(),lambvalues))

    # Returns
    r = np.array(r)
    # Weights
    weights = np.array(weights)
    # Covariances
    cov = np.array(cov)
    # Statistics about f values
    f_stats = [fvalues.min(), fvalues.max(), fvalues.mean(), fvalues.std()]
    # Statistics about returns
    r_stats = [r.min(), r.max(), r.mean(), r.std()]
    # Statisitcs of the covariances
    cov_stats = [cov.min(), cov.max(), cov.mean(), cov.std()]
    # The actual f values
    fs = np.array(fvalues)
    # Lambda values
    ls = np.array(lambvalues)

    # Statistical values about the F, Cov and R
    stats = pd.DataFrame(f_stats)
    stats[1] = r_stats
    stats[2] = cov_stats
    stats.columns = ['F value stats', 'Return stats', 'Covariance stats']

    # Results for the 50 lambda values
    results = pd.DataFrame(fs)
    results[1] = r
    results[2] = cov
    results.columns = ['F values', 'Returns', 'Covariances']

    # Weights of the best portfolios for the 50 lambda values
    weights = pd.DataFrame(weights, columns=list(range(1, Nvalues[0]+1)))

    # Indexes of  Assets used in each of the best portfolios for the 50 lambda values
    col_names = ['asset_{}'.format(i) for i in range(1, 11)]
    assets = pd.DataFrame(assets, columns=col_names)

    #Lambdas used
    Lambdas = pd.DataFrame(ls, columns=['Lambda'])

    # Creating CSV files for further analysis
    df_results = pd.concat([Lambdas, results, assets, weights], axis=1)
    stats.to_csv('stats_RS_'+file[:-4]+'.csv', index=False)
    df_results.to_csv('results_RS_'+file[:-4]+'.csv', index=False)
