import sys
from itertools import product
from abc import ABC, abstractmethod, abstractproperty
import numpy as np
from functools import lru_cache
import warnings
from .problem import DiscreteProblem
from sklearn import preprocessing, metrics
from typing import Tuple, List
import sys
import heapq
import time
import scipy.stats as scs


class PriorityQueue(list):
    """Regardless of push order, minimum out."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        heapq.heapify(self)

    def push(self, obj: Tuple[float, Tuple[int, int]]):
        """Push object into PriorityQueue while retaining order."""
        heapq.heappush(self, obj)

    def pop(self):
        """Pop object from PriorityQueue while retaining order."""
        obj = heapq.heappop(self)
        return obj

    def remove(self, node):
        """Remove object from PriorityQueue while retaining order."""
        for i, obj in enumerate(self):
            if obj[1] == node:
                super().pop(i)
                heapq.heapify(self)
                break


class Graph:
    """Adjacency graph for feature information gain.

    Arguments:
    --------------
    vertices: int, numbe of vertices

    Notes:
    --------------
    Adapted from Divyanshu Mehta's implementation:
    https://www.geeksforgeeks.org/prims-minimum-spanning-tree-mst-greedy-algo-5/
    """

    def __init__(self, vertices, graph=None):
        self.V = vertices
        if graph is None:
            graph = [[0 for column in range(vertices)]
                          for row in range(vertices)]
        self.graph = graph
        self.parent = None

    def read_dependencies(self, parent):
        return [((parent[i],i), self.graph[i][ parent[i] ]) for i in range(1, self.V)]

    def minKey(self, key, mstSet):
        min = sys.maxsize

        for v in range(self.V):
            if key[v] < min and not mstSet[v]:
                min = key[v]
                min_index = v

        try:
            return min_index
        except Exception as e:
            import pdb; pdb.set_trace()
            raise e
        else:
            pass

    def calculate_deps(self):
        """Keys for picking the minimum weight edge in cut."""
        key = [sys.maxsize] * self.V
        parent = [None] * self.V  # Array to store constructed MST
        # Make key 0 so that this vertex is picked as first vertex
        key[0] = 0
        mstSet = [False] * self.V

        parent[0] = -1  # First node is always the root of

        for cout in range(self.V):

            u = self.minKey(key, mstSet)

            mstSet[u] = True

            for v in range(self.V):
                if self.graph[u][v] > 0 and mstSet[v] == False and key[v] > self.graph[u][v]:
                    key[v] = self.graph[u][v]
                    parent[v] = u

        self.parent = parent
        return self.read_dependencies(parent)


def get_random_bits(k):
    bits = np.random.choice(['0', '1'], size=k)
    bits = ''.join(bits)
    return bits


def get_neighbors(s: str):
    s = np.array([int(i) for i in list(s)]).reshape(1, -1)
    s = np.repeat(s, s.shape[1], axis=0)
    s[range(s.shape[1]), range(s.shape[1])] = 1 - \
        s[range(s.shape[1]), range(s.shape[1])]
    return [''.join([str(b) for b in row]) for row in s]


def get_random_neighbor(s: str):
    s = [int(i) for i in s]
    i = np.random.choice(range(len(s)))
    s[i] = 1 - s[i]
    s = [str(i) for i in s]
    s = ''.join(s)
    return s


def generate_solution(nbits):
    for candidate in product(*[[0,1]] * nbits):
        candidate = ''.join([str(i) for i in candidate])
        yield candidate

class Solver:
    """Base class for solvers."""

    def __init__(self, nbits=8, verbose=0):
        self.nbits = nbits
        self.maks = -float('inf')
        self.best = None
        self.verbose = verbose
        self.solution_time = None
        self.solution_arr = []
        self.initial_solution = None

    def vprint(self, *args, **kwargs):
        if self.verbose:
            print(*args, **kwargs)

    def _iter_callback(self):
        self.solution_arr.append(self.maks)

    @abstractmethod
    def _a_search(self, problem: DiscreteProblem) -> str:
        pass

    def ara(self, problem: DiscreteProblem) -> str:
        then = time.time()
        try:
            res = self._a_search(problem)
        except KeyboardInterrupt:
            res = self.best
        now = time.time()
        self.solution_time = round(now - then, 4)
        return res

    @lru_cache(maxsize=5000)
    def query(self, problem, x):
        return problem.evaluate(x)

    def fit(self, problem):
        self.nbits = problem.required_bits
        if hasattr(problem, "provide_initial_solution"):
            self.initial_solution = problem.provide_initial_solution()
        return self


class BruteSolver(Solver):
    """Perform a brute search over feature space."""

    def _a_search(self, problem):
        if self.nbits > 25:
            warnings.warn(f'Do you really want a brute search over {2**self.nbits} candidates?')
        maksskor = self.maks
        for i in generate_solution(self.nbits):
            skor = self.query(problem, i, )
            if skor > maksskor:
                maksskor = skor
                sifre = i
                self.maks = maksskor
                self.best = sifre
                self._iter_callback()
            self.vprint(maksskor)
        return sifre


class RandomSearcher(Solver):
    """Perform a random search in feature space.

    Arguments:
    ------------
    nbits: int, default 8. number of binary features
    maksiter: int, default 500. number of iterations
    random_neighbors: unused.
    """

    def __init__(self, nbits=8, maksiter=500, verbose=0, random_neighbors=None):
        super().__init__(nbits, verbose=verbose)
        if random_neighbors is None:
            random_neighbors = 0
        self.random_neighbors = random_neighbors
        self.maksiter = maksiter

    def _a_search(self, problem):
        maksskor = self.maks
        for i in range(self.maksiter):
            si = get_random_bits(self.nbits)
            skor = self.query(problem, si, )
            if skor > maksskor:
                maksskor = skor
                self.maks = maksskor
                self.vprint('Max score:', maksskor)
                sifre = si
            self._iter_callback()
        self.maks = maksskor
        self.best = sifre
        return sifre


class HillClimber(RandomSearcher):
    """HillClimber algorithm.

    Move in the direction of neighbor maximizing fitness function.

    Arguments:
    ------------------
    nbits: int, default 8. number of binary features
    maksiter: int, default 500. number of iterations
    random_neighbors: [int, None], default None. Number of random neighbors to evaluate.
                      if None or 0, all neighbors will be evaluated.
    """

    def _a_search(self, problem):
        point = get_random_bits(self.nbits) if self.initial_solution is not None else self.initial_solution
        self.best = point
        iters = 0
        while iters < self.maksiter:
            this = self.query(problem, point, )
            if this > self.maks:
                self.maks = this
                self.best = point
            if self.random_neighbors and self.nbits > self.random_neighbors:
                neighbors = [get_random_neighbor(point) for i in range(self.random_neighbors)]
            else:
                neighbors = get_neighbors(point)
            fits = np.array(
                list(map(lambda x: self.query(problem=problem, x=x), neighbors, )))
            if (this > fits).all():
                self.vprint('Search terminated.')
                self.vprint('Best score:', self.maks)
                return self.best
            point = neighbors[fits.argmax()]
            self.vprint(iters, 'iterations completed.')
            self.vprint('Best score:', self.maks)
            self._iter_callback()
            iters += 1
        print('max iters reached.')
        return self.best

    def _iter_callback(self):
        for i in range(self.nbits):
            super()._iter_callback()


class RandomizedRestarts(RandomSearcher):
    """Perform search with randomized restarts.

    Arguments:
    ----------------
    nbits: int, default 8. number of binary features
    n_iters: int, default 10. number of iterations
    base_optimizer: solver.Solver, default HillClimber. Solver to be used in search.
    verbose: int, default 0. verbosity
    optimizer_kw: dict or None, default None. Keywords to pass on to base solver.
    """

    def __init__(self, nbits=8, n_iters=10, base_optimizer=None, verbose=0, optimizer_kw=None):
        if optimizer_kw is None:
            optimizer_kw = {}
        super().__init__(nbits, verbose=verbose)
        self.n_iters = n_iters
        if base_optimizer is None:
            base_optimizer = HillClimber
        self.optimizers = [base_optimizer(
            nbits=nbits, **optimizer_kw) for _ in range(n_iters)]

    def _a_search(self, problem):
        for i, optimizer in enumerate(self.optimizers):
            optimizer._a_search(problem)
            self.vprint(f'Search {i+1}/{self.n_iters} completed.')
            self.vprint(f'Score: {optimizer.maks}')

            best_optimizer = max(self.optimizers, key=lambda x: x.maks)
            self.best = best_optimizer.best
            self.maks = best_optimizer.maks
            self.solution_arr += [(max(self.optimizers, key=lambda x: x.maks).maks)] * len(optimizer.solution_arr)

        return self.best

    def fit(self, problem):
        for optimizer in self.optimizers:
            optimizer.fit(problem)
        return self


class Annealer(Solver):
    """Simulated Annealer.

    Arguments:
    ----------------
    nbits: int, number of binary features
    initial_temperature: float, starting temperature to calculate jump probability.
                         Higher temperatures mean more exploration.
    decay_rate: float, (0, 1), default 0.99. cooldown rate. Temperature is
                multiplied by this rate after each evaluation.
    patience: int, default 20. Number of iterations until stopping search if no
              improvement is made.
    verbose: int, default 0. verbosity
    """

    def __init__(self, nbits=8, initial_temperature=50, decay_rate=0.99, patience=20, max_iters=200000, verbose=0):
        super().__init__(nbits, verbose=verbose)
        self.temperature = initial_temperature
        self.decay_rate = decay_rate
        self.patience = patience
        self.max_iters = max_iters

    def cooldown(self):
        """Lower the temperature."""
        self.temperature *= self.decay_rate

    def get_jump_proba(self, f0, f1):
        """Calculate jump probability."""
        if f1 >= f0:
            return 1.
        else:
            return np.exp((f1 - f0) / self.temperature)

    def _a_search(self, problem):
        point = get_random_bits(self.nbits) if self.initial_solution is not None else self.initial_solution
        wait = 0
        iters = 0
        while True:
            f0 = self.query(problem, point, )
            # neighbor = np.random.choice(get_neighbors(point))
            neighbor = get_random_neighbor(point)
            f1 = self.query(problem, neighbor, )
            maks = max(f0, f1)
            if not np.isnan(maks) and maks >= self.maks:
                self.maks = maks
                self.best = [point, neighbor][np.array([f0, f1]).argmax()]
            self._iter_callback()
            proba = self.get_jump_proba(f0, f1)
            self.cooldown()
            # import pdb; pdb.set_trace()
            if np.isnan(proba):
                proba = 1.
            jump = bool(np.random.choice([0, 1], p=[1 - proba, proba]))
            if iters % 100 == 0:
                self.vprint(proba)
                self.vprint('Temperature:', self.temperature)
                self.vprint('Best score:', max(f0, f1))
            iters += 1
            if iters >= self.max_iters:
                break
            if jump:
                point = neighbor
                wait = 0
            else:
                wait += 1
            if wait >= self.patience:
                break
        return self.best


def to_bits(bits: np.ndarray):
    return np.array([''.join(s) for s in bits.astype(int).astype(str)])


def to_array(bits: List[str]):
    return np.array([list(b) for b in bits]).astype(int)


class Genetic(Solver):
    """Survival of the fittest.

    Arguments:
    ----------------
    nbits: int, number of binary features
    mutation_proba: float, [0, 1]. Mutation probability for every single bit.
    crossover_len: int, [1, nbits]. Designate crossover schema.
                   e.g. for crossover_len=3, cross-over is performed for every 3
                   bits.
    population_size: int, default 100. Size of population to be kept.
    patience: int, default 100. Number of iterations until stopping search if no
              improvement is made.
    survival_function: callable, default None.
        A survival function that accepts fitness scores and returns indices.
        Default is a Gaussian centered on maximum score with scale=1.0.
    sigma: float, default 1.0
        Standard deviation of Gaussian for survival. (only relevant if survival_function
        is None.)
    verbose: int, default 0. verbosity
    """

    def __init__(self, nbits, mutation_proba=0.1, crossover_len=1, population_size=100, patience=100, survival_function=None, sigma=1.0, verbose=0):
        super().__init__(nbits, verbose=verbose)
        self.mutation_proba = 0.1
        self.crossover_len = crossover_len
        self.population_size = population_size
        self.patience = patience
        self._population_means = []
        if survival_function is None:
            survival_function = self._survive
        self.survival_function = survival_function
        self.sigma = sigma

    def generate_population(self):
        return np.array([
            get_random_bits(self.nbits) for i in range(self.population_size)
            ])

    def _iter_callback(self):
        super()._iter_callback()
        self._population_means.append(self._population_scores.mean())

    def evaluate_population(self, problem, population):
        return np.array([self.query(problem, s) for s in population])

    def crossover(self, p1, p2):
        n = len(p1) // self.crossover_len

        p2_inds = [(i * self.crossover_len, (i + 1) * self.crossover_len)
                   for i in range(1, n, 2)]
        child = np.array(list(p1))
        p1 = np.array(list(p1))
        p2 = np.array(list(p2))
        for left, right in p2_inds:
            child[left:right] = p2[left:right]
        pmutate = np.random.random(size=child.shape[0])
        f = pmutate < self.mutation_proba
        child = child.astype(int)
        child[f] = 1 - child[f]
        child = child.astype(str)
        # for i in range(child.shape[0]):
        #     if np.random.random() < self.mutation_proba:
        #         child[i] = str(int((not bool(int(child[i])))))
        child = ''.join(child)
        return child

    def _a_search(self, problem):
        p = self.generate_population()
        wait = 0
        while True:
            fits = self.evaluate_population(problem, p)
            self._population = p
            self._population_scores = fits
            maks = fits.max()
            if maks > self.maks:
                self.maks = maks
                self.best = p[fits.argmax()]
                wait = 0
            else:
                wait += 1
            self.solution_arr += [self.maks] * self.population_size
            # self._iter_callback()
            self.vprint('Best score:', self.maks)

            if wait >= self.patience:
                break

            inds = self.survival_function(fits)
            if not list(inds):
                min_size = min(self.population_size, 10)
                warnings.warn(f'No one would survive. Selecting best {min_size}')
                inds = fits.argsort()[::-1][:min_size]

            p = p[inds]
            fits = fits[inds]
            children = []
            psize = p.shape[0]
            while psize < self.population_size:
                # parent1, parent2 = np.random.choice(p, p=neww, size=2)
                parent1, parent2 = np.random.choice(p, size=2)
                child = self.crossover(parent1, parent2)
                children.append(child)
                psize += 1
            p = np.concatenate([p, children])
        return self.best

    def _survive(self, scores, sigma=1.0):
        """Return indices of survivors."""
        distro = scs.norm(loc=scores.max(), scale=sigma)
        probas = distro.pdf(scores)
        probas = probas / probas.sum()
        is_survivor = [np.random.choice([True, False], p=[p, 1-p]) for p in probas]
        inds = np.arange(scores.shape[0])
        return inds[is_survivor]


class MIMIC(Genetic):
    """Mutual-Information-Maximizing Input Clustering (MIMIC)"""

    def __init__(self, nbits=8, population_size=200, patience=30, tol=1e-4, shrink=0.50, verbose=0, gain_func=None):
        super().__init__(nbits, population_size=population_size, verbose=verbose, patience=patience)
        self.tol = tol
        self.shrink = shrink
        if gain_func is None:
            gain_func = metrics.mutual_info_score
        self.gain_func = gain_func

    def calculate_gain(self, population):
        population = to_array(population)
        if self.gain_func == 'corr':
            info = np.abs(np.corrcoef(population, rowvar=False))
            info[np.isnan(info)] = 0.
            info = info - np.eye(*info.shape)
        else:
            info = self._calculate_gain(population)
        return info

    def _calculate_gain(self, population):
        info = metrics.pairwise._pairwise_callable(
            population.T, population.T, self.gain_func)
        info += 1e-12
        info = info * (1. - np.zeros_like(info) - np.diag(np.ones_like(info[0]))) + np.diag(np.ones_like(info[0]))
        return info

    def resample(self, population):
        population = np.array([list(p) for p in population]).astype(int)
        samples = np.zeros((self.population_size, population.shape[1]))
        for sample in samples:
            self.dep_tree.sample(sample)
        # import pdb; pdb.set_trace()
        # samples[:population.shape[0],:] = population
        samples = to_bits(samples)
        return samples

    def _a_search(self, problem):
        theta = -float('inf')
        self.tmin = theta
        wait = 0
        self.vprint('Initial sampling...')
        p = self.generate_population()
        while True:
            self.vprint('evaluate_population...')
            fits = self.evaluate_population(problem, p)
            self._population_scores = fits
            maks = fits.max()
            if maks > self.maks:
                self.maks = maks
                self.best = p[fits.argmax()]
            self._iter_callback()
            # self.solution_arr += [self.maks] * self.population_size
            self.vprint('shrinking...')
            # new_theta = max(self.tmin, np.percentile(fits, self.shrink * 100))
            # new_theta = max(theta, np.percentile(fits, self.shrink * 100))
            new_theta = np.percentile(fits, self.shrink * 100)

            self.tmin = new_theta
            # new_theta = fits.max() * self.shrink
            self.vprint('New theta:', new_theta)
            self.vprint('Best score:', self.maks)
            # if abs(theta - new_theta) <= self.tol:
            if new_theta <= theta:
                wait += 1
                # import pdb; pdb.set_trace()
                if wait > self.patience:
                    break
            theta = new_theta
            p = p[fits >= theta]
            self.vprint('calculate_gain...')
            info = self.calculate_gain(p)
            # import pdb; pdb.set_trace()
            if np.allclose(info, 1):
                warnings.warn('Population is identical.')
                import pdb; pdb.set_trace()
                p = self.generate_population()
            else:
                self._calculate_deps(info, p)
                p = self.resample(p)
            # import pdb; pdb.set_trace()
        return self.best

    def _calculate_deps(self, info, p):
        self.graph = Graph(self.nbits, 1 - info)
        agac = self.graph.calculate_deps()
        self.dep_tree = DependencyTree(0, [a[0] for a in agac])
        self.vprint('estimate_proba...')
        self.dep_tree.estimate_proba(p)
        self.vprint('resample...')



def to_int(B):
    return np.array([list(b) for b in B]).astype(int)


def to_bits(S):
    return np.array([''.join(row) for row in S.astype(int).astype(str)])


class DependencyTree:
    def __init__(self, name, childs=None, parent=None):
        self.name = name
        if childs is None:
            childs = []
        mychilds = [c[1] for c in childs if c[0] == name]
        self.parent = parent
        self.childs = [DependencyTree(i, childs, parent=self) for i in mychilds]
        self.distros = {c.name: {0: 0.5, 1: 0.5} for c in self.childs}
        self.proba = 0.5

    def estimate_proba(self, X):
        Xi = to_int(X)
        if self.parent is None:
            self.proba = Xi.T[self.name].mean()
        for child in self.childs:
            try:
                # self.distros[child.name][0] = ((Xi.T[self.name] == 0) & (Xi.T[child.name] == 1)).mean()
                self.distros[child.name][0] = Xi[Xi.T[self.name]==0].T[child.name].mean()
                self.distros[child.name][1] = Xi[Xi.T[self.name]==1].T[child.name].mean()
                # self.distros[child.name][1] = ((Xi.T[self.name] == 1) & (Xi.T[child.name] == 1)).mean()
            except Exception as e:
                print(e)
                import pdb; pdb.set_trace()
            child.estimate_proba(X)

    def sample(self, s):
        if self.parent is None:
            s[self.name] = np.random.choice([0, 1], p=[1 - self.proba, self.proba])
        for c in self.childs:
            p = self.distros[c.name][s[self.name]]
            s[c.name] = np.random.choice([0, 1], p=[1 - p, p])
            c.sample(s)

    def get_tree(self):
        return {c.name: c.get_tree() for c in self.childs}
