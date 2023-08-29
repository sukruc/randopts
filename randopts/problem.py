from itertools import product
from abc import ABC, abstractmethod, abstractproperty
import numpy as np
from typing import List, Tuple
from functools import lru_cache
from sklearn import neural_network, preprocessing, metrics
import bitstring


class DiscreteProblem:
    """Base class for discrete problems."""
    query_cost_td = 0.
    params = ('cost_per_query',)

    def __init__(self, *args, cost_per_query=None, **kwargs):
        if cost_per_query is None:
            cost_per_query = 1.
        self.cost_per_query = cost_per_query

    def evaluate(self, query: str) -> float:
        """Verilen parametre setini uygunluk fonksiyonuna sokup sonucu ver."""
        result = self.fitness_function(query)
        self.incur_cost()
        return result

    def fitness_function(self, query: str) -> float:
        """Her probleme ozel uygunluk fonksiyonu."""
        pass

    def incur_cost(self):
        """Toplam maliyeti arttir."""
        self.query_cost_td += self.cost_per_query

    def get_params(self, deep=True):
        return {k: getattr(self, k) for k in self.params}

    def copy(self):
        return self.__class__(**self.get_params())

    @property
    def required_bits(self):
        if hasattr(self, 'nbits'):
            return getattr(self, 'nbits')


class SomeRandomProblem(DiscreteProblem):
    def fitness_function(self, query: str):  # '10101001010'
        """+8 if bits 1 and 8 the same, -4 if bits 1 and 3 are different."""
        skor = (query[1] == query[5]) * 8 - (query[1] != query[3]) * 4
        return skor


class KBitProblem(DiscreteProblem):
    """Guess my 8-bit string.

    Arguments:
    ----------------
    sifrem: str, target.
    """
    params = ('sifrem', 'cost_per_query')

    def __init__(self, sifrem: str, cost_per_query=None):
        super().__init__(cost_per_query=cost_per_query)
        self.sifrem = np.array(list(sifrem))

    def fitness_function(self, query):
        """Return number of the matching bits."""
        return (np.array(list(query)) == self.sifrem).sum()


class TravelingSalesman(DiscreteProblem):
    """Visit all the cities, with minimum cost."""
    def __init__(self, cost_matrix, cost_per_query=None):
        super().__init__(cost_per_query=cost_per_query)
        self.cost_matrix = cost_matrix

    def cost_matrix():
        doc = "The cost_matrix property."
        def fget(self):
            return self._cost_matrix
        def fset(self, value):
            self._cost_matrix = np.array(value)
        def fdel(self):
            del self._cost_matrix
        return locals()
    cost_matrix = property(**cost_matrix())

    @property
    def required_bits(self):
        return self.cost_matrix.shape[0] ** 2

    def fitness_function(self, query):
        query = np.array(list(query)).astype(int).reshape(*self.cost_matrix.shape)
        if not ((query.sum(axis=1) == 1).all() and (query.sum(axis=0) == 1).all()):
            fitness = -query.sum()
        elif (self.cost_matrix[query == 1] < 0).any():
            fitness = -500
        else:
            fitness = 1 / self.cost_matrix[query == 1].sum()
        return fitness



class Knapsack(DiscreteProblem):
    """3 things you would take to a Desert Island."""
    params = ('weights', 'values', 'limit')

    def __init__(self, weights: List[float], values: List[float], limit: float, cost_per_query=None):
        super().__init__(cost_per_query=cost_per_query)
        assert len(weights) == len(values)
        self.weights = np.array(weights)
        self.values = np.array(values)
        self.limit = limit

    def fitness_function(self, query):
        inds = np.array(list(query)).astype(int).astype(bool)
        selection = self.weights[inds]
        top = selection.sum()
        if top > self.limit:
            return self.limit - top
        return self.values[inds].sum()

    @property
    def required_bits(self):
        return self.weights.shape[0]


class CustomProblem(DiscreteProblem):
    """Create a new problem with a custom fitness function."""
    params = ('fitness_function', 'cost_per_query', 'nbits')

    def __init__(self, fitness_function, nbits, cost_per_query=None):
        super().__init__(cost_per_query=cost_per_query)
        self.fitness_function = fitness_function
        self.nbits = nbits


class AlternateProblem(DiscreteProblem):
    """
    # '101' -> 2
    # '111' -> 0
    """
    params = ('nbits', 'cost_per_query')

    def __init__(self, nbits, cost_per_query=None):
        super().__init__(cost_per_query=cost_per_query)
        self.nbits = nbits

    def fitness_function(self, query):
        """Alternate bits for maximum gain."""
        return (np.array(list(query[:-1])) != np.array(list(query[1:]))).sum()

    @property
    def required_bits(self):
        return self.nbits


class Interpreter:
    """Base class for interpreters."""
    precision = None

    @abstractmethod
    def convert(self, bits: str) -> float:
        pass


class BitFloatInterpreter(Interpreter):
    """Floating point interpreter."""

    def convert(self, bits: str) -> float:
        result = bitstring.BitArray(bin='0' + bits+(32 - self.precision)*'0').float
        if np.isnan(result):
            result = 0.
        return result


class BitStringInterpreter(Interpreter):
    """Integer interpreter."""

    def __init__(self, precision=4):
        self.precision = precision

    def convert(self, bits: str) -> int:
        result = bitstring.BitArray(bin='0' + bits).int
        return result
    
    def encode(self, value):
        raw = bin(value)[2:]
        if len(raw) > self.precision:
            raise OverflowError
        return raw.zfill(self.precision)


class BitFloat16Interpreter(BitFloatInterpreter):
    """16-bit float interpreter."""
    precision = 16


class BitFloat32Interpreter(BitFloatInterpreter):
    """32-bit float interpreter."""
    precision = 32


class CustomInterpeter:
    """Sometimes simple is better."""
    table = {'00': 0., '01': 1., '10': -1, '11': 0.2}
    precision = 2

    def convert(self, bits):
        return self.table[bits]


class WeightConverter:
    """Interpret bit strings as neural network weights."""

    def __init__(self, network, converter, scale=True):
        self.coef_dims = [layer.shape for layer in network.coefs_]
        self.coef_sizes = [layer.size for layer in network.coefs_]
        self.intercept_dims = [layer.shape for layer in network.intercepts_]
        self.intercept_sizes = [layer.size for layer in network.intercepts_]
        self.precision = converter.precision
        self.converter = converter
        self.scale = scale

    @property
    def required_bits(self):
        """Number of required input bits."""
        return (sum(self.coef_sizes) + sum(self.intercept_sizes)) * self.precision

    def to_float(self, bit_string: str) -> List[float]:
        """Convert bit strings to float."""
        floats = []
        i = 0
        while i < len(bit_string):
            floats.append(self.converter.convert(bit_string[i:i + self.precision]))
            i += self.precision
        return floats

    def rebuild(self, flat_coefs: List[float]) -> Tuple[Tuple[np.ndarray]]:
        """"""
        i = 0
        coefs = []
        intercepts = []
        # import pdb; pdb.set_trace()
        for dim, size in zip(self.coef_dims, self.coef_sizes):
            built = np.array(flat_coefs[i:i+size]).reshape(dim)
            if self.scale:
                built = (built - built.mean()) / built.std()
                built[np.isnan(built)] = 0.
            coefs.append(built)
            i += size

        for dim, size in zip(self.intercept_dims, self.intercept_sizes):
            built = np.array(flat_coefs[i:i+size]).reshape(dim)
            if self.scale:
                built = (built - built.mean()) / built.std()
                built[np.isnan(built)] = 0.
            intercepts.append(built)
            i += size

        return coefs, intercepts

    def interpret(self, bit_string: str) -> Tuple[np.ndarray]:
        floats = self.to_float(bit_string)
        coefs, intercepts = self.rebuild(floats)
        return coefs, intercepts



def logistic(x):
    return 1. / (1. + np.exp(-x))


def relu(x):
    x[x < 0] = 0
    return x


def tanh(x):
    return np.tanh(x)


def binary_crossentropy(y_true, y_pred):
    return (y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred)).sum()


class NeuralProblem(DiscreteProblem):
    """Neural network weight optimization problem.

    Arguments:
    ---------------
    X: np.ndarray (M, N), features
    y: np.ndarray (M,), labels
    network: neural_network.MLPClassifier
    converter:
    """
    params = ('X', 'y', 'network', 'cost_per_query', 'converter', 'scoring')
    tanh = staticmethod(tanh)
    relu = staticmethod(relu)
    logistic = staticmethod(logistic)

    def __init__(self, X, y, network, converter, cost_per_query=None, copy=True, scale=True, scoring='accuracy'):
        super().__init__(cost_per_query=cost_per_query)
        if copy:
            X = X.copy()
            y = y.copy()
        self.X = preprocessing.scale(X)
        self.y = y
        self.network = network
        self.precision = converter.precision
        self.converter = converter
        self._init_network()
        self.interpeter = WeightConverter(self.network, self.converter)
        self.scoring = scoring

    def _init_network(self):
        dX = np.random.randn(1, self.X.shape[1])
        dy = [1.]
        self.network.fit(dX, dy)

    def _predict(self, X=None):
        proba = self._predict_proba(X)
        return (proba > .5).astype(int)

    def predict(self, X):
        return self._predict(X)

    def score(self, X, y):
        return self._score(X, y)

    def _predict_proba(self, X=None):
        if X is None:
            X = self.X
        a = X
        i = -1
        for i in range(len(self.network.coefs_) - 1):
            a = getattr(self, self.network.activation)(a @ self.network.coefs_[i] + self.network.intercepts_[i])
        i += 1
        # import pdb; pdb.set_trace()
        a = getattr(self, 'logistic')(a @ self.network.coefs_[i] + self.network.intercepts_[i])
        return a.ravel()

    def _score(self, X=None, y=None):
        if y is None:
            y = self.y
        pred = self._predict(X)
        return metrics.accuracy_score(y, pred)

    def fitness_function(self, query):
        coef, intercept = self.interpeter.interpret(query)
        self.network.coefs_ = coef
        self.network.intercepts_ = intercept
        if self.scoring == 'accuracy':
            pred = self._predict()
            score = metrics.accuracy_score(self.y, pred)
        elif self.scoring == 'loss':
            pred = self._predict_proba()
            score = binary_crossentropy(self.y, pred)
        return score

    @property
    def required_bits(self):
        return self.interpeter.required_bits

    def get_weights(self):
        return self.network.coefs_, self.network.intercepts_

    def set_weights(self, bitweights: str):
        coef, intercept = self.interpeter.interpret(bitweights)
        self.network.coefs_ = coef
        self.network.intercepts_ = intercept


def get_3bits(s):
    n = len(s) // 3
    for i in range(n):
        yield s[i*3:(i+1)*3]


def get_nbits(s, k):
    n = len(s) // k
    for i in range(n):
        yield s[i*k:(i+1)*k]


def to_dec(bits):
    groups = get_3bits(bits)
    s = [str(eval('0b' + i)) for i in groups]
    s = ''.join(s)
    return s


def hex_to_dec(bits):
    groups = get_nbits(bits, 4)
    s = [hex(eval('0b' + i))[-1].upper() for i in groups]
    s = ''.join(s)
    return s


class BullsAndCows(KBitProblem):
    """Orijinal bull&cow sacmalatmaca.

    Aklimdaki sayiyi 2-bit parcalara bolup bulself.

    Ornek:
    Sayim : 243
    Senden istedigim tahmin: 010100011
    Aciklama: 010-100-011  -> sirasiyla oktal olarak 2-4-3'u temsil eder.
    """
    params = ('sifrem', 'cbull', 'ccow', 'cost_per_query')
    converter = staticmethod(to_dec)
    base = staticmethod(int)
    ub = 7
    charlen = 3

    def __init__(self, sifrem: str, cbull=10, ccow=2, cost_per_query=None):
        """
        Argumanlar:
        ----------------
        cbull: float, bull'larin skora etki katsayisi
        ccow: float, cow'larin skora etki katsayisi
        cost_per_query: sorgu maliyeti
        """
        super().__init__(sifrem, cost_per_query)
        self.cbull = cbull
        self.ccow = ccow

    def fitness_function(self, query):
        """Yerleri ayni olanlar bull, ayni olmayanlar cow."""
        qarr = np.array(list(self.converter(query)))
        bulls = (qarr == self.sifrem).sum()
        fcow = qarr != self.sifrem
        cows = len(set(qarr[fcow]) & set(self.sifrem[fcow]))
        return bulls * self.cbull + cows * self.ccow

    @property
    def required_bits(self):
        return len(self.sifrem)* self.charlen

    def sifrem():
        doc = "The sifrem property."
        def fget(self):
            return self._sifrem
        def fset(self, value):
            for i in value:
                if not 0 <= self.base(i) <= self.ub:
                    raise ValueError('Yalnizca oktal sayilara izin verilir: 0< =i <=7')
            self._sifrem = value
        def fdel(self):
            del self._sifrem
        return locals()
    sifrem = property(**sifrem())


class BullsAndCowsHex(BullsAndCows):
    """CCAB420F91A4FEEC45D764DFBB346650
    """
    converter = staticmethod(hex_to_dec)
    base = staticmethod(lambda x: eval('0x'+x))
    ub = 15
    charlen = 4

class BitStringNormalizedInterpreter(BitStringInterpreter):
    """Integer interpreter."""

    def __init__(self, precision=4, ub=1):
        self.precision = precision
        self.ub = ub

    def convert(self, bits: str) -> int:
        result = super().convert(bits) / 2**self.precision * self.ub
        return result
    
    def encode(self, value):
        if value > self.ub:
            raise OverflowError
        val = bin(int(value / self.ub * 2**self.precision))[2:].zfill(self.precision)
        return val

