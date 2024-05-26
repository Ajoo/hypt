import numpy as np
from abc import ABC, abstractmethod
from typing import Union, Optional, Any
from numpy.typing import ArrayLike
from numpy.random import Generator

from hypt.static import StaticParams

__all__ = [
    'UniformCategorical', 'UniformInt', 'UniformPower',
    'Uniform', 'UniformLog', 'UniformIntLog', 'LogUniform', 'IntLogUniform',
    'OrValue', 'OrZero', 'ProductDistribution'
]


@staticmethod
def _identity(x: Any):
    return x


def _log(x, base):
    if base == 2:
        return np.log2(x)
    elif base == 10:
        return np.log10(x)

    return np.log(x)/np.log(base)


class Distribution(ABC):
    """Distribution base class.
    """
    @abstractmethod
    def sample(self, size: int, prng: Generator = None):
        pass


Seed = Union[None, int, Generator]


def _sample_method(sampler):
    def sampler_with_default_prng(self, size: int, seed: Seed = None) -> ArrayLike:
        """Samples from distribution.

        Args:
            size (int): Number of samples.
            prng (Seed, optional): Optional seed. Defaults to None.

        Returns:
            ArrayLike: Array of samples.
        """
        prng = np.random.default_rng(seed)
        return sampler(self, size, prng)

    return sampler_with_default_prng


# Discrete

class UniformCategorical(Distribution):
    """Uniform distribution over a categorical space.

    Defines a uniform distribution over a specified categorical space.

    Args:
        values (ArrayLike): Possible categorical values
    """

    def __init__(self, values: ArrayLike):
        self.values = np.asarray(values)

    @_sample_method
    def sample(self, size, prng):
        return prng.choice(self.values, size=size, replace=True)


class DiscreteNumericalDistribution(Distribution):
    """Uniform distribution over the range of a sequence.

    Defines a uniform distribution over the range of a given sequence.
    I.e., specifies a uniform distribution over a domain:
        {sequence(i): i in [start, stop[}

    A `sequence` method or attribute should be defined by a subclass.

    Args:
        start (int): Starting index of the sequence.
        stop (int): Stopping index of the sequence (exclusive).
    """

    def __init__(self, start: int, stop: Optional[int] = None):
        if stop is None:
            stop = start
            start = 0

        self.start = start
        self.stop = stop

    @_sample_method
    def sample(self, size: int, prng: Generator):
        i = prng.integers(self.start, self.stop, size=size)
        return self.sequence(i)


class UniformInt(DiscreteNumericalDistribution):
    """Uniform distribution an integer range.

    Defines a uniform distribution over a specified integer range.

    Args:
        start (int): Starting index.
        stop (int): Stopping index (exclusive).
    """
    sequence = _identity

    @staticmethod
    def from_domain(start: int, stop: Optional[int] = None):
        return UniformInt(start, stop)


class UniformPower(DiscreteNumericalDistribution):
    """Uniform distribution on the sequence of powers of a base.

    The domain of the uniform distribution is:
        {base**i: i in [start, stop[}

    Args:
        start (int): Starting index.
        stop (int): Stopping index.
    """

    def __init__(self, start, stop, base=2):
        super().__init__(start, stop)
        self.base = base

    @staticmethod
    def from_domain(start: int, stop: int, base=2):
        start = np.round(_log(start, base)).astype(np.int_)
        stop = np.round(_log(stop, base)).astype(np.int_)
        return UniformPower(start, stop)

    def sequence(self, i):
        return np.power(self.base, i)


# Continuous

class StaticInverseMixin:
    """Provides initialization from values in the target domain
    for classes that can define a static inverse function.
    """
    @classmethod
    def from_domain(cls, lb: float, ub: float):
        lb = cls.inverse_function(lb)
        ub = cls.inverse_function(ub)
        return cls(lb, ub)


class ContinuousNumericalDistribution(Distribution):
    """Defines a distribution by transformation of a continuous uniform random variable.

    Defines a distribution of the random variable f(U) where U ~ Uniform([lb, ab]).
    f is given by a `function` method or attribute that should be defined by a subclass.

    Args:
        lb (float): Lower bound.
        ub (float): Upper bound.
    """

    def __init__(self, lb: float, ub: float):
        self.lb = lb
        self.ub = ub

    @_sample_method
    def sample(self, size: int, prng: Generator):
        x = prng.uniform(self.lb, self.ub, size=size)
        return self.function(x)


class Uniform(StaticInverseMixin, ContinuousNumericalDistribution):
    """Uniform distribution over a bounded domain.

    Defines a uniform distribution over a bounded domain [lb, ub].

    Args:
        lb (float): Lower bound.
        ub (float): Upper bound.
    """
    function = _identity
    inverse_function = _identity


class UniformLog(StaticInverseMixin, ContinuousNumericalDistribution):
    """Log-uniform distribution.

    Defines a log-uniform distribution over a bounded domain [exp(lb), exp(ub)].
    To specify the domain [a, b] directly (instead of log) use instead
    `LogUniform.from_domain(a, b)`

    Args:
        lb (float): Lower bound (in log domain).
        ub (float): Upper bound (in log domain).
    """
    function = staticmethod(np.exp)
    inverse_function = staticmethod(np.log)


class UniformIntLog(UniformLog):
    """Distribution for the floor of a Log-Uniform random variable.

    Distribution for floor(L) where L is a log-uniform distribution over a bounded 
    domain [exp(lb), exp(ub)].

    To specify the domain [a, b] directly (instead of log) use instead
    `LogUniformInt.from_domain(a, b)`

    Args:
        lb (float): Lower bound (in log domain).
        ub (float): Upper bound (in log domain).
    """

    @staticmethod
    def function(x):
        return np.exp(x).astype(np.int_)


# Alias
def LogUniform(lb: float, ub: float):
    """Log-uniform distribution.

    Defines a log-uniform distribution over a bounded domain [lb, ub].

    Args:
        lb (float): Lower bound.
        ub (float): Upper bound.
    """
    return UniformLog.from_domain(lb, ub)


def IntLogUniform(lb: float, ub: float):
    """Int Log-uniform distribution.

    Defines a log-uniform distribution over a bounded domain [lb, ub].
    After sampling, floor is taken to make the result an int.

    Args:
        lb (float): Lower bound.
        ub (float): Upper bound.
    """
    return UniformIntLog.from_domain(lb, ub)


# Mixture

class OrValue(Distribution):
    """Mixture between specified distribution and a fixed value.

    Samples are either a fixed `value` (with `prob_value` probability) or sampled from
    a specified `distribution` (with `1 - prob_value` probability).

    Args:
        distribution (Distribution): The distribution part of the mixture.
        value (Any): The fixed value.
        prob_value (float, optional): The mixture probability for the value. Defaults to 0.5.
    """

    def __init__(self, distribution: Distribution, value: Any, prob_value: float = 0.5):
        self.distribution = distribution
        self.value = value
        self.pvalue = prob_value

    @_sample_method
    def sample(self, size, prng):
        i = prng.uniform(size=size) > self.pvalue
        nonzero = self.distribution.sample(np.sum(i), seed=prng)

        output = np.full(size, self.value, dtype=np.promote_types(nonzero.dtype, type(self.value)))
        output[i] = nonzero
        return output


class OrZero(OrValue):
    """Mixture between specified distribution and a zero.

    Samples are either 0 (with `prob_zero` probability) or sampled from
    a specified `distribution` (with `1 - prob_zero` probability).

    Args:
        distribution (Distribution): The distribution part of the mixture.
        prob_zero (float, optional): The mixture probability for zero. Defaults to 0.5.
    """

    def __init__(self, distribution: Distribution, prob_zero=0.5):
        super().__init__(distribution, 0, prob_zero)


ParamsDistributions = dict[str, Union[Distribution, Any]]


class ProductDistribution(Distribution):
    """Product of marginal distributions.

    Samples each parameter independently from its own distribution.

    Args:
        distributions (dict): dictionary where keys are parameter names and values are Distributions or fixed values.
    """

    def __init__(self, distributions: ParamsDistributions):
        self.distributions = distributions

    @_sample_method
    def sample(self, size, prng):
        dynamic = {}
        static = {}
        for name, distribution in self.distributions.items():
            if isinstance(distribution, Distribution):
                dynamic[name] = distribution.sample(size, prng)
            else:
                static[name] = distribution

        return StaticParams(dynamic, static, size)


def RandomSearch(parameter_distributions: ParamsDistributions, size: int, seed: Seed = None) -> StaticParams:
    """Samples parameters for a Random Search.

    Samples each parameter from its own distribution as specified in `parameter_distributions`.

    Args:
        parameter_distributions (ParamsDistributions): Specification of per parameter distributions.
            keys are parameter names and values are Distributions or fixed.
        size (int): Number of samples.
        seed (Seed, optional): Seed for the sampling. Can be an integer or a numpy Generator.

    Returns:
        StaticParams: Iterable over sampled parameters.
    """

    return ProductDistribution(parameter_distributions).sample(size, seed)
