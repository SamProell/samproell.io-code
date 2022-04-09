# digitalfilter.py

from collections import deque

import numpy as np


class LiveFilter:
    """Base class for live filters.
    """
    def process(self, x):
        # do not process NaNs
        if np.isnan(x):
            return x

        return self._process(x)

    def __call__(self, x):
        return self.process(x)

    def _process(self, x):
        raise NotImplementedError("Derived class must implement _process")


class LiveLFilter(LiveFilter):
    """Live implementation of digital filter using difference equations.

    The following is almost equivalent to calling scipy.lfilter(b, a, xs):
    >>> lfilter = LiveLFilter(b, a)
    >>> [lfilter(x) for x in xs]
    """
    def __init__(self, b, a):
        """Initialize live filter based on difference equation.

        Args:
            b (array-like): numerator coefficients obtained from scipy
                filter design.
            a (array-like): denominator coefficients obtained from scipy
                filter design.
        """
        self.b = b
        self.a = a
        self._xs = deque([0] * len(b), maxlen=len(b))
        self._ys = deque([0] * (len(a) - 1), maxlen=len(a)-1)

    def _process(self, x):
        """Filter incoming data with standard difference equations.
        """
        self._xs.appendleft(x)
        y = np.dot(self.b, self._xs) - np.dot(self.a[1:], self._ys)
        y = y / self.a[0]
        self._ys.appendleft(y)

        return y


class LiveSosFilter(LiveFilter):
    """Live implementation of digital filter with second-order sections.

    The following is equivalent to calling scipy.sosfilt(sos, xs):
    >>> sosfilter = LiveSosFilter(sos)
    >>> [sosfilter(x) for x in xs]
    """
    def __init__(self, sos):
        """Initialize live second-order sections filter.

        Args:
            sos (array-like): second-order sections obtained from scipy
                filter design (with output="sos").
        """
        self.sos = sos

        self.n_sections = sos.shape[0]
        self.state = np.zeros((self.n_sections, 2))

    def _process(self, x):
        """Filter incoming data with cascaded second-order sections.
        """
        for s in range(self.n_sections):  # apply filter sections in sequence
            b0, b1, b2, a0, a1, a2 = self.sos[s, :]

            # compute difference equations of transposed direct form II
            y = b0*x + self.state[s, 0]
            self.state[s, 0] = b1*x - a1*y + self.state[s, 1]
            self.state[s, 1] = b2*x - a2*y
            x = y  # set biquad output as input of next filter section.

        return y
