"""Independent PRNG and literal-window implementation for stream checks."""
import numpy as np
from calibration import baci_statistic

MASK = 2**64-1


def splitmix64(x: int) -> int:
    x = (x + 0x9e3779b97f4a7c15) & MASK
    x = ((x ^ (x >> 30))*0xbf58476d1ce4e5b9) & MASK
    x = ((x ^ (x >> 27))*0x94d049bb133111eb) & MASK
    return x ^ (x >> 31)


class MT19937_64:
    """Independent direct Python implementation of the standard engine."""
    def __init__(self, seed):
        self.state = [seed & MASK]
        for i in range(1,312):
            prev = self.state[-1]
            self.state.append((6364136223846793005*(prev ^ (prev >> 62))+i) & MASK)
        self.index = 312

    def __call__(self):
        if self.index >= 312:
            # The in-place recurrence deliberately wraps to the updated prefix,
            # matching the standardized linear recurrence beyond the old block.
            for i in range(312):
                x = (self.state[i] & 0xffffffff80000000) | (self.state[(i+1)%312] & 0x7fffffff)
                self.state[i] = self.state[(i+156)%312] ^ (x >> 1) ^ (0xb5026f5aa96619e9 if x&1 else 0)
            self.index = 0
        y = self.state[self.index]
        self.index += 1
        y ^= (y >> 29) & 0x5555555555555555
        y ^= (y << 17) & 0x71d67fffeda60000
        y ^= (y << 37) & 0xfff7eee000000000
        y ^= y >> 43
        return y & MASK


def independent_trial(rows, columns, repeats, seed_base, trial, fixed=None):
    rng = MT19937_64(splitmix64(seed_base+trial))
    values = []
    for r in range(repeats):
        pixels = []
        while len(pixels) < rows*columns:
            word = rng()
            chunks = 4 if fixed is None else 8
            for k in range(chunks):
                if len(pixels) == rows*columns:
                    break
                if fixed is None:
                    pixels.append(abs((word & 255)-((word >> 8)&255)))
                    word >>= 16
                else:
                    pixels.append(abs(int(fixed.flat[len(pixels)])-(word & 255)))
                    word >>= 8
        a = np.array(pixels, dtype=np.uint8).reshape(rows, columns)
        values.append(int(baci_statistic(a, exact=True, method='literal') * (6*(rows-1)*(columns-1)*255)))
    return np.array(values,dtype=np.uint32)

