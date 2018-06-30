'''Utility for plotting a 2nd deg polynomial and
its derivative
'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import matplotlib.pyplot as plt

plt.style.use('grayscale')
x = np.arange(-1, 2, 0.1)
c = [1, -1, -1]
d = [2, -1]
y = np.polyval(c, x)
z = np.polyval(d, x)
plt.xlabel('x')
plt.ylabel(r'$y\/\/\/and\/\/\/\frac{dy}{dx}$')
plt.plot(x, y, label=r'$y=x^2 -x -1$')
plt.plot(x, z, label=r'$\frac{dy}{dx},\/\/\/y_{min}\/\/at\/\/x=0.5$')
plt.legend(loc=0)
plt.grid(b=True)
plt.savefig("sgd.png")
plt.show()
plt.close('all')
