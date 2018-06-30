'''Utility for plotting a polynomial with 2 minima 
and its derivative
'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import matplotlib.pyplot as plt

plt.style.use('grayscale')
x = np.arange(-2.5, 2.5, 0.1)
c = [1, -0.2, -5, 0, 4]
d = [4, -0.6, -10, 0]
y = np.polyval(c, x)
z = np.polyval(d, x)
plt.xlabel('x')
plt.ylabel(r'$y\/\/\/and\/\/\/\frac{dy}{dx}$')
plt.plot(x, y, label=r'$y=x^4 -0.2x^3 -5x^2 +4$')
plt.plot(x, z, label=r'$\frac{dy}{dx},\/\/\/y_{min}\/\/\/at\/\/\/x=-1.51\/\/\/&\/\/\/1.66$')
plt.legend(loc=0)
plt.grid(b=True)
plt.savefig("sgd-2-min.png")
plt.show()
plt.close('all')
