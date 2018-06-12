'''
Utility for plotting 2 parallel distributions

'''

import numpy as np
import matplotlib.pyplot as plt

want_noise = True
# grayscale plot, comment if color is wanted
plt.style.use('grayscale')

x = np.zeros((1000,))
y = np.random.uniform(0, 1, x.shape)
plt.xlabel('x')
plt.ylabel('y')
plt.plot(x, y, 'o-', label=r'$p_{data}$')

x = 0.5 * np.ones((1000,))
y = np.random.uniform(0, 1, x.shape)
plt.xlabel('x')
plt.ylabel('y')
plt.plot(x, y, 's-', label=r'$p_{g}$')

plt.legend(loc=0)
plt.grid(b=True)
plt.savefig("divergence.png")
plt.show()
plt.close('all')
