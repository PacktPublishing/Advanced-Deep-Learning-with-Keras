'''Utility for plotting a linear function
with and without noise
'''

import numpy as np
import matplotlib.pyplot as plt

want_noise = True
# grayscale plot, comment if color is wanted
plt.style.use('grayscale')

# generate data bet -1,1 interval of 0.2
x = np.arange(-1,1,0.2)
y = 2*x + 3
plt.xlabel('x')
plt.ylabel('y=f(x)')
plt.plot(x, y, 'o-', label="y")

if want_noise:
    # generate data with uniform distribution
    noise = np.random.uniform(-0.2, 0.2, x.shape)
    xn = x + noise

    plt.ylabel('y=f(x)')
    plt.plot(xn, y, 's-', label="y with noised x")

plt.legend(loc=0)
plt.grid(b=True)
plt.savefig("linear_regression.png")
plt.show()
plt.close('all')
