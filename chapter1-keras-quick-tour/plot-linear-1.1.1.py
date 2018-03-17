'''
Utility for plotting a linear function
with and without noise

Project: https://github.com/roatienza/dl-keras
Dependency: keras 2.0
Usage: python3 <this file>
'''

import numpy as np
import matplotlib.pyplot as plt

want_noise = False
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
    noise = np.random.uniform(-0.1, 0.1, y.shape)
    yn = y + noise

    plt.ylabel('y and yn')
    plt.plot(x, yn, 's-', label="yn = y + noise")

plt.legend(loc=0)
plt.grid(b=True)
plt.show()
plt.close('all')
