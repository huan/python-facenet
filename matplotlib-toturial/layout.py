""" layout """
import matplotlib.pyplot as plt
import numpy as np

x = np.linspace(0, 2*np.pi, 400)
y = np.sin(x**2)
f, axes = plt.subplots(2, 2, sharex=True)
print(plt.figure)
axes[0, 0].plot(x, y)
axes[1, 1].set_title('Sharing Y axis')
axes[1, 0].scatter(x, y)

plt.show()
