import numpy as np
import matplotlib.pyplot as plt
x = np.linspace(0, 2*np.pi, 400)
y = np.sin(x**2)
f, axs = plt.subplots(2, 2, sharex=True)
print(plt.figure)
ax01.plot(x, y)
ax01.set_title('Sharing Y axis')
ax02.scatter(x, y)

plt.show()
