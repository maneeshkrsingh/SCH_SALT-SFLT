import numpy as np

import matplotlib.pyplot as plt

energy_SFLT = np.load('SFLT_Energy.npy')

plt.plot(energy_SFLT)
plt.xlabel('Time Steps')
plt.ylabel('Energy')
plt.title('Energy vs Time Steps')
plt.show()

