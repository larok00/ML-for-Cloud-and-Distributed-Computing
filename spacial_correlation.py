import numpy as np
import matplotlib.pyplot as plt

c = np.load('google-cpu-full.npy')
print(c.shape)
plt.plot(c[0])
plt.show()