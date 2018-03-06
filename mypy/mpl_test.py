import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

# Come up with x and y
x = np.arange(0,5,0.1)
y = np.sin(x)

# Just print x and y for fun
print(x)
print(y)

# plot the x and y
print(plt.get_backend())
plt.plot(x,y)

# without the line below, the figure won't show
plt.show()