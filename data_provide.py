import numpy as np
import matplotlib.pyplot as plt

img = np.random.normal(size=(28, 28))
plt.imshow(img, cmap="gray")
plt.savefig('./test.jpg')