#!/usr/bin/env python3

import matplotlib.pyplot as plt
import numpy as np

with open('pca.data') as file:
	x = np.loadtxt(file)
with open('pca_v.data') as file:
	v = np.loadtxt(file).transpose()

print(x.shape)
print(v.shape)
data = x @ v

for i in range(25):
	plt.subplot(5,5,(i%25)+1)
	plt.xticks([])
	plt.yticks([])
	plt.grid(False)
	plt.imshow(data[i].reshape(48,48,1))
plt.show()
