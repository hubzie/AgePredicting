#!/usr/bin/env python3

import matplotlib.pyplot as plt
import numpy as np

plt.rcParams['image.cmap'] = 'gray'

with open('pca.data') as file:
	x = np.loadtxt(file)
with open('pca_v.data') as file:
	v = np.loadtxt(file).transpose()

print(x.shape)
print(v.shape)
data = x @ v

W, H = 10, 10

fig = plt.figure(figsize=(8,8))

for i in range(W*H):
	plt.subplot(W,H,(i%(W*H))+1)
	plt.xticks([])
	plt.yticks([])
	plt.grid(False)
	plt.imshow(data[i].reshape(48,48,1))

fig.tight_layout(h_pad=0, w_pad=0)
plt.show()
