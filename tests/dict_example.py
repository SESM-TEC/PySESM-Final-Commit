import math
from tqdm import tqdm
import torch
import matplotlib.pyplot as plt
import numpy as np
from PySESM.models.DictLayer import DictLayer
from PySESM.models.Wrapper import Wrapper
from PySESM.models.ISTA import generate_h
from PySESM.base_functions.Function import GaussianFunctions

from mpl_toolkits.mplot3d import Axes3D
from sklearn.datasets import make_spd_matrix
from scipy.stats import multivariate_normal

N_points = 50
xl = -2
xr = 2
x = np.linspace(xl, xr, N_points)
xx, yy = np.meshgrid(x, x)
X = np.column_stack([xx.ravel(), yy.ravel()])

mu1 = torch.tensor([1, 1])
mu2 = torch.tensor([1, -1])
mu3 = torch.tensor([-1, -1])

sigma1 = 0.3 * torch.eye(2)
sigma2 = 0.15 * torch.eye(2)
sigma3 = 0.1 * torch.eye(2)

pdf1 = torch.tensor(multivariate_normal.pdf(X, mu1, sigma1)/multivariate_normal.pdf(mu1,mu1,sigma1))
pdf2 = torch.tensor(multivariate_normal.pdf(X, mu2, sigma2)/multivariate_normal.pdf(mu2,mu2,sigma2))
pdf3 = torch.tensor(multivariate_normal.pdf(X, mu3, sigma3)/multivariate_normal.pdf(mu3,mu3,sigma3))


zz = (pdf1 + pdf2 + pdf3)
zz = zz.reshape(xx.shape)

fig = plt.figure(figsize=(10, 6))
ax = fig.add_subplot(111, projection='3d', navigate=True)
ax.plot_surface(xx, yy, zz, cmap='plasma')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('PDF')
ax.set_title('Target Gaussian Distributions')


x_values = xx.ravel()
y_values = yy.ravel()
z_values = zz.ravel()

n_samples = 1000
n_features = 2
l_functions =  10
total_points = len(x_values)

min_separation = 1

selected_indexes = []

while len(selected_indexes) < n_samples:

    random_index = np.random.randint(total_points)

    if all(abs(random_index - existing_index) >= min_separation for existing_index in selected_indexes):
        selected_indexes.append(random_index)

sampled_indices = selected_indexes

sampled_x = torch.tensor(x_values[sampled_indices], dtype=torch.float32)
sampled_y = torch.tensor(y_values[sampled_indices], dtype=torch.float32)

X = torch.stack((sampled_x, sampled_y), dim=1)
y = torch.tensor(z_values[sampled_indices], dtype=torch.float32)

h = generate_h(l_functions)
gaussian_function = GaussianFunctions(n_features= n_features, n_functions = l_functions) 
layer = DictLayer(n_features=n_features, n_samples =n_samples,n_functions = l_functions, psi=gaussian_function.gaussian)
model = Wrapper(model=layer)
model.fit(X, y, epochs=2000, h=h, alpha=0.01)

x_tensor = torch.tensor(x_values)
y_tensor = torch.tensor(y_values)
XY = torch.cat((x_tensor.unsqueeze(1), y_tensor.unsqueeze(1)), dim=1)
Z = model.predict(XY,h)

fig = plt.figure(figsize=(12, 6))

ax1 = fig.add_subplot(221, projection='3d')
ax1.scatter(x_values, y_values, z_values,c=z_values)
ax1.set_xlabel('X')
ax1.set_ylabel('Y')
ax1.set_zlabel('Z')
ax1.set_title('Original Function')


ax2 = fig.add_subplot(222, projection='3d')
ax2.scatter(x_values, y_values, Z.detach(), c=Z.detach())
ax2.set_xlabel('X')
ax2.set_ylabel('Y')
ax2.set_zlabel('Z2')
ax2.set_title('Surrogate Model')

ax2 = fig.add_subplot(223, projection='3d')
ax2.scatter(sampled_x, sampled_y, y, c=y)
ax2.set_xlabel('X')
ax2.set_ylabel('Y')
ax2.set_zlabel('Z')
ax2.set_title('Samples')

ax2 = fig.add_subplot(224, projection='3d')
ax2.scatter(sampled_x, sampled_y, model.predict(X,h).detach(), c=model.predict(X,h).detach())
ax2.set_xlabel('X')
ax2.set_ylabel('Y')
ax2.set_zlabel('Z')
ax2.set_title('Samples Predicted')

plt.subplots_adjust(wspace=0.4)

# Show the plot
plt.show()
