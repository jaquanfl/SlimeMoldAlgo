import numpy as np
from matplotlib import pyplot as plt
import random
import math	
import copy

# Population initialization function
def initial(SMApop, dim, ub, lb):
	X = np.zeros([SMApop, dim])
	for i in range(SMApop):
		for j in range(dim):
			X[i, j] = random.random() * (ub[j] - lb[j]) + lb[j]
	return X, lb, ub
# Boundary check function
def BorderCheck(X, ub, lb, SMApop, dim):
	for i in range(SMApop):
		for j in range(dim):
			if X[i, j] > ub[j]:
				X[i, j] = ub[j]
			elif X[i, j] < lb[j]:
				X[i, j] = lb[j]
	return X
# Calculate fitness function
def CaculateFitness(X, fun,nn, Xdraw,Ydraw,Zdraw):
	SMApop = X.shape[0]
	fitness = np.zeros([SMApop, 1])
	for i in range(SMApop):
		fitness[i],nn,Xdraw,Ydraw,Zdraw = fun(X[i, :],nn,Xdraw,Ydraw,Zdraw)
	return fitness,nn,Xdraw,Ydraw,Zdraw
# Minimum fitness sorting function
def SortFitness(Fit):
	fitness = np.sort(Fit, axis=0)
	index = np.argsort(Fit, axis=0)
	return fitness, index
# Sort locations according to fitness
def SortPosition(X, index):
	Xnew = np.zeros(X.shape)
	for i in range(X.shape[0]):
		Xnew[i, :] = X[index[i], :]
	return Xnew
# SMA Main function
def SMA(low,up,SMApop,dim,SMALoop,nn,Xdraw,Ydraw,Zdraw,fun):
	z = 0.03
	lb = low * np.ones([dim, 1])
	ub = up * np.ones([dim, 1])
	X, lb, ub = initial(SMApop, dim, ub, lb)
	fitness,nn,Xdraw,Ydraw,Zdraw = CaculateFitness(X, fun,nn,Xdraw,Ydraw,Zdraw)
	fitness, sortIndex = SortFitness(fitness)
	X = SortPosition(X, sortIndex)
	GbestScore = copy.copy(fitness[0])
	GbestPositon = copy.copy(X[0, :])
	Curve =[]
	AvgCurve = []
	iteration=[]
	agent_history = []
	for f in range(fitness.shape[0]):
		Curve.append(float(fitness[-(f+1),0]))
		AvgCurve.append(float(np.mean(fitness)))
		iteration.append(f+1)		
	W = np.zeros([SMApop, dim])
	for t in range(1,SMALoop):
		worstFitness = fitness[-1]
		bestFitness = fitness[0]
		S = bestFitness - worstFitness + 10E-8
		for i in range(SMApop):
			if i < SMApop / 2:
				W[i, :] = 1 + np.random.random([1, dim]) * np.log10((bestFitness - fitness[i]) / (S) + 1)
			else:
				W[i, :] = 1 - np.random.random([1, dim]) * np.log10((bestFitness - fitness[i]) / (S) + 1)
		tt = -(t / SMALoop) + 1
		if tt != -1 and tt != 1:
			a = math.atanh(tt)
		else:
			a = 1
		b = 1 - t / SMALoop
		for i in range(SMApop):
			if np.random.random() < z:
				X[i, :] = (ub.T - lb.T) * np.random.random([1, dim]) + lb.T
			else:
				p = np.tanh(abs(fitness[i] - GbestScore))
				vb = 2 * a * np.random.random([1, dim]) - a
				vc = 2 * b * np.random.random([1, dim]) - b
				for j in range(dim):
					r = np.random.random()
					A = np.random.randint(SMApop)
					B = np.random.randint(SMApop)
					if r < p:
						X[i, j] = GbestPositon[j] + vb[0, j] * (W[i, j] * X[A, j] - X[B, j])
					else:
						X[i, j] = vc[0, j] * X[i, j]
		X = BorderCheck(X, ub, lb, SMApop, dim)
		fitness,nn,Xdraw,Ydraw,Zdraw = CaculateFitness(X, fun,nn,Xdraw,Ydraw,Zdraw)
		fitness, sortIndex = SortFitness(fitness)
		X = SortPosition(X, sortIndex)
		if (fitness[0] <= GbestScore):
			GbestScore = copy.copy(fitness[0])
			GbestPositon = copy.copy(X[0, :])
		Curve.append(float(GbestScore))
		AvgCurve.append(float(np.mean(fitness)))
		iteration.append(nn)
		agent_history.append(X.copy())
	return GbestScore, GbestPositon,iteration,Curve,nn,Xdraw,Ydraw,Zdraw, AvgCurve, X

# Rastrigin Function
def rastrigin(position, nn, Xdraw, Ydraw, Zdraw):
    A = 10
    fitness = A * len(position) + sum([x**2 - A * np.cos(2 * np.pi * x) for x in position])
    nn += 1
    Xdraw.append(position[0])
    Ydraw.append(position[1])
    Zdraw.append(fitness)
    return fitness, nn, Xdraw, Ydraw, Zdraw

# Griewank Function
def griewank(position, nn, Xdraw, Ydraw, Zdraw):
    sum_sq = sum([x**2 for x in position]) / 4000
    prod_cos = np.prod([np.cos(x / np.sqrt(i + 1)) for i, x in enumerate(position)])
    fitness = sum_sq - prod_cos + 1
    nn += 1
    Xdraw.append(position[0])
    Ydraw.append(position[1])
    Zdraw.append(fitness)
    return fitness, nn, Xdraw, Ydraw, Zdraw

# Sphere Function
def sphere(position, nn, Xdraw, Ydraw, Zdraw):
    fitness = sum([x**2 for x in position])
    nn += 1
    Xdraw.append(position[0])
    Ydraw.append(position[1])
    Zdraw.append(fitness)
    return fitness, nn, Xdraw, Ydraw, Zdraw

# Rosenbrock Function
def rosenbrock(position, nn, Xdraw, Ydraw, Zdraw):
    fitness = sum([100 * (position[i+1] - position[i]**2)**2 + (position[i] - 1)**2 for i in range(len(position)-1)])
    nn += 1
    Xdraw.append(position[0])
    Ydraw.append(position[1])
    Zdraw.append(fitness)
    return fitness, nn, Xdraw, Ydraw, Zdraw

# Set general parameters
low=-0.8 # Lower bound for the search space
up=0.8 # Upper bound for the search space
dim = 2 # Dimension of the problem (2D for visualization)
nn=0 # Iteration counter for the function evaluations
fun = rastrigin  # Choose your function: rastrigin, griewank, sphere, rosenbrock 
# Defines the variables used to draw the graph
Xdraw=[]
Ydraw=[]
Zdraw=[]
# SMA Special parameters
SMApop=1000 # Number of agents in the population
SMALoop = 50 # Number of iterations for the algorithm
# Call SMA
GbestScore,GbestPositon,iteration,Curve,nn,Xdraw,Ydraw,Zdraw, AvgCurve, FinalPopulation = SMA(low,up,SMApop,dim,SMALoop,nn,Xdraw,Ydraw,Zdraw,fun)

fig, axs = plt.subplots(2, 2, figsize=(12, 10))
# Convergence Curve (Best Fitness)
axs[0, 0].plot(iteration, Curve, color='green', label='Best Fitness', linewidth=2)
axs[0, 0].set_title('Best Fitness Over Iterations (Linear)')
axs[0, 0].set_xlabel('Iteration')
axs[0, 0].set_ylabel('Fitness')
axs[0, 0].set_xlim([0, 1000])
# axs[0, 0].set_ylim([0, 10])
axs[0, 0].grid(True)
axs[0, 0].legend()

# Average Fitness Curve
axs[0, 1].plot(iteration, AvgCurve, color='blue', label='Average Fitness', linewidth=2)
axs[0, 1].set_title('Average Fitness Over Iterations')
axs[0, 1].set_xlabel('Iteration')
axs[0, 1].set_ylabel('Fitness')
axs[0, 1].grid(True)
axs[0, 1].legend()

# Best Fitness Over Iterations
axs[1, 0].semilogx(iteration, Curve, color='red', label='Best Fitness', linewidth=2)
axs[1, 0].set_title('Best Fitness Over Iterations (Logarithmic)')
axs[1, 0].set_xlabel('Iteration')
axs[1, 0].set_ylabel('Fitness')
axs[1, 0].grid(True)
axs[1, 0].legend()	

# Final Agent Distribution
axs[1, 1].scatter(FinalPopulation[:, 0], FinalPopulation[:, 1], color='red', label='Final Agents')
axs[1, 1].scatter(GbestPositon[0], GbestPositon[1], color='black', marker='x', s=100, label='Best Agent')
axs[1, 1].set_title('Final Agent Positions in Search Space')
axs[1, 1].set_xlabel('X')
axs[1, 1].set_ylabel('Y')
axs[1, 1].grid(True)
axs[1, 1].legend()

plt.tight_layout()
plt.show()
