#!/usr/bin/env python
# coding: utf-8

import StackGP as sgp
import sys
import numpy as np
import pandas as pd




Z = np.loadtxt(f"datasets/dataset_3.csv", delimiter=",", skiprows=1)
X, y = Z[:, :-1], Z[:, -1]

models=sgp.evolve(X.transpose(), y, generations=20000, ops=sgp.allOps(), elitismRate=10,tourneySize=5,popSize=50)
#models=sgp.activeLearning(X.transpose(), y,100)
print(sgp.printGPModel(models[0]))
print(sgp.fitness(models[0],X.transpose(), y))
fits=[sgp.fitness(i, X.transpose(), y) for i in models]
#print(fits)
best=np.argmin(fits)
print(fits[best],sgp.printGPModel(models[best]))

