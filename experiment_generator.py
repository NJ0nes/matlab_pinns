#!/usr/bin/python3
import math
import numpy as np
import itertools

# ====== Configuration ==========
# Defaults written next to each parameter
viscosity  = np.linspace(0.01 / math.pi, 0.01 / math.pi, 1) # 0.01 / pi
num_epochs = np.linspace(1000, 1000, 1) # 3000
mini_batch_size = np.linspace(1000, 1000, 1) # 1000
weightF = np.linspace(0.5, 0.5, 1) # 0.5
weightU = np.linspace(0.5, 0.5, 1) # 0.5
num_bc_points = np.linspace(200, 200, 1) # 100
num_ic_points = np.linspace(300, 300, 1) # 200
num_colloc_points = np.linspace(10000, 10000, 1) # 10000
num_layers = np.linspace(9, 16, (16 - 9) + 1) # 9
num_neurons = np.linspace(8, 32, (32 - 8) // 2 + 1) # 20
learn_rate = np.linspace(0.01, 0.01, 1) # 0.01, 0.05 was optimal for swish
decay_rate = np.linspace(0.005, 0.005, 1) # 0.005, 0.03 was optimal for swish
label = "Tanh Structure"
activation = ["actTanh"] # actTanh
collocation = ["sobolpoints"] #sobolpoints

# ====== Code ===================

for vi, ne, mi, wf, wu, nb, ni, nc, nl, nn, lr, dr, ac, co in itertools.product( \
        viscosity, num_epochs, mini_batch_size, weightF, weightU, num_bc_points, \
        num_ic_points, num_colloc_points, num_layers, num_neurons, learn_rate,   \
        decay_rate, activation, collocation):

    parameters = [
        "experiment_list(l).label = \"" + label + "\"",
        "experiment_list(l).viscosity = " + str(vi),
        "experiment_list(l).numEpochs = " + str(int(ne)),
        "experiment_list(l).miniBatchSize = " + str(int(mi)),
        "experiment_list(l).weightF = " + str(wf),
        "experiment_list(l).weightU = " + str(wu),
        "experiment_list(l).numBoundaryConditionPoints = " + str(int(nb)),
        "experiment_list(l).numInitialConditionPoints = " + str(int(ni)),
        "experiment_list(l).numInternalCollocationPoints = " + str(int(nc)),
        "experiment_list(l).numLayers = " + str(int(nl)),
        "experiment_list(l).numNeurons = " + str(int(nn)),
        "experiment_list(l).initialLearnRate = " + str(lr),
        "experiment_list(l).decayRate = " + str(dr),
        "experiment_list(l).activationFn = @" + ac,
        "experiment_list(l).activationName = \"" + ac + "\"",
        "experiment_list(l).collocationFn = @" + co,
        "experiment_list(l).collocationName = \"" + co + "\""
    ]

    print("l = length(experiment_list) + 1; " + "; ".join(parameters) + "; ")
