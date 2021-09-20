'''
    Based on the demo at https://pennylane.ai/qml/demos/tutorial_state_preparation.html
'''

import pennylane as qml
import numpy as np
from pennylane import measure
import torch
from torch import tensor
from torch.autograd import Variable

np.random.seed(42)

targetState = [[0,0,1,1], [0,1,0,1], [1,0,1,0], [1,1,0,0]]

# Dimension of each state
startingQubitsDim = 4

# Number of states to be mapped
startingQubitsNum = 4

# Generating the random starting (normalized) states
startingQubitParams = []
for i in range(startingQubitsNum):
    arr = []
    for k in range(2 ** startingQubitsDim):
        arr.append(np.random.rand())
    arr = arr/np.sqrt(np.dot(arr, arr))
    startingQubitParams.append(arr)

# number of qubits in the circuit
nr_qubits = startingQubitsDim + 1

# number of layers in the circuit
nr_layers = 5

# randomly initialize parameters from a normal distribution. There are four parameters for each line
params = np.random.normal(0, np.pi, (nr_layers, 3*nr_qubits))
params = Variable(torch.tensor(params), requires_grad=True)

# a layer of the circuit ansatz
def layer(params, i):
    for k in range(nr_qubits):
        qml.RX(params[i, 3*k], wires=[k])
        qml.RZ(params[i, 3*k+1], wires=[k])
        qml.IsingXX(params[i, 3*k+2], wires = [k, (k+1) % nr_qubits])

# faster than the default.qubit device
dev = qml.device('qulacs.simulator', wires=nr_qubits)

# use PyTorch for the training
@qml.qnode(dev, interface="torch")
def circuit(params, num): # TODO: modify this to use arbitrary starting states
    # initialize the circuit using the random starting state numbered j
    # the ancilla qubit is initialised in the state 0
    arr = [1]
    for i in range(int(2 ** (nr_qubits - startingQubitsDim)) - 1):
        arr.append(0)
    startingState = np.kron(startingQubitParams[num], arr)
    qml.QubitStateVector(startingState, wires=[*range(nr_qubits)])
    # repeatedly apply each layer in the circuit
    for j in range(nr_layers):
        layer(params, j)
    return qml.expval(qml.PauliZ(0)), qml.expval(qml.PauliZ(1)), qml.expval(qml.PauliZ(2)), qml.expval(qml.PauliZ(3)) # TODO: modify this to use startingQubitsDim

# cost function computes the l2 norm of the costs from each of the four states
def cost_fn(params):
    cost = 0
    for j in range(startingQubitsNum):
        measureRes = circuit(params, j)
        x = 0
        for k in range(startingQubitsDim):
            x += torch.abs(measureRes[k] - (targetState[j])[k])
        cost += x ** 2
    return torch.sqrt(cost)

# set up the optimizer
opt = torch.optim.Adam([params], lr=1)
# the learning rate drops by 0.95 after every run
scheduler = torch.optim.lr_scheduler.ExponentialLR(opt, gamma=0.95)

# number of steps in the optimization routine
steps = 100

# the final stage of optimization isn't always the best, so we keep track of
# the best parameters along the way
best_cost = cost_fn(params)
best_params = np.zeros((nr_layers, 3*nr_qubits))

print("Cost after 0 steps is {:.4f}".format(cost_fn(params)))

# optimization begins
for n in range(steps):
    opt.zero_grad()
    loss = cost_fn(params)
    loss.backward()
    opt.step()
    scheduler.step()

    # keeps track of best parameters
    if loss < best_cost:
        best_cost = loss
        best_params = params

    # Keep track of progress every 10 steps
    if n % 10 == 9 or n == steps - 1:
        print("Cost after {} steps is {:.4f}".format(n + 1, loss))

# calculate the Bloch vector of the output state
for i in range(startingQubitsNum):
    print("Target state =", targetState[i])
    print("Obtained state=", circuit(best_params, i).tolist())
