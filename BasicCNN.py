import math
import random

import matplotlib.pyplot
import numpy
import numpy as np
import pandas
from sklearn.neural_network import MLPRegressor
a = random.randint(1,100)
b = random.randint(1,100)
# inputSincos = [(x*x*(a*math.sin(math.radians(x))))-((b*math.cos(math.radians(x))*x)) for x in range(0,360)]
inputSincos = [x**2 for x in range(0,360)]

# Training .....
A = []
B = []
itter = range(0,360)
for i in range(0,360):
    A.append(a)
    B.append(b)

df = pandas.DataFrame()
df["A"] = A
df["B"] = B
df["Itter"] = itter
df["OUT"] = inputSincos
X = df.iloc[:,0:3]
y = np.ravel(inputSincos)
print(X.shape)
print(y.shape)
print(X.shape[0] == y.shape[0])
# NN = MLPRegressor(solver='adam', alpha=1e-5,hidden_layer_sizes=(50,100,100,50),activation="relu",learning_rate="adaptive",max_iter=2)
# NN.fit(X, y)
#
# # matplotlib.pyplot.plot(np.array(range(0,360)),np.array(inputSincos))
# # matplotlib.pyplot.show()
#
# # Testing
# outVal = NN.predict(X)
#
# matplotlib.pyplot.plot(np.array(range(0,360)),np.array(inputSincos),color="green")
# matplotlib.pyplot.plot(numpy.array(range(0,360)),np.array(outVal),color="purple")
# # matplotlib.pyplot.show(block=False)
# matplotlib.pyplot.pause(0.5)

for i in range(1,200):
    NN = MLPRegressor(solver='adam', alpha=1e-5, hidden_layer_sizes=(50, 100, 100, 50), activation="relu",
                      learning_rate="adaptive", max_iter=i)
    NN.fit(X, y)

    # matplotlib.pyplot.plot(np.array(range(0,360)),np.array(inputSincos))
    # matplotlib.pyplot.show()

    # Testing
    outVal = NN.predict(X)

    matplotlib.pyplot.plot(np.array(range(0, 360)), np.array(inputSincos), color="green")
    matplotlib.pyplot.plot(numpy.array(range(0, 360)), np.array(outVal), color="purple")
    # matplotlib.pyplot.show(block=False)
    matplotlib.pyplot.draw()
    matplotlib.pyplot.pause(0.0000001)
    matplotlib.pyplot.clf()
