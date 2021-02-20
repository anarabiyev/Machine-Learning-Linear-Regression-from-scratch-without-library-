import pandas as pd                         
data = pd.read_csv("train.csv")               
print(data.head())  

X = data['x']; Y = data['y']
X = X.tolist()
Y = Y.tolist()

import matplotlib.pyplot as plt
plt.scatter(X, Y, s = 5)
plt.grid()
plt.xlabel("x values")
plt.ylabel("y values")
plt.show()


#alpha - learning rate
def gradient_descent(X, Y, w, b, alpha):
 
    dl_dw = 0.0
    dl_db = 0.0
    N = len(X)

    for i in range(N):
        dl_dw += -1*X[i] * (Y[i] - (w*X[i] + b))
        dl_db += -1*(Y[i] - (w*X[i] + b))

    w = w - (1/float(N)) * dl_dw * alpha
    b = b - (1/float(N)) * dl_db * alpha

    return w, b


def cost_function (X, Y, w, b):

    N = len(X)
    total_error = 0.0
    for i in range(N):
        total_error += (Y[i] - (w*X[i] - b))**2

    return total_error / (2*float(N))
    


def train(X, Y, w, b, alpha, n_iter):

    for i in range(n_iter):
        w, b = gradient_descent(X, Y, w, b, alpha)

        if i % 400 == 0:
            print ("iteration:", i, "cost: ", cost_function(X, Y, w, b))


    return w, b


def predict(x, w, b):
	return x*w + b


w, b = train(X, Y, 0.0, 0.0, 0.0001, 7000)
x_new = 50.0
y_new = predict(x_new, w, b)
print(y_new)