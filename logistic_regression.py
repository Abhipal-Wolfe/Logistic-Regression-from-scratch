import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt

class Logreg:
    def __init__(self, n_features, batchsize=10, lr=0.001, n_iters=1000):
        self.lr = lr
        self.n_iters = n_iters
        self.batchsize = batchsize
        self.weights = np.zeros((n_features, 1))
        self.bias = 0

    def gradient_descent(self, X, y):
        n_samples, n_features = X.shape
        for i in range(self.n_iters):
            linear_model = np.dot(X, self.weights) + self.bias
            y_predicted = self._sigmoid(linear_model)

            dw = (1 / n_samples) * np.dot(X.T, (y_predicted-y))
            db = (1 / n_samples) * np.sum(y_predicted - y)

            self.weights -= self.lr * dw
            self.bias -= self.lr * db

    def stochastic_gradient_descent(self, X, y):
        n_samples, n_features = X.shape

        for _ in range(self.n_iters):
            data = np.concatenate((X, y), axis=1)
            np.random.shuffle(data)
            X = data[:, :-1]
            y = data[:, -1:]

            for start_ix in range(0, n_samples, self.batchsize):
                Xbatch = X[start_ix:start_ix + self.batchsize, :]
                ybatch = y[start_ix:start_ix + self.batchsize]

                linear_model = np.dot(Xbatch, self.weights) + self.bias
                y_predicted = self._sigmoid(linear_model)

                dw = (1 / n_samples) * np.dot(Xbatch.T, (y_predicted - ybatch))
                db = (1 / n_samples) * np.sum(y_predicted - ybatch)

                self.weights -= self.lr * dw
                self.bias -= self.lr * db

    def loss(self,X,y):
        n_samples, n_features = X.shape
        linear_model = np.dot(X, self.weights) + self.bias
        y_pred = self._sigmoid(linear_model)
        loss = -(1 / n_samples) * (np.dot(y.T,np.log(y_pred)) + np.dot((1-y).T , np.log(1 - y_pred)))
        return loss

    def predict(self, X):
        linear_model = np.dot(X, self.weights) + self.bias
        y_predicted = self._sigmoid(linear_model)
        y_predicted_cls = [1 if i > 0.5 else 0 for i in y_predicted]
        return np.asarray(y_predicted_cls)[:, np.newaxis]

    def _sigmoid(self, x):
        return 1/(1+np.exp(-x))

def accuracy(y_true, y_pred):
    accuracy = np.sum(y_true == y_pred)/len(y_true)
    return accuracy

def precision(y_true, y_pred):
    tp = np.sum(y_pred[y_true==1] == y_true[y_true==1])
    fp = np.sum(y_pred[y_true==0] != y_true[y_true==0])
    precision = tp/(tp+fp)
    return precision

def recall(y_true, y_pred):
    tp = np.sum(y_pred[y_true == 1] == y_true[y_true == 1])
    fn = np.sum(y_pred[y_true == 1] != y_true[y_true == 1])
    recall = tp/(tp+fn)
    return recall

def f1score(precision,recall):
    f1score = (2*recall*precision)/(recall+precision)
    return f1score

dataset = pd.read_csv('dataset_LR.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1:].values
X = np.array(X)
y = np.array(y)
data = np.concatenate((X, y), axis=1)
np.random.shuffle(data)
X = data[:, :-1]
y = data[:, -1:]
avg_accuracy = 0
cost = 0
avg_cost = 0
accurate_test = np.zeros((20,1))
accurate_train = np.zeros((20,1))
loss_train = np.zeros((20,1))
loss_test = np.zeros((20,1))
accurate_test_split = np.zeros((10,1))
accurate_train_split = np.zeros((10,1))
loss_test_split = np.zeros((10,1))
loss_train_split = np.zeros((10,1))
pre_split = np.zeros((10,1))
rec_split = np.zeros((10,1))
f1_split = np.zeros((10,1))
weight_split = np.zeros((10,4))
for i in range(10):
    splits = np.array([random.randint(0,10) for n in range(X.shape[0])])
    foldsx = [X[splits == f] for f in range(0,10)]
    foldsy = [y[splits == f] for f in range(0,10)]
    foldsx = list(foldsx)
    foldsy = list(foldsy)
    foldsx = np.array(foldsx)
    foldsy = np.array(foldsy)
    x_train = np.concatenate((foldsx[0], foldsx[1], foldsx[2], foldsx[3], foldsx[4], foldsx[5],foldsx[6]))
    y_train = np.concatenate((foldsy[0], foldsy[1], foldsy[2], foldsy[3], foldsy[4], foldsy[5],foldsy[6]))
    x_test = np.concatenate((foldsx[7], foldsx[8], foldsx[9]))
    y_test = np.concatenate((foldsy[7], foldsy[8], foldsy[9]))

    regressor = Logreg(lr=0.00001, n_iters=50, n_features=x_train.shape[1])
    for j in range(20):
        regressor.stochastic_gradient_descent(x_train,y_train)
        predictions_test = regressor.predict(x_test)
        predictions_train = regressor.predict(x_train)
        accurate_test[j] = accuracy(y_test,predictions_test)
        accurate_train[j] = accuracy(y_train, predictions_train)
        loss_train[j] = regressor.loss(x_train,y_train)
        loss_test[j] = regressor.loss(x_test,y_test)

    loss_train_split[i] = loss_train[19]
    loss_test_split[i] = loss_test[19]
    accurate_train_split[i] = accurate_train[19]
    accurate_test_split[i] = accurate_test[19]
    pre = precision(y_test, predictions_test)
    rec = recall(y_test, predictions_test)
    f1 = f1score(pre, rec)

    weight_split[i] = regressor.weights.reshape(4,)

    pre_split[i] = pre
    rec_split[i] = rec
    f1_split[i] = f1
    k = [50*i for i in range(20)]
    if i == 5:
        fig, axs = plt.subplots(2, 2)
        axs[0,0].plot(k, loss_train, color="orange")
        axs[0,0].set_xlabel('iterations')
        axs[0,0].set_ylabel('training loss')
        axs[0,0].set_title('training loss vs iteration')

        axs[0,1].plot(k, loss_test, color="green")
        axs[0,1].set_xlabel('iterations')
        axs[0,1].set_ylabel('testing loss')
        axs[0,1].set_title('testing loss vs iteration')

        axs[1,0].plot(k, accurate_train, color="blue")
        axs[1,0].set_xlabel('iterations')
        axs[1,0].set_ylabel('training accuracy')
        axs[1,0].set_title('training accuracy vs iteration')

        axs[1,1].plot(k, accurate_test, color="red")
        axs[1,1].set_xlabel('iterations')
        axs[1,1].set_ylabel('testing accuracy')
        axs[1,1].set_title('testing accuracy vs iteration')

        plt.show()
print("loss_train_split:", np.mean(loss_train_split))
print("loss_test_split:", np.mean(loss_test_split))
print("accurate_train_split:", np.mean(accurate_train_split))
print("accurate_test_split:", np.mean(accurate_test_split))
print("pre_split:", np.mean(pre_split))
print("rec_split:", np.mean(rec_split))
print("f1_split:", np.mean(f1_split))
print("weights:",np.mean(weight_split, axis=0))
