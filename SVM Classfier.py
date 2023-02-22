from sklearn import datasets
import numpy as np
import matplotlib.pyplot as plt
from sklearn.base import BaseEstimator

class LinearGDC(BaseEstimator):
    def __init__(self, C=1, eta0=1, eta_d=10000, n_epochs=1000, random_state=None):
        self.C = C
        self.eta0 = eta0
        self.n_epochs = n_epochs
        self.random_state = random_state
        self.eta_d = eta_d

    def eta(self, epoch):
        return self.eta0 / (epoch + self.eta_d)

    def fit(self, X, y):
        # Random initialization
        if self.random_state:
            np.random.seed(self.random_state)

            w= np.random.randn(X.shape[1],1)
            b= 0
            y_truelabel = y*2-1
            xy = X*y_truelabel

        
        self.Js=[]

        # Training
        for epoch in range(self.n_epochs):
            sv_id = (1 - (xy.dot(w) + y_truelabel * b) >= 0).ravel()
            X_flabel = xy[sv_id]
            y_fLabel = y_truelabel[sv_id]
            J = 0.1 * np.sum(w * w) + self.C * (np.sum(1 - (X_flabel.dot(w))) - b * np.sum(y_fLabel))
            self.Js.append(J)

            w_gradient_vector = 2 * 0.1 * w - np.sum(X_flabel, axis=0).reshape(-1, 1)
            b_derivative = -C * np.sum(y_fLabel, axis=0).reshape(-1, +1)




            w = w - self.eta(epoch) * w_gradient_vector
            b = b - self.eta(epoch) * b_derivative


        self.intercept_ = np.array([b])
        self.coef_ = np.array([w])

        #plt.plot(self.Js)
        #plt.show()

        return self


    def decision_function(self, X):
        return X.dot(self.coef_[0]) + self.intercept_[0]

    # output the predicted class
    def predict(self, X):
        return (self.decision_function(X) >= 0).astype(np.float64)

# We will use iris dataset in this example
iris = datasets.load_iris()
X = iris["data"][:, (2, 3)] # petal length, petal width
y = (iris["target"] == 2).astype(np.float64).reshape(-1, 1) # Iris virginica


C=2
svm_clf = LinearGDC(C=C, eta0 = 10, eta_d = 1000, n_epochs=60000, random_state=2)
svm_clf.fit(X, y)
print(svm_clf.predict(X))
plt.plot(range(svm_clf.n_epochs),svm_clf.Js)
plt.axis([0,svm_clf.n_epochs,0,100])
plt.show()
