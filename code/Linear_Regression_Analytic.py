from cost_functions import RMSE

class Linear_Regression_Analytic:
    def __init__(self):
        self.W = 0
        self.last_cost= 0
    
    def fit(self, X,y):
        X = np.c_[np.ones((len(X), 1)), X]
        self.W = np.linalg.inv(X.T.dot(X)).dot(X.T).dot(y)
        self.last_cost = RMSE(y,X.dot(self.W)) 
    
    def predict(self, X):
        X  = np.c_[np.ones((len(X), 1)), X]
        return X.dot(self.W)
