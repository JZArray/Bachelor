from mlxtend.feature_selection import SequentialFeatureSelector as SFS
from sklearn import datasets
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
from sklearn import tree

class SFS:

    def __init__(self,k:int,X_train:np, X_test:np, y_train:np, y_test:np,name_of_regressor):
        self.k = k
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.best_feature = None
        self.final_score = 0
        self.name = name_of_regressor

    def Forward(self):
        Sub_set_Test = []
        Sub_set_Train = []
        index = 0

        for i in range(self.k):

            for j in range(self.X_train.shape[1]):

                score = 0
                if len(Sub_set_Train) == 0 or len(Sub_set_Test) == 0:
                    train = self.X_train[:, j].reshape(-1, 1)
                    test = self.X_test[:, j].reshape(-1, 1)
                    current_score = self.try_different_method(train,self.y_train,test,self.y_test)


                    if current_score > score:
                        score = current_score
                        index = j
                else:
                    a = np.array(Sub_set_Train)
                    b = np.array(Sub_set_Test)
                    train = np.column_stack((a.T, self.X_train[:, j]))
                    test = np.column_stack((b.T, self.X_test[:, j]))


                    current_score = self.try_different_method(train,self.y_train, test,self.y_test)

                    if current_score > score:
                        score = current_score
                        self.final_score = score
                        index = j

            self.best_feature = self.X_train[:, index]


            Sub_set_Train.append(self.X_train[:, index])
            Sub_set_Test.append(self.X_test[:, index])
            self.X_train = np.delete(self.X_train, index, 1)
            self.X_test = np.delete(self.X_test, index, 1)


        return Sub_set_Train,Sub_set_Test

    def try_different_method(self,x_train,y_train,x_test,y_test):
        self.name.fit(x_train, y_train)
        score = self.name.score(x_test, y_test)
        result = np.around(self.name.predict(x_test))

        return score

#
#
# from sklearn import linear_model
#
# data = pd.read_csv("12月13_Extraction_db100.csv")
#
# a = np.array(data)
# a = np.delete(a,0,1)
#
# x,y = a[:,0:a.shape[1]-1],a[:,a.shape[1]-1]
#
# X_train,X_test,y_train,y_test = train_test_split(x,y,test_size=0.3,random_state=0)
#
# linear_reg = linear_model.LinearRegression()
#
# sfs = SFS(10,X_train,X_test,y_train,y_test,linear_reg)
#
# m,n = sfs.Forward()
# print(m)