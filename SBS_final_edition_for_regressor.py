from sklearn import datasets
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd


class SBS():

    def __init__(self,k:int,X_train:np, X_test:np, y_train:np, y_test:np,name_of_regressor):
        self.k = k
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.final_score = 0
        self.name = name_of_regressor


    def Backward(self):

        index = 0

        current_score = self.try_different_method(self.X_train,self.y_train,self.X_test,self.y_test)

        for i in range(self.k):

            for j in range(self.X_train.shape[1]):
                score = 0
                train = np.delete(self.X_train, j, 1)
                test = np.delete(self.X_test, j, 1)

                score = self.try_different_method(train,self.y_train,test,self.y_test)



                if score > current_score:
                    current_score = score
                    self.final_score = current_score
                    index = j



            self.X_train = np.delete(self.X_train, index, 1)
            self.X_test = np.delete(self.X_test, index, 1)

        return self.X_train,self.X_test

    def try_different_method(self,x_train,y_train,x_test,y_test):
        self.name.fit(x_train, y_train)
        score = self.name.score(x_test, y_test)
        result = np.around(self.name.predict(x_test))

        return score


#
#
#
#
#
# from sklearn import linear_model
#
# data = pd.read_csv("12月15_Extraction_without_Loss.csv")
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
# sbs = SBS(15,X_train,X_test,y_train,y_test,linear_reg)
#
# x,y = sbs.Backward()
# print(x.shape)
