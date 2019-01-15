from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn import datasets
import numpy as np
import random
import pandas as pd

class RGSS():

    def __init__(self,number_of_feature:int,iteration:int,count:int,X_train:np, X_test:np, y_train:np, y_test:np,name_of_regressor):
        self.iteration = iteration
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.count = count
        self.number_of_feature = number_of_feature
        self.resultList = []
        self.name = name_of_regressor

    def RGSS_SFS(self):

        Subset_Train = None
        Subset_Test = None
        score = 0
        m = self.number_of_feature - self.count

        for i in range(self.iteration):

            current_score,Train,Test = self.SFS(m)


            if current_score > score :

                score = current_score
                Subset_Train = Train
                Subset_Test = Test

        return Subset_Train,Subset_Test

    def RGSS_SBS(self):

        Subset_Train = None
        Subset_Test = None
        score = 0
        m = self.count - self.number_of_feature

        for i in range(self.iteration):

            current_score,Train,Test = self.SBS(m)

            if current_score > score:

                score = current_score
                Subset_Train = Train
                Subset_Test = Test

        return Subset_Train,Subset_Test

    def Subset_Generation(self):

        B = self.X_train.shape[1]
        self.resultList = random.sample(range(0, B), self.count)

        Sub_Set_Train = self.X_train[:,self.resultList]
        Sub_set_Test = self.X_test[:,self.resultList]

        return Sub_Set_Train,Sub_set_Test

    def delete_present_feature(self):

        X_train = np.delete(self.X_train,self.resultList, 1)
        X_test = np.delete(self.X_test, self.resultList, 1)

        return X_train,X_test

    def SFS(self,m):

        score = 0
        Sub_set_Train,Sub_set_Test = self.Subset_Generation()

        X_Train,X_Test = self.delete_present_feature()


        for i in range(m):
            index = 0
            for j in range(X_Train.shape[1]):

                train = np.column_stack((Sub_set_Train, X_Train[:, j]))
                test = np.column_stack((Sub_set_Test, X_Test[:, j]))

                current_score = self.try_different_method(train, self.y_train, test, self.y_test)


                if current_score > score:
                    score = current_score
                    index = j


            Sub_set_Train = np.column_stack((Sub_set_Train, X_Train[:, index]))

            Sub_set_Test = np.column_stack((Sub_set_Test, X_Test[:, index]))
            X_Train = np.delete(X_Train, index, 1)
            X_test = np.delete(X_Test, index, 1)


        return score,Sub_set_Train,Sub_set_Test

    def SBS(self,m):

        X_train,X_test= self.Subset_Generation()


        current_score = self.try_different_method(self.X_train, self.y_train, self.X_test, self.y_test)
        #index = 0
        for i in range(m):
            index = 0
            for j in range(X_train.shape[1]):
                score = 0

                train = np.delete(X_train, j, 1)
                test = np.delete(X_test, j, 1)

                score = self.try_different_method(train, self.y_train, test, self.y_test)

                if score > current_score:
                    current_score = score
                    index = j


            X_train = np.delete(X_train, index, 1)
            X_test = np.delete(X_test, index, 1)

        return  current_score,X_train, X_test



    def try_different_method(self,x_train,y_train,x_test,y_test):
        self.name.fit(x_train, y_train)
        score = self.name.score(x_test, y_test)
        result = np.around(self.name.predict(x_test))

        return score



# from sklearn import tree
# from sklearn import linear_model
# from sklearn import svm
# linear_reg = linear_model.LinearRegression()
# tree_reg = tree.DecisionTreeRegressor()
# #svr = svm.SVR()
#
# data = pd.read_csv("12月13_Extraction_db100.csv")
#
# a = np.array(data)
# a = np.delete(a,0,1)
# x,y = a[:,0:48],a[:,48]
# X_train,X_test,y_train,y_test = train_test_split(x,y,test_size=0.3,random_state=0)
#
# Test = RGSS(15,1,40,X_train,X_test,y_train,y_test,linear_reg)
#
# #Test.RGSS_SFS()
#
# x , y = Test.RGSS_SBS()
#
#
#
