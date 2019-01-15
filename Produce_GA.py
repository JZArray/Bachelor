import random
import Produce_Gen
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

class GA:

    #构造，gen_len为基因长度，size为种群大小，mate_p和mutate_p分别是交配概率和变异概率
    def __init__(self,gen_len,size,X_train:np,X_test:np,y_train:np,y_test:np,name_of_regressor,iteration,mate_p=0.1,mutate_p=0.1):
        # if not isinstance(gen_len,int) or gen_len<=0:
        #     raise TypeError('<gen_len> should be a positive integer')
        # if not isinstance(size,int) or size<=0:
        #     raise TypeError('<size> should be a positive integer')
        # if not isinstance(mate_p,float) or mate_p<0 or mate_p>1:
        #     raise TypeError('<mate_p> should be a float in [0,1]')
        # if not isinstance(mutate_p,float) or mutate_p<0 or mutate_p>1:
        #     raise TypeError('<mutate_p> should be a float in [0,1]')
        self.__gen_len=gen_len
        self.__size=size
        self.__mate_p=mate_p
        self.__mutate_p=mutate_p
        #self.__gens=[Produce_Gen.Gen(gen_len) for i in range(size)]#迭代产生一个种群list
        self.__gens = []
        self.__best=None
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.name = name_of_regressor
        self.iteration = iteration

    def generate(self):

        for i in range(int(self.__size * self.__mate_p / 2)):
            # g1, g2 = [self.__gens[random.randint(0, self.__size - 1)] for i in range(2)]

            a = random.sample(range(0, self.__size), 2)
            g1 = self.__gens[a[0]]
            g2 = self.__gens[a[1]]
            g3, g4 = g1.mate(g2)
            self.__gens.append(g3)
            self.__gens.append(g4)
        for i in range(self.__size):
            if random.random() < self.__mutate_p:
                self.__gens.append(self.__gens[i].mutate())

        gen_fits = []
        #gen_fits = [[g, self.fitness(g)] for g in self.__gens]
        for m in range(len(self.__gens)):
            gen_fits.append([self.__gens[m], self.fitness(self.__gens[m])])

        gen_fits.sort(key=lambda entry: entry[1], reverse=True)
        self.__best = gen_fits[0][0].bins()
        #self.__gens = [gt[0] for gt in gen_fits[:self.__size]]
        for n in range(self.__size):
            self.__gens[n] = gen_fits[m][0]


    # 获取当前最优解(基因,适应度)
    # def best(self):
    #     return self.__best


    def fitness(self,gen):
        List_of_Useful_Code = []
        for i in range(len(gen.bins())):
            if gen.bins()[i] == 1:
                List_of_Useful_Code.append(i)

        train = self.X_train[:, List_of_Useful_Code]
        test = self.X_test[:, List_of_Useful_Code]

        return self.try_different_method(train, self.y_train, test, self.y_test)


    def try_different_method(self,x_train,y_train,x_test,y_test):
        self.name.fit(x_train, y_train)
        score = self.name.score(x_test, y_test)
        result = np.around(self.name.predict(x_test))

        return score


    def generate_Population(self):
        for i in range(self.__size):
            self.__gens.append(Produce_Gen.Gen(self.__gen_len))


    def run(self):
        for i in range(self.iteration):
            self.generate()

        return self.__best





from sklearn import tree
from sklearn import linear_model
from sklearn import svm
linear_reg = linear_model.LinearRegression()
tree_reg = tree.DecisionTreeRegressor()
svr = svm.SVR()

data = pd.read_csv("12月13_Extraction_db4.csv")

a = np.array(data)
a = np.delete(a,0,1)
x,y = a[:,0:48],a[:,48]
X_train,X_test,y_train,y_test = train_test_split(x,y,test_size=0.3,random_state=0)


ga = GA(48,100,X_train,X_test,y_train,y_test,linear_reg,50)
ga.generate_Population()
print(ga.run())







#
# ga=GA(100,50,fitness,0.2,0.3)
# for i in range(100):
#     ga.generate()
#     g,f=ga.best()
#     print(f)
# x=g.decode(0,3,0,49)
# y=g.decode(0,3,50,99)
# print(x,y)