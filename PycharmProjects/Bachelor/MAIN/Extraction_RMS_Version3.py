import pandas as pd
import math
import numpy as np
from sklearn import preprocessing

class RMS_Extraction:

    def __init__(self,segment:int,Rohdata_name:str):
        self.Rohdata = Rohdata_name
        #self.Name_of_Extraction = Name_of_Extraction
        self.segment = segment
        test_data = pd.read_csv(self.Rohdata).values[4:, 0]
        self.list = [float(x[27:35]) for x in test_data]

    def calc_energie(self):
        fea_energie = []
        n = math.ceil(len(self.list) / self.segment)

        for i in range(self.segment - 1):
            sum = 0
            for m in self.list[i * n : (i + 1) * n]:
                sum = sum + pow(m,2)

            fea_energie.append(sum)

        sum2 = 0

        for k in self.list[n * (self.segment - 1) : ]:
            sum2 = sum2 + pow(k,2)

        fea_energie.append(sum2)

        return fea_energie

    def calc_entropy(self):
        fea_entropy = []
        n = math.ceil(len(self.list) / self.segment)


        for i in range(self.segment - 1):
            sum = 0
            for m in self.list[i * n : (i + 1) * n]:
                sum = sum + pow(m,2) * math.log(m * m,10)

            fea_entropy.append(-sum)

        sum2 = 0

        for k in self.list[n * (self.segment - 1):]:
            sum2 = sum2 + pow(k,2) * math.log(k * k,10)
        fea_entropy.append(-sum2)


        return fea_entropy

    def calc_means(self):
        fea_mean = []
        n = math.ceil(len(self.list) / self.segment)

        for i in range(self.segment - 1):
            fea_mean.append(np.mean(self.list[i * n : (i + 1) * n]))

        fea_mean.append(np.mean(self.list[n * (self.segment - 1):]))

        return fea_mean


    def calc_standardabweichung(self):
        fea_standardabweichung = []
        n = math.ceil(len(self.list) / self.segment)

        for i in range(self.segment - 1):
            sum = 0
            mean = np.mean(self.list[i * n : (i + 1) * n])
            for m in self.list[i * n : (i + 1) * n]:
                sum = sum + pow(m - mean , 2)

            fea_standardabweichung.append(math.sqrt(sum/(n - 1)))

        sum2 = 0
        mean2 = np.mean(self.list[n * (self.segment - 1):])

        if len(self.list[n * (self.segment - 1):]) > 1 :

            for k in self.list[n * (self.segment - 1):]:
                sum = sum + pow(k - mean2 , 2)

            fea_standardabweichung.append(math.sqrt(sum2 / (len(self.list[n * (self.segment - 1):]) - 1)))
        else:
            fea_standardabweichung.append(0)


        return fea_standardabweichung

    def calc_maximum(self):
        fea_max = []
        n = math.ceil(len(self.list) / self.segment)
        for i in range(self.segment - 1):
            fea_max.append(np.max(self.list[i * n : (i + 1) * n]))

        fea_max.append(np.max(self.list[n * (self.segment - 1):]))

        return fea_max

    def calc_minimum(self):
        fea_mini = []
        n = math.ceil(len(self.list) / self.segment)
        for i in range(self.segment - 1):
            fea_mini.append(np.min(self.list[i * n: (i + 1) * n]))

        fea_mini.append(np.min(self.list[n * (self.segment - 1):]))

        return fea_mini

    def calc_range(self):
        fea_range = []
        n = math.ceil(len(self.list) / self.segment)
        for i in range(self.segment - 1):
            fea_range.append(np.max(self.list[i * n : (i + 1) * n]) - np.min(self.list[i * n : (i + 1) * n]))

        fea_range.append(np.max(self.list[n * (self.segment - 1):]) - np.min(self.list[n * (self.segment - 1):]))

        return fea_range

    def create_Data_after_extraction(self):

        feature_extraction = []
        feature_extraction.append(self.calc_energie())
        feature_extraction.append(self.calc_entropy())
        feature_extraction.append(self.calc_means())
        feature_extraction.append(self.calc_standardabweichung())
        feature_extraction.append(self.calc_maximum())
        feature_extraction.append(self.calc_minimum())
        feature_extraction.append(self.calc_range())

        array = np.array(feature_extraction).T

        min_max_scaler = preprocessing.MinMaxScaler()

        array_minmax = min_max_scaler.fit_transform(array)

        f = pd.DataFrame(array_minmax)

        f.columns = ['energie', 'entropy', 'mean', 'standardabweichung', 'maximum', 'minimum', 'range']

        #f.to_csv('Extractiondata/' + self.Name_of_Extraction)

        return f




#a = RMS_Extraction(300,'DATA/20180601-20180603_E_TR1_1_UrmsL1.csv','RMS的Extraction.csv')

#print(a.calc_energie())
#
#print(a.create_Data_after_extraction())

