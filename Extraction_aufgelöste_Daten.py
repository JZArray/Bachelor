import pywt as dwt
import math
import numpy as np
import pandas as pd
from sklearn import preprocessing


class Extraction:

    def __init__(self,Rohdata_name:str,level:int,name_of_Wave:str):
        self.Rohdata = Rohdata_name
        #self.Name_of_Extraction = Name_of_Extraction
        self.n = level
        self.txt = name_of_Wave
        self.list_of_name = []



#Energie of coeffizints but the first one is An not D1 different from literatur
    def calc_energie(self,signal):
        count = 0
        fea_en = []
        for i in self.calc_wavelet(signal):
            energie = 0
            for j in range(i.size):
                energie += pow(i.data[j],2)

            fea_en.append(energie)
            if count == 0:
                self.list_of_name.append('Energie of Approximation'+str(self.n))
            else:
                self.list_of_name.append('Energie of Details'+str(self.n +1 -count))
            count = count + 1

        return fea_en

#Entropy of coeffizints but the first one is An not D1 different from literatur
    def calc_entropy(self,signal):
        count = 0
        fea_ent = []
        for i in self.calc_wavelet(signal):
            entropy = 0
            for j in range(i.size):
                if(i.data[j] != 0):
                    entropy += pow(i.data[j],2) * math.log(i.data[j] * i.data[j],10)

            fea_ent.append(-entropy)
            if count == 0:
                self.list_of_name.append('Entropy of Approximation'+str(self.n))
            else:
                self.list_of_name.append('Entropy of Details'+str(self.n +1 -count))
            count = count + 1
        return fea_ent

#Standard deviation of coeffizints but the first one is An not D1 different from literatur
    def calc_standard_deviation(self,signal):
        count = 0
        fea_standard = []
        for i in self.calc_wavelet(signal):
            sum = 0
            for j in range(i.size):
                sum += pow(i.data[j] - i.mean(),2)

            fea_standard.append(math.sqrt(sum/(i.size -1)))
            if count == 0:
                self.list_of_name.append('Standard deviation of Approximatio'+str(self.n))
            else:
                self.list_of_name.append('Standard deviation of Details'+str(self.n +1 -count))

            count = count + 1
        return fea_standard

#Mean of coeffizints but the first one is An not D1 different from literatur
    def calc_mean(self,signal):
        count = 0
        fea_mean = []
        for i in self.calc_wavelet(signal):

            fea_mean.append(i.mean())
            if count == 0:
                self.list_of_name.append('Mean of Approximatio'+str(self.n))
            else:
                self.list_of_name.append('Mean deviation of Details'+str(self.n +1 -count))
            count = count + 1
        return fea_mean

#Kurtosis of coeffizints but the first one is An not D1 different from literatur
    def calc_kurtosis(self,signal):
        count = 0
        fea_kurt = []
        for i in self.calc_wavelet(signal):
            n = i.size
            m = 0
            m2 = 0
            m3 = 0
            m4 = 0
            sum = 0
            for j in range(i.size):
                sum += pow(i.data[j] - i.mean(), 2)
                m += i.data[j]
                m2 += i.data[j] * i.data[j]
                m3 += i.data[j] ** 3
                m4 += i.data[j] ** 4
            m /= n
            m2 /= n
            m3 /= n
            m4 /= n
            mu = m
            sigma = math.sqrt(sum/n)
            kurtosis = (m4 - 4 * mu * m3 + 6 * mu * mu * m2 - 4 * mu ** 3 * mu + mu ** 4) / sigma ** 4

            fea_kurt.append(kurtosis)
            if count == 0:
                self.list_of_name.append('Kurtosis of Approximatio'+str(self.n))
            else:
                self.list_of_name.append('Kurtosis of Details'+str(self.n +1 -count))
            count = count + 1
        return fea_kurt

#Skewness of coeffizints but the first one is An not D1 different from literatur
    def calc_skewness(self,signal):
        count = 0
        fea_skew = []
        for i in self.calc_wavelet(signal):
            n = i.size
            m = 0
            m2 = 0
            m3 = 0
            m4 = 0
            sum = 0
            for j in range(i.size):
                sum += pow(i.data[j] - i.mean(), 2)
                m += i.data[j]
                m2 += i.data[j] * i.data[j]
                m3 += i.data[j] ** 3
                m4 += i.data[j] ** 4
            m /= n
            m2 /= n
            m3 /= n
            m4 /= n
            mu = m
            sigma = math.sqrt(sum/n)
            skew = (m3 - 3 * mu * m2 + 2 * mu ** 3) / sigma ** 3

            fea_skew.append(skew)
            if count == 0:
                self.list_of_name.append('Skewness of Approximatio'+str(self.n))
            else:
                self.list_of_name.append('Skewness of Details'+str(self.n +1 -count))
            count = count + 1
        return fea_skew

#RMS_Value of coeffizints but the first one is An not D1 different from literatur
    def calc_rms(self,signal):
        count = 0
        fea_rms = []
        for i in self.calc_wavelet(signal):
            sum = 0
            for j in range(i.size):
                sum += i.data[j]*i.data[j]

            fea_rms.append(math.sqrt(sum / i.size))
            if count == 0:
                self.list_of_name.append('RMS_Extraction of Approximatio'+str(self.n))
            else:
                self.list_of_name.append('RMS_Extraction of Details'+str(self.n +1 -count))
            count = count + 1
        return fea_rms

#Rang of coeffizints but the first one is An not D1 different from literatur
    def calc_range(self,signal):
        count = 0
        fea_range = []
        for i in self.calc_wavelet(signal):
            fea_range.append(max(i) - min(i))
            if count == 0:
                self.list_of_name.append('Range of Approximatio'+str(self.n))
            else:
                self.list_of_name.append('Range of Details'+str(self.n +1 -count))
            count = count + 1
        return fea_range
    #
    # def calc_loss_energie(self,signal):
    #     count = 0
    #     fea_loss = []
    #     for i in self.calc_wavelet(signal):
    #         loss = 0
    #         for j in range(i.size):
    #             if (i.data[j] != 0):
    #                 loss += math.log(i.data[j] * i.data[j], 10)
    #
    #         fea_loss.append(loss)
    #         if count == 0:
    #             self.list_of_name.append('Loss of Approximatio'+str(self.n))
    #         else:
    #             self.list_of_name.append('Loss of Details'+str(self.n +1 -count))
    #         count = count + 1
    #     return fea_loss


    def calc_wavelet(self,signal):
        #[cA_n, cD_n, cD_n-1, …, cD2, cD1]
        return dwt.wavedec(signal,self.txt,level=self.n)


    def create_Data_after_extraction(self):
        test_data = pd.read_csv(self.Rohdata).values
        Data = pd.DataFrame()
        m = 0
        List = []
        for x in range(test_data.shape[0]):
            signal = test_data[x, 1:test_data.shape[1] - 1]
            feature_extraction = []
            feature_extraction.append(self.calc_energie(signal))
            feature_extraction.append(self.calc_entropy(signal))
            feature_extraction.append(self.calc_standard_deviation(signal))
            feature_extraction.append(self.calc_mean(signal))
            feature_extraction.append(self.calc_kurtosis(signal))
            feature_extraction.append(self.calc_skewness(signal))
            feature_extraction.append(self.calc_rms(signal))
            feature_extraction.append(self.calc_range(signal))
            #feature_extraction.append(self.calc_loss_energie(signal))
            array = np.array(feature_extraction).T
            min_max_scaler = preprocessing.MinMaxScaler()
            array_minmax = min_max_scaler.fit_transform(array)
            m = array_minmax.shape[0]*array_minmax.shape[1]
            c = array_minmax.reshape(1,array_minmax.shape[0]*array_minmax.shape[1])
            Feature_with_label = np.c_[c,np.array(test_data[x,test_data.shape[1]-1])]
            f=pd.DataFrame(Feature_with_label)
            Data = pd.concat([Data,f])

        for i in range(m):
            List.append(self.list_of_name[i])

        List.append('Label')
        Data.columns = List
        #Data.to_csv(self.Name_of_Extraction)

        return Data


#test = Test_extraction('Störung__2018-12-05_15-24-27_RMS_220_Sampling_5000_casePerClass_50.csv','12月15_Extraction_without_Loss.csv',6,'db4')
#test.create_Data_after_extraction()