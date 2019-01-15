# def main():
#     print("hallo")
# if __name__ == '__main__':
#     main()
import time
from Extraction_RMS_Version3 import RMS_Extraction
from Extraction_aufgelöste_Daten import Extraction

def main(name_of_type:str,Rohdata_name:str,Name_of_Extraction:str,segment = None,level = None,name_of_Wave = None):

    if segment == None :
        segment = 200
    if level == None :
        level = 6
    if name_of_Wave == None:
        name_of_Wave = 'haar'

    s = 'Rohdaten/' + Rohdata_name
    dictionary = {'RMS_Extraction': 1,'Extraction': 2}
    stand = dictionary[name_of_type]

    if stand == 1:
        tempo = RMS_Extraction(segment, s).create_Data_after_extraction()
    else:
        tempo = Extraction(s,level, name_of_Wave).create_Data_after_extraction()

    zeit = time.strftime("%d-%m-%Y %H:%M:%S", time.localtime())

    tempo.to_csv('Extractiondata/' + name_of_type + '/' + Name_of_Extraction + zeit)











if __name__ == '__main__':
    #main('RMS_Extraction','20180601-20180603_E_TR1_1_UrmsL1.csv','RMS的Extraction.csv',300,6,'db4')
    #main('Extraction','Störung__2018-12-05_15-24-27_RMS_220_Sampling_5000_casePerClass_50.csv','12月15_Extraction_without_Loss.csv',300,6,'db4')
    #main('RMS_Extraction', '20180601-20180603_E_TR1_1_UrmsL1.csv', 'RMS的Extraction.csv')
    main('Extraction', 'Störung__2018-12-05_15-24-27_RMS_220_Sampling_5000_casePerClass_50.csv','12月15_Extraction_without_Loss.csv')