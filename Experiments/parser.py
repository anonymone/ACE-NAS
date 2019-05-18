import sys
sys.path.append("./hidata")
from hidata import hireader
import json
from pandas import DataFrame
import pandas
import numpy as np
import os

def readJson(file_path):
        file = open(file_path)
        file = file.read()
        return json.loads(file)

def tableParser(path):
    data = []
    for i in range(28):
        data.append(pandas.read_csv(filepath_or_buffer= path+"Generation"+str(i)+".dat.csv"))
    data = pandas.concat(data)
    error = data["Error"]
    comp = data["Complexity"]
    data = pandas.concat([error,comp],1)
    data.to_csv(path+"total.csv")
    return data

def Normalization(data):
    data = (data - np.mean(data))/(np.std(data))
    return data

class jsonHireader(hireader.hireader):
    def __init__(self):
        super(jsonHireader,self).__init__()

    def parser(self, data):
        tabel = {
            "Error": [],
            "Complexity" : []
        }
        for key in data:
            if "candidates" in key:
                ind = data[key].replace('         ',' ').replace('\n',"").replace('[',"").replace(']',"")
                ind = ind.split(' ')
                while '' in ind:
                    ind.remove('')
                ind = np.array(ind,dtype=np.float)
                tabel["Error"].append(ind[-2])
                tabel["Complexity"].append(ind[-1])
        return tabel
    
    def save(self, save_path= None, data = None, file_name = None):
        if save_path is not None:
            self.save_path = save_path
        dataF = DataFrame(data)
        dataF.to_csv(file_name.replace('./logs','./logs/csv')+".csv")

if __name__=="__main__":
    # parser = jsonHireader()
    # parser.init(work_path="./logs", read_tool= readJson,log_path="./parser_log")
    # parser.read()
    # parser.close()
    tableParser('./logs/csv/')