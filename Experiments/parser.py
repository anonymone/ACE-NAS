from hidata import hidata
import json

class jsonHireader(hidata.hireader):
    def __init__(self):
        super(jsonHireader,self).__init__()

    def save(self, save_path=None, data = None, file_name = None):
        pass
    
    def parser(self, data):
        pass