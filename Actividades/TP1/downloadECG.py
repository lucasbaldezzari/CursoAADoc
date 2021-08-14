# -*- coding: utf-8 -*-
"""
Created on Fri Aug  6 10:36:54 2021

@author: Lucas
"""

import wfdb
import os 

import json

def loadParams(fileName):
    with open(fileName,'r') as json_file:
       return json.load(json_file)
   
def saveParams(data, fileName):   
    with open(fileName,'w') as fp:
        fp.write(json.dumps(data))

def main():

    
    wfdb.dl_files('mitdb', "mitdata", ["100.hea", "100.dat"])                # Descarga la señal 100
    ecg, fields = wfdb.rdsamp(os.path.join("mitdata", "100"), channels=[0])  # Lectura de la señal
    print(fields)

    saveParams(fields, "atributosECG-TP1.txt")
    
    
if __name__ == "__main__":
    main()
    
#atributos = loadParams("atributosECG-TP1.txt")

