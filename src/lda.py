import pickle
import os
import gensim
import pyLDAvis

def getData():
    for dirs in os.scandir("../data"):
        for files in os.scandir(dirs):
            print(files)

getData()