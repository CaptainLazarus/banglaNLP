import pickle
import os

def getData(fileName , rem):
    data = []
    idx=0
    for file in os.scandir(fileName):
        data.append([])
        with open(file , "r") as f:
            for line in f:
                raw_text = list(line.split("\t"))
                word = raw_text[0]
                if word == "\n":
                    pass
                if rem is None:
                    data[idx].append(word)
                else:
                    pos = raw_text[-1][:-1]
                    if pos in rem:
                        pass
                    else:
                        data[idx].append(word)
        idx+=1
    with open(fileName+".txt" , "wb") as f:
        pickle.dump(data , f)