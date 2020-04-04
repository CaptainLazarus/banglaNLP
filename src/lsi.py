import data
import os
import argparse
import pyLDAvis
import gensim
import json
import pickle

def lsi(fileName):
    with open(fileName , 'rb') as f:
        a = pickle.load(f)
        dictionary = gensim.corpora.Dictionary(a)
        dictionary.filter_extremes(no_below = 2 , no_above=0.1)


        corpse = [dictionary.doc2bow(text) for text in a]
        lsi = gensim.models.LsiModel(corpse ,id2word=dictionary , num_topics=8)

    name = list(list(fileName.split('/'))[-1].split("."))[-2]
    print(name)

    with open("../output/"+name+"_lsi.json" , 'w') as f:
        f.write(json.dumps(lsi.print_topics(-1), indent=4))

if __name__ == "__main__":    
    parser = argparse.ArgumentParser(description="Processing data")
    parser.add_argument("-rem" , metavar="R" , nargs="*" , type=str , help="List of POS tags to remove")
    args = parser.parse_args()

    fileName = "../data/94-documents"

    if not os.path.isfile(fileName):
        data.getData(fileName , args.rem)
        # print("\nEntered\n")
    lsi(fileName+".txt")