import data
import pickle
import os
import argparse
import pyLDAvis.gensim
import pyLDAvis
import gensim
import json

def lda(fileName):
    with open(fileName , 'rb') as f:
        a = pickle.load(f)
        dictionary = gensim.corpora.Dictionary(a)
        dictionary.filter_extremes(no_below = 2 , no_above=0.1)


        corpse = [dictionary.doc2bow(text) for text in a]
        lda = gensim.models.ldamodel.LdaModel(corpse , num_topics = 8 , id2word=dictionary , passes=10)
        lsi = gensim.models.LsiModel(corpse ,id2word=dictionary , num_topics=8)

    name = list(list(fileName.split('/'))[-1].split("."))[-2]
    # print(name)

    #LDA
    lda_name = "../output/"+name+"_lda.json"
    with open(lda_name , 'w') as f:
        f.write(json.dumps(lda.print_topics(-1), indent=4))
    
    web = pyLDAvis.gensim.prepare(topic_model=lda , corpus=corpse , dictionary=dictionary)
    htmlName = lda_name+".html"
    pyLDAvis.save_html(web , htmlName)

    #LSI
    lsi_name = "../output/"+name+"_lsi.json"
    with open(lsi_name , 'w') as f:
        f.write(json.dumps(lsi.print_topics(-1), indent=4))


if __name__ == "__main__":    
    parser = argparse.ArgumentParser(description="Processing data")
    parser.add_argument("-rem" , metavar="R" , nargs="*" , type=str , help="List of POS tags to remove")
    args = parser.parse_args()

    fileName = "../data/94-documents"

    if not os.path.isfile(fileName):
        data.getData(fileName , args.rem)
        # print("\nEntered\n")
    lda(fileName+".txt")