import numpy as np
import sklearn
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.datasets import load_files
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
import os

import spacy
# python -m spacy download en
import json



# with open('train_data.json') as json_file:  
#     test_dic = json.load(json_file)

def readData(rootPath):
    category = ["pos","neg"]
    #load only labeled data
    movie_train = load_files(rootPath + "aclImdb/train", shuffle=True, categories=category)
    movie_test = load_files(rootPath + "aclImdb/test", shuffle=True, categories=category)
    return [movie_train, movie_test]

train_data, test_data = readData("")

corpus = train_data.data
corpus = corpus[0:100]

# defines a custom vectorizer class
class CustomVectorizer(CountVectorizer): 

    count = 0

    # overwrite the build_analyzer method, allowing one to
    # create a custom analyzer for the vectorizer
    def build_analyzer(self):

        # load stop words using CountVectorizer's built in method
        stop_words = self.get_stop_words()
        

        # create the analyzer that will be returned by this method
        def analyser(doc):
            doc = doc.decode("utf-8")

            # load spaCy's model for english language
            spacy.load('en')
            
            # instantiate a spaCy tokenizer
            lemmatizer = spacy.lang.en.English()
            
            # apply the preprocessing and tokenzation steps
            doc_clean = (doc).lower()
            tokens = lemmatizer(doc_clean)

            lemmatized_tokens = [token.lemma_ for token in tokens]
            # print(lemmatized_tokens)
            
            # use CountVectorizer's _word_ngrams built in method
            # to remove stop words and extract n-grams

            # for that sentence, find the unigram and bigram
            unigram_bigram = self._word_ngrams(lemmatized_tokens, stop_words)

            count = CustomVectorizer.count

            # lst_pair = test_dic[str(count)]

            # for pair in lst_pair:
            #     temp_str = "" + str(pair[0]) + " " + str(pair[1])
            #     unigram_bigram.append(temp_str)

            CustomVectorizer.count = count + 1

            progress = count/len(corpus)*100
            print (progress,"% completed")
            # print(unigram_bigram)
            return(unigram_bigram)

        return(analyser)
    

custom_vec = CustomVectorizer(ngram_range=(1,1),stop_words='english')

matrix = custom_vec.fit_transform(corpus).toarray()
name = custom_vec.get_feature_names()