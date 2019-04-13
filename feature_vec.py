import numpy as np
import sklearn
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.datasets import load_files
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
import os
from scipy import sparse
import re
import pickle

import spacy
# python -m spacy download en
import json


<<<<<<< HEAD

# with open('train_data.json') as json_file:  
#     test_dic = json.load(json_file)
=======
with open('train_data.json') as json_file:  
    dependency_dic = json.load(json_file)
>>>>>>> 4bf4bfc8ab8db95947f86ff9c533b5a4e7a9de4f

def readData(rootPath):
    category = ["pos","neg"]
    #load only labeled data
    movie_train = load_files(rootPath + "aclImdb/train", shuffle=True, categories=category)
    movie_test = load_files(rootPath + "aclImdb/test", shuffle=True, categories=category)
    return [movie_train, movie_test]

train_data, test_data = readData("")

corpus = train_data.data
<<<<<<< HEAD
corpus = corpus[0:100]
=======
corpus = corpus
>>>>>>> 4bf4bfc8ab8db95947f86ff9c533b5a4e7a9de4f

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
<<<<<<< HEAD
            doc = doc.decode("utf-8")
=======
            doc = doc.decode("utf-8-sig")
            doc = re.sub(r'(\s*<br.*?>)+\s*', " ", doc)
            doc = re.sub("[^a-zA-Z]+", " ", doc)
>>>>>>> 4bf4bfc8ab8db95947f86ff9c533b5a4e7a9de4f

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

<<<<<<< HEAD
            # lst_pair = test_dic[str(count)]
=======
            lst_pair = dependency_dic[str(count)]
>>>>>>> 4bf4bfc8ab8db95947f86ff9c533b5a4e7a9de4f

            # for pair in lst_pair:
            #     temp_str = "" + str(pair[0]) + " " + str(pair[1])
            #     unigram_bigram.append(temp_str)

<<<<<<< HEAD
            CustomVectorizer.count = count + 1

            progress = count/len(corpus)*100
            print (progress,"% completed")
=======

            progress = CustomVectorizer.count/len(corpus)*100
            CustomVectorizer.count += 1

            print (progress,"% processed")
>>>>>>> 4bf4bfc8ab8db95947f86ff9c533b5a4e7a9de4f
            # print(unigram_bigram)
            return(unigram_bigram)

        return(analyser)
    

<<<<<<< HEAD
custom_vec = CustomVectorizer(ngram_range=(1,1),stop_words='english')

matrix = custom_vec.fit_transform(corpus).toarray()
name = custom_vec.get_feature_names()
=======
custom_vectorizer = CustomVectorizer(ngram_range=(1,3),stop_words='english',
                                encoding="utf-8-sig",
                                token_pattern=r"(?u)\b\w\w+\b",
                                max_df=1.0, min_df=1, lowercase=True,
                                max_features=1000000)

# vec = CountVectorizer(ngram_range=(1,2),stop_words='english')

custom_vector = custom_vectorizer.fit_transform(corpus)
# smatrix = sparse.csr_matrix(matrix)
# name = custom_vec.get_feature_names()

# matrix_vec = vec.fit_transform(corpus)
# name_vec = vec.get_feature_names()

with open('custom_vectorizer.pickle', 'wb') as f:
    # Pickle the 'data' dictionary using the highest protocol available.
    pickle.dump(custom_vectorizer, f, pickle.HIGHEST_PROTOCOL)

with open('custom_vector.pickle', 'wb') as f:
    # Pickle the 'data' dictionary using the highest protocol available.
    pickle.dump(custom_vector, f, pickle.HIGHEST_PROTOCOL)

with open('feature_names.pickle',"wb") as f:
    pickle.dump(custom_vectorizer.get_feature_names(), f, pickle.HIGHEST_PROTOCOL)

print ("number of features:", len(custom_vectorizer.get_feature_names()))

# with open('data.pickle', 'rb') as f:
#     # The protocol version used is detected automatically, so we do not
#     # have to specify it.
#     data = pickle.load(f)


>>>>>>> 4bf4bfc8ab8db95947f86ff9c533b5a4e7a9de4f
