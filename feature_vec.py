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

test_dic = {"0":[["first","document","nsubj"], ["is", "first", "dobj"], ["ss", "ass", "asas"]],
            "1":[["second","document","xcomp"]],
            "2":[["one", "document","sat"]],
            "3":[["this", "is", "sasi"]]}

with open('test_dic.json', 'w') as outfile:  
    json.dump(test_dic, outfile)


with open('test_dic.json') as json_file:  
    test_dic = json.load(json_file)


corpus = ['This is the first document.',
'This is the second second document.',
'And the third one.',
'Is this the first document?']

bigram_vectorizer = CountVectorizer(ngram_range=(1, 2), 
    token_pattern=r'\b\w+\b', min_df=1)
# print(bigram_vectorizer)


analyze = bigram_vectorizer.build_analyzer()

res = bigram_vectorizer.fit_transform(corpus).toarray()
# print(res)




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

            lst_pair = test_dic[str(count)]

            for pair in lst_pair:
                temp_str = "" + str(pair[0]) + " " + str(pair[1])
                unigram_bigram.append(temp_str)

            CustomVectorizer.count = count + 1
            print(unigram_bigram)
            return(unigram_bigram)

        return(analyser)
    

custom_vec = CustomVectorizer(ngram_range=(1,2),
                              stop_words='english')

matrix = custom_vec.fit_transform(corpus).toarray()
name = custom_vec.get_feature_names()


print(matrix)
print(name)















# def createNewFeatureVec():



### ### ###
# def readData(rootPath):
#     category = ["pos","neg"]
#     #load only labeled data
#     movie_train = load_files(rootPath + "aclImdb/train", shuffle=True, categories=category)
#     movie_test = load_files(rootPath + "aclImdb/test", shuffle=True, categories=category)
#     return [movie_train, movie_test]

# def createFeatureVec(dataset, ngram_range, feature_type = "bow"):   
#     count_vect = CountVectorizer(ngram_range, stop_words="english")
#     feature_vector = count_vect.fit_transform(dataset.data)
#     if feature_type == "tf":
#         tf_transformer = TfidfTransformer(use_idf=False).fit(feature_vector)
#         feature_vector = tf_transformer.transform(feature_vector)
#     elif feature_type == "tf_idf":
#         tfidf_transformer = TfidfTransformer()
#         feature_vector = tfidf_transformer.fit_transform(feature_vector)
#     return count_vect, feature_vector


    



# def trainNB(train_data, feature_type="bow", ngram_range=(1, 2)):
#   # default is unigram + bigram and with stop words removed
#   train_count_vect, train_feature_vector = createFeatureVec(train_data, ngram_range, feature_type)
#   clf = MultinomialNB().fit(train_feature_vector, train_data.target)
#   train_error = np.mean(clf.predict(train_feature_vector) == train_data.target)
#   return ([train_acc, train_count_vect, clf])


# def testNB_SVM(clf, train_count_vect, test_data, feature_type = "bow"):
#     test_feature_vector = train_count_vect.transform(test_data.data)
#     if feature_type == "tf":
#         tf_transformer = TfidfTransformer(use_idf=False).fit(test_feature_vector)
#         test_feature_vector = tf_transformer.transform(test_feature_vector)
#     elif feature_type == "tf_idf":
#         tfidf_transformer = TfidfTransformer()
#         test_feature_vector = tfidf_transformer.fit_transform(test_feature_vector)

#     test_acc = np.mean(clf.predict(test_feature_vector) == test_data.target)
#     return test_error

# def trainSVM(train_data, feature_type="bow", ngram_range=(1, 2)):
#     train_count_vect, train_feature_vector = createFeatureVec(train_data, ngram_range, feature_type)
#     # clf = SVC(kernel="linear").fit(train_feature_vector, train_data.target)
#     clf = SVC(kernel="rbf").fit(train_feature_vector, train_data.target)
#     print(clf.predict(train_feature_vector))
#     train_acc = np.mean(clf.predict(train_feature_vector) == train_data.target)
#     return ([train_acc, train_count_vect, clf])


# #uncomment lines below

# feature_type = "bow" 
# # with unigram + bigram: training acc = 0.99704; test acc = 0.84272

# # feature_type = "tf" 
# #with unigram + bigram: training acc = 0.94792; test acc = 0.85372

# # feature_type = "tf_idf" 
# #with unigram + bigram: training acc = 0.9844; test acc = 0.85476

# ngram_range = (1, 2)

# train_data, test_data = readData("")
# # train_acc, train_count_vect, clf = trainNB(train_data, feature_type, ngram_range)
# # test_acc = testNB(clf, train_count_vect, test_data, feature_type)

# train_acc_svm, train_count_vect_error, clf_svm = trainSVM(train_data, feature_type, ngram_range)
# test_acc_svm = testNB_SVM(clf_svm, train_count_vect_error, test_data, feature_type)
# print("SVM Training Accuracy:", train_acc_svm, "\nTesting Accuracy", test_acc_svm)
# # print("Training Accuracy:", train_acc,"\nTesting Accuracy",test_acc)



