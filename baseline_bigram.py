import numpy as np
import sklearn
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.datasets import load_files
from sklearn.pipeline import Pipeline
import os


def readData(rootPath):
	category = ["pos","neg"]
	#load only labeled data
	movie_train = load_files(rootPath + "/aclImdb/train", shuffle=True, categories=category)
	movie_test = load_files(rootPath + "/aclImdb/test", shuffle=True, categories=category)
	return [movie_train, movie_test]

def createFeatureVec(dataset, ngram_range, feature_type = "bow"):	
	count_vect = CountVectorizer(ngram_range, stop_words="english")
	feature_vector = count_vect.fit_transform(dataset.data)
	if feature_type == "tf":
		tf_transformer = TfidfTransformer(use_idf=False).fit(feature_vector)
		feature_vector = tf_transformer.transform(feature_vector)
	elif feature_type == "tf_idf":
		tfidf_transformer = TfidfTransformer()
		feature_vector = tfidf_transformer.fit_transform(feature_vector)
	return count_vect, feature_vector

def trainNB(train_data, feature_type="bow", ngram_range=(1, 2)):
	# default is unigram + bigram and with stop words removed
	train_count_vect, train_feature_vector = createFeatureVec(train_data, ngram_range, feature_type)
	clf = MultinomialNB().fit(train_feature_vector, train_data.target)
	train_error = np.mean(clf.predict(train_feature_vector) == train_data.target)
	return ([train_error, train_count_vect, clf])


def testNB(clf, train_count_vect, test_data, feature_type = "bow"):
	test_feature_vector = train_count_vect.transform(test_data.data)
	if feature_type == "tf":
		tf_transformer = TfidfTransformer(use_idf=False).fit(test_feature_vector)
		test_feature_vector = tf_transformer.transform(test_feature_vector)
	elif feature_type == "tf_idf":
		tfidf_transformer = TfidfTransformer()
		test_feature_vector = tfidf_transformer.fit_transform(test_feature_vector)

	test_error = np.mean(clf.predict(test_feature_vector) == test_data.target)
	return test_error


#uncomment lines below

# feature_type = "bow" 
#with unigram + bigram: training acc = 0.99704; test acc = 0.84272

# feature_type = "tf" 
#with unigram + bigram: training acc = 0.94792; test acc = 0.85372

# feature_type = "tf_idf" 
#with unigram + bigram: training acc = 0.9844; test acc = 0.85476

# ngram_range = (1, 2)

# train_data, test_data = readData("701-project")
# train_error, train_count_vect, clf = trainNB(train_data, feature_type, ngram_range)
# test_error = testNB(clf, train_count_vect, test_data, feature_type)

# print("Training Error:", train_error,"\nTesting Error",test_error)

