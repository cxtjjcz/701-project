import os, argparse
import numpy as np
import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.datasets import load_files
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.decomposition import NMF, LatentDirichletAllocation
import pandas as pd

from utils import *

parser = argparse.ArgumentParser(description = 'Movie Review Sentiment Analysis')
parser.add_argument('--vect', default = 'tf', type = str, help = 'Specify vectorization method.')
parser.add_argument('--ngram', default = 2, type = str, help = 'Specify ngram range.')
parser.add_argument('--topic', default = 'LDA', type = str, help = 'Specify topic model type.')
parser.add_argument('--clf', default = 'NB', type = str, help = 'Specify choice of classifier.')
parser.add_argument('--num_feat', default = 1000, type = int)
parser.add_argument('--num_topic', default = 20, type = int)
parser.add_argument('--display_topics', default = False, type = bool)
parser.add_argument('--num_top_topics', default = 5, type = int) 
# An example is contained in the training sets of its top-5 most relevant topic-specific classifiers

################################################################################
# RESULTS:

# feature_type = "bow" 
# with unigram + bigram: training acc = 0.99704; test acc = 0.84272

# feature_type = "tf" 
#with unigram + bigram: training acc = 0.94792; test acc = 0.85372

# feature_type = "tf_idf" 
#with unigram + bigram: training acc = 0.9844; test acc = 0.85476

# TODO:
# 1. Optimize topic model to extract more meaningful topics, consider doing cross validation
#    to select the optimal number of topics/classifiers being used later on

################################################################################

def readData(rootPath):
	category = ["pos","neg"]
	#load only labeled data
	movie_train = load_files(rootPath + "aclImdb/train", shuffle=True, categories=category)
	movie_test = load_files(rootPath + "aclImdb/test", shuffle=True, categories=category)
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

def createFeatureVecForTopic(dataset, feature_type, ngram_range, num_feat, topic):
	if topic == 'LDA':
		assert(feature_type == 'tf')
		tf_vectorizer = CountVectorizer(max_df=0.95, min_df=2, max_features=num_feat, stop_words='english')
		tf = tf_vectorizer.fit_transform(dataset.data)
		tf_feature_names = tf_vectorizer.get_feature_names()
		vectors, features = (tf, tf_feature_names)

	if topic == 'NMF':
		assert(feature_type == 'tf_idf')
		tfidf_vectorizer = TfidfVectorizer(max_df=0.95, min_df=2, max_features=num_feat, stop_words='english')
		tfidf = tfidf_vectorizer.fit_transform(dataset.data)
		tfidf_feature_names = tfidf_vectorizer.get_feature_names() # words whose frequency 'matters'
		vectors, features = (tfidf, tfidf_feature_names)

	return vectors, features

def trainNB(train_data, feature_type="bow", ngram_range=(1, 2)):
	# default is unigram + bigram and with stop words removed
	train_count_vect, train_feature_vector = createFeatureVec(train_data, ngram_range, feature_type)
	clf = MultinomialNB().fit(train_feature_vector, train_data.target)
	train_acc = np.mean(clf.predict(train_feature_vector) == train_data.target)
	return ([train_acc, train_count_vect, clf])

def testNB_SVM(clf, train_count_vect, test_data, feature_type = "bow"):
	test_feature_vector = train_count_vect.transform(test_data.data)
	if feature_type == "tf":
		tf_transformer = TfidfTransformer(use_idf=False).fit(test_feature_vector)
		test_feature_vector = tf_transformer.transform(test_feature_vector)
	elif feature_type == "tf_idf":
		tfidf_transformer = TfidfTransformer()
		test_feature_vector = tfidf_transformer.fit_transform(test_feature_vector)

	test_acc = np.mean(clf.predict(test_feature_vector) == test_data.target)
	return test_acc

def trainSVM(train_data, feature_type="bow", ngram_range=(1, 2)):
	train_count_vect, train_feature_vector = createFeatureVec(train_data, ngram_range, feature_type)
	# clf = SVC(kernel="linear").fit(train_feature_vector, train_data.target)
	clf = SVC(kernel="rbf").fit(train_feature_vector, train_data.target)
	print(clf.predict(train_feature_vector))
	train_acc = np.mean(clf.predict(train_feature_vector) == train_data.target)
	return ([train_acc, train_count_vect, clf])

def createTopicModel(dataset, feature_type, ngram_range, num_feat, topic, num_topic):
	# get document-word matrix (vectors) and total words (features)
	vectors, features = createFeatureVecForTopic(dataset, feature_type, ngram_range, num_feat, topic)
	# run topic model
	if topic == 'LDA':
		topic_model = LatentDirichletAllocation(n_components=num_topic, max_iter=5, learning_method='online', \
												learning_offset=50., random_state=0).fit(vectors)
	if topic == 'NMF':
		topic_model = NMF(n_components=no_topics, random_state=1, alpha=.1, l1_ratio=.5, init='nndsvd').fit(vectors)
		# 'nndsvd' initialization helps with convergence
	# topic_model = document-topic matrix + topic-word matrix (decomposition of a doc-word matrix)
	return topic_model, features, vectors

def split_by_topic(args, topic_model, train_X):
	'''
	Generates a matrix of size (num_samples, num_topics) in which X_i,j = 1 if doc i is used for clf j
	'''
	num_samples = train_X.shape[0]
	num_topics = args.num_topic
	num_top_topics = args.num_top_topics

	doc_clf_mask = np.zeros((num_samples, num_topics))
	doc_topic_distr = topic_model.transform(train_X) # num_samples x num_topics
	#doc_topic_distr = np.random.randn(num_samples, num_topics)

	print("--- Iterating through training examples to find their topic mixtures...")

	for doc_i, doc in enumerate(doc_topic_distr):
		sorted_clf_idx = sorted(list(range(num_topics)), key=doc.__getitem__, reverse=True)[:num_top_topics]
		for i in sorted_clf_idx:
			doc_clf_mask[doc_i, i] = 1

	return doc_clf_mask

def train_main(args, doc_clf_mask, train_data, train_feature_vector):
	'''
	Note:
	- train_data.data is a list of strings
	'''
	num_topics = args.num_topic
	feature_type = args.vect
	ngram_range = (1, args.ngram)

	clfs = []
	clf_accs = []
	# feature_vector: np.array
	for clf_i in range(num_topics):
		curr_mask = (doc_clf_mask[:, clf_i]).astype(bool)
		curr_X = train_feature_vector[curr_mask]
		curr_Y = np.array(train_data.target)[curr_mask]
		curr_clf = MultinomialNB().fit(curr_X, curr_Y)
		curr_train_acc = np.mean(curr_clf.predict(curr_X) == curr_Y)
		clfs.append(curr_clf)
		clf_accs.append(curr_train_acc)

	print(clf_accs) # expect better performance during topic-specific training
	return clfs

def main():
	args = parser.parse_args()

	print('Loading data...')
	train_data, test_data = readData("")
	feature_type = args.vect
	ngram_range = (1, args.ngram)

	print('Creating topic model for the corpus...')
	topic_model, features, vectors = createTopicModel(train_data, feature_type, ngram_range, args.num_feat, args.topic, args.num_topic)
	if args.display_topics: 
		num_top_words = 10 # display 10 top words from extracted topics
		display_topics(topic_model, features, num_top_words)

	print('Creating topic-specific classifiers...')
	# vectors, features = createFeatureVecForTopic(train_data, feature_type, ngram_range, args.num_feat, args.topic)
	doc_clf_mask = split_by_topic(args, topic_model, vectors)

	print('Training topic-specific classifiers...')
	clfs = train_main(args, doc_clf_mask, train_data, vectors)

	# print('Training model...')
	# train_acc, train_count_vect, clf = trainNB(train_data, feature_type, ngram_range)
	# test_acc = testNB_SVM(clf, train_count_vect, test_data, feature_type)
	# test_acc = testNB(clf, train_count_vect, test_data, feature_type)

	# train_acc_svm, train_count_vect_error, clf_svm = trainSVM(train_data, feature_type, ngram_range)
	# test_acc_svm = testNB_SVM(clf_svm, train_count_vect_error, test_data, feature_type)
	# print("SVM Training Accuracy:", train_acc_svm, "\nTesting Accuracy", test_acc_svm)
	# print("Training Accuracy:", train_acc,"\nTesting Accuracy",test_acc)

if __name__ == '__main__':
	main()



