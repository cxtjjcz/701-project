import os, argparse, pickle
import numpy as np
import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.datasets import load_files
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.decomposition import NMF, LatentDirichletAllocation
# import pandas as pd

from utils import *
from train_val_split import load

parser = argparse.ArgumentParser(description = 'Movie Review Sentiment Analysis')
parser.add_argument('--vect', default = 'tf', type = str, help = 'Specify vectorization method.')
parser.add_argument('--ngram', default = 3, type = int, help = 'Specify ngram range.')
parser.add_argument('--topic', default = 'LDA', type = str, help = 'Specify topic model type.')
parser.add_argument('--clf', default = 'NB', type = str, help = 'Specify choice of classifier.')
parser.add_argument('--num_feat', default = 5000, type = int)
parser.add_argument('--num_topic', default = 20, type = int)
parser.add_argument('--display_topics', default = True, type = bool)
parser.add_argument('--num_top_topics', default = 3, type = int) 
parser.add_argument('--display_features', default = False, type = bool)
parser.add_argument('--test_style', default = 'R', type = str)
parser.add_argument('--load_prev_model', default = False, type = bool)
parser.add_argument('--load_prev_clf', default = False, type = bool)
# An example is contained in the training sets of its top-5 most relevant topic-specific classifiers

# <-------------------------- Read Data ----------------------------> 

def readData(rootPath):
	category = ["pos","neg"]
	#load only labeled data
	movie_train = load_files(rootPath + "aclImdb/train", shuffle=True, categories=category)
	movie_test = load_files(rootPath + "aclImdb/test", shuffle=True, categories=category)
	return [movie_train, movie_test]

# <-------------------------- Vectorization ----------------------------> 

def createFeatureVecForTopic(dataset, feature_type, ngram_range, num_feat, topic):
	'''
	feature_type = 'bow': regular trigram + chop off max and min freq
	feature_type = 'better_bow': trigram + rm 'br' and lemmatize (cleaned trigram) + max 1000K features
	feature_type = 'custom_bow': cleaned trigram + dependencies + max 1000K features
	feature_type = 'better_custom': cleaned trigram + dependencies + max 1000K features? + chop off max and min freq

	'''

	if feature_type == 'bow':
		vectorizer = CountVectorizer(max_df=0.95, min_df=2, max_features=None, stop_words='english', ngram_range = ngram_range)
		vectors = vectorizer.fit_transform(dataset.data)
		features = vectorizer.get_feature_names()

	elif feature_type == 'better_bow':
		with open('custom_vector_nodep.pickle', 'rb') as f:
			vectors = pickle.load(f)
		with open('feature_names_nodep.pickle', 'rb') as f:
			features = pickle.load(f)
		vectorizer = None

	elif feature_type == 'custom_bow':
		with open('custom_vector.pickle', 'rb') as f:
			vectors = pickle.load(f)
		with open('feature_names.pickle', 'rb') as f:
			features = pickle.load(f)
		vectorizer = None

	elif feature_type == 'better_custom':
		with open('vector-3-0.95-2-dep-True.pickle', 'rb') as f:
			vectors = pickle.load(f)
		with open('vector-3-0.95-2-dep-True-feature_name.pickle', 'rb') as f:
			features = pickle.load(f)
		vectorizer = None

	return vectors, features, vectorizer

# <-------------------------- Topic modeling ----------------------------> 

def createTopicModel(dataset, feature_type, ngram_range, num_feat, topic, num_topic):
	# get document-word matrix (vectors) and total words (features)
	vectors, features, vectorizer = createFeatureVecForTopic(dataset, feature_type, ngram_range, num_feat, topic)
	# run topic model
	if topic == 'LDA':
		topic_model = LatentDirichletAllocation(n_components=num_topic, max_iter=5, learning_method='online', \
												learning_offset=50., random_state=0).fit(vectors)
	if topic == 'NMF':
		topic_model = NMF(n_components=num_topic, random_state=1, alpha=.1, l1_ratio=.5, init='nndsvd').fit(vectors)
		# 'nndsvd' initialization helps with convergence
	# topic_model = document-topic matrix + topic-word matrix (decomposition of a doc-word matrix)
	return topic_model, features, vectors, vectorizer

def split_by_topic(num_topic, num_top_topics, topic_model, train_X):
	'''
	Generates a matrix of size (num_samples, num_topics) in which X_i,j = 1 if doc i is used for clf j
	'''
	num_samples = train_X.shape[0]
	num_topics = num_topic
	num_top_topics = num_top_topics # a document will be included in the training set of this number of topics

	doc_clf_mask = np.zeros((num_samples, num_topics))
	doc_topic_distr = topic_model.transform(train_X) # num_samples x num_topics
	# doc_topic_distr = np.exp(standardize_topic_distr(doc_topic_distr))

	print("--- Iterating through training examples to find their topic mixtures...")

	for doc_i, doc in enumerate(doc_topic_distr):
		sorted_clf_idx = sorted(list(range(num_topics)), key=doc.__getitem__, reverse=True)[:num_top_topics]
		for i in sorted_clf_idx:
			doc_clf_mask[doc_i, i] = 1

	return doc_clf_mask

def standardize_topic_distr(doc_topic_distr):
	'''
	This function takes into consideration infrequent topics. The document-topic matrix
	is normalized vertically such that each entry represents the z-score within topic.

	'''
	means = np.mean(doc_topic_distr, axis = 0)
	stds = np.std(doc_topic_distr, axis = 0)
	eps = 1e-05
	return (doc_topic_distr - means) / (stds + eps)

# <-------------------------- Main functions ---------------------------->

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

		if args.clf == 'SVM':
			curr_clf = SVC(kernel="linear").fit(curr_X, curr_Y)
		elif args.clf == 'NB':
			curr_clf = MultinomialNB().fit(curr_X, curr_Y)

		curr_train_acc = np.mean(curr_clf.predict(curr_X) == curr_Y)
		clfs.append(curr_clf)
		clf_accs.append(curr_train_acc)

		print('Finish training the %i-th classifier.' % (clf_i + 1))
		print('Total number of training data seen by the current clfs: %i' % sum(curr_mask))

	print(clf_accs) # expect better performance during topic-specific training
	return clfs, clf_accs

def test_main_R(clfs, test_vectors, topic_model, test_labels, num_top_topics):
	'''
	Only use the most relevant topic-specific classifiers for each document
	'''
	doc_topic_distr = topic_model.transform(test_vectors)
	# doc_topic_distr = np.exp(standardize_topic_distr(doc_topic_distr)) # standardize vertically to account for rare topics
	# doc_topic_distr = standardize_topic_distr(doc_topic_distr)
	# doc_topic_distr = doc_topic_distr / np.sum(doc_topic_distr, axis = 1).reshape(-1, 1) # standardize horizontally to get a topic distribution per document

	# num_top_topics = 1

	num_samples = test_vectors.shape[0]
	num_topics = len(clfs)
	doc_clf_mask = np.zeros((num_samples, num_topics))
	for doc_i, doc in enumerate(doc_topic_distr):
		sorted_clf_idx = sorted(list(range(num_topics)), key=doc.__getitem__, reverse=True)[:num_top_topics]
		for i in sorted_clf_idx:
			doc_clf_mask[doc_i, i] = 1

	all_preds = np.zeros((num_samples, num_topics))
	for clf_i, clf in enumerate(clfs):
		all_preds[:, clf_i] = clf.predict(test_vectors)

	masked_weights = doc_topic_distr * doc_clf_mask

	print(np.sum(masked_weights < 0)) # total number of negative z-scores

	normalized_masked_weights = masked_weights/np.sum(masked_weights, axis = 1).reshape(-1, 1)
	weighted_preds = (np.sum(normalized_masked_weights * all_preds, axis = 1) > 0.5).astype(int)

	#weighted_preds = (np.sum(doc_topic_distr * all_preds, axis = 1) > 0.5).astype(int)
	print(weighted_preds)
	test_acc = np.mean(weighted_preds == test_labels)
	# print(test_acc)
	print("{:.10f}".format(test_acc))
	return test_acc

def test_main_P(clfs, clf_accs, test_vectors, topic_model, test_labels):
	num_samples = test_vectors.shape[0]
	num_topics = len(clfs)
	all_preds = np.zeros((num_samples, num_topics))
	for clf_i, clf in enumerate(clfs):
		all_preds[:, clf_i] = clf.predict(test_vectors)

	doc_topic_distr = topic_model.transform(test_vectors)
	clf_acc_thd = 0.9 # for now
	clf_mask = (np.array(clf_accs) > clf_acc_thd).reshape(1,-1) # only look at these clfs
	masked_weights = doc_topic_distr * clf_mask
	normalized_masked_weights = masked_weights/np.sum(masked_weights, axis = 1).reshape(-1, 1) # sum to one horizontally
	raw_weighted_preds = np.sum(normalized_masked_weights * all_preds, axis = 1)
	weighted_preds = (raw_weighted_preds > 0.5).astype(int)

	# for pred in raw_weighted_preds:
	# 	print(pred) # check how confused these clfs are

	test_acc = np.mean(weighted_preds == test_labels)
	# print(test_acc)
	print("{:.10f}".format(test_acc))
	return test_acc

def test_main_A(clfs, test_vectors, topic_model, test_labels, num_top_topics):
	doc_topic_distr = topic_model.transform(test_vectors)
	# doc_topic_distr = np.exp(standardize_topic_distr(doc_topic_distr)) # standardize vertically to account for rare topics
	# doc_topic_distr = doc_topic_distr / np.sum(doc_topic_distr, axis = 1).reshape(-1, 1) # standardize horizontally to get a topic distribution per document

	num_samples = test_vectors.shape[0]
	num_topics = len(clfs)

	all_preds = np.zeros((num_samples, num_topics))
	for clf_i, clf in enumerate(clfs):
		all_preds[:, clf_i] = clf.predict(test_vectors)

	weighted_preds = (np.sum(doc_topic_distr * all_preds, axis = 1) > 0.5).astype(int)

	# print(weighted_preds)
	test_acc = np.mean(weighted_preds == test_labels)
	print("{:.10f}".format(test_acc))
	return test_acc

def baseline_train_n_test(train_vectors, train_labels, test_vectors, test_labels, clf_type):
	if clf_type == 'NB':
		clf = MultinomialNB().fit(train_vectors,train_labels) # a single classifier trained on all data
	elif clf_type == 'SVM':
		clf = SVC(kernel="linear").fit(train_vectors, train_labels)
	acc = np.mean(clf.predict(test_vectors) == test_labels)
	# print("{:.10f}".format(acc))
	return acc

def main_helper(train_data, test_data, args, cv_id, cv = True):

	print('****************** Starting the %i-th fold validation ******************' % cv_id)
	feature_type = args.vect
	ngram_range = (1, args.ngram)

	try:
		pickle_in = open('saved_model/cvfold_%i_%s_%s_%i' % (cv_id, args.topic, args.vect, args.num_topic), 'rb')
		topic_model, features, vectors, vectorizer = pickle.load(pickle_in)
		print('Successfully loaded previous topic model...')
	except:
		print('Previous model does not exist. Creating topic model for the corpus...')
		topic_model, features, vectors, vectorizer = createTopicModel(train_data, feature_type, ngram_range, args.num_feat, args.topic, args.num_topic)
		pickle_out = open('saved_model/cvfold_%i_%s_%s_%i' % (cv_id, args.topic, args.vect, args.num_topic), 'wb')
		pickle.dump((topic_model, features, vectors, vectorizer), pickle_out)
	
	# if args.display_topics: 
	# 	num_top_words = 10 # display 10 top words from extracted topics
	# 	display_topics(topic_model, features, num_top_words)

	# if args.display_features:
	# 	# for i in features:
	# 	# 	print(i)
	# 	print(len(features))

	try:
		pickle_in = open('saved_clf/%s_cvfold_%i_%s_%s_%i_%i' % (feature_type, cv_id, args.topic, args.clf, args.num_topic, args.num_top_topics), 'rb')
		clfs, clf_accs, vectorizer = pickle.load(pickle_in)
		print('Successfully loaded previously trained classifiers...')
	except:
		print('Previous classifiers do not exist. Creating topic-specific classifiers...')
		# vectors, features = createFeatureVecForTopic(train_data, feature_type, ngram_range, args.num_feat, args.topic)
		doc_clf_mask = split_by_topic(args, topic_model, vectors)

		print('Training topic-specific classifiers...')
		clfs, clf_accs = train_main(args, doc_clf_mask, train_data, vectors)
		pickle_out = open('saved_clf/%s_cvfold_%i_%s_%s_%i_%i' % (feature_type, cv_id, args.topic, args.clf, args.num_topic, args.num_top_topics), 'wb')
		pickle.dump((clfs, clf_accs, vectorizer), pickle_out)

	print('Testing topic-specific classifiers...')

	if feature_type == "custom_bow":
		pickle_in = open('custom_vector_test.pickle', 'rb')
		test_vectors = pickle.load(pickle_in)

	elif feature_type == 'better_custom':
		pickle_in = open('vector-3-0.95-2-dep-True-test.pickle', 'rb')
		test_vectors = pickle.load(pickle_in)

	elif feature_type == "custom_tf":
		pickle_in = open('custom_vector_test.pickle', 'rb')
		test_vectors = pickle.load(pickle_in)
		tf_transformer = TfidfTransformer(use_idf=False).fit(test_vectors)
		test_vectors = tf_transformer.transform(test_vectors)
	elif feature_type == "custom_tfidf":
		pickle_in = open('custom_vector_test.pickle', 'rb')
		test_vectors = pickle.load(pickle_in)
		tfidf_transformer = TfidfTransformer(use_idf=True).fit(test_vectors)
		test_vectors = tfidf_transformer.transform(test_vectors)
	else:
		test_vectors = vectorizer.transform(test_data.data)

	if args.test_style == 'R':
		acc = test_main_R(clfs, test_vectors, topic_model, test_data.target, args.num_top_topics) # test based on topic relevance
	elif args.test_style == 'P':
		acc = test_main_P(clfs, clf_accs, test_vectors, topic_model, test_data.target) # test based on topic's ability to do sentiment classification
	elif args.test_style == 'A':
		acc = test_main_A(clfs, test_vectors, topic_model, test_data.target, args.num_top_topics)

	# baseline_acc = baseline_train_n_test(vectors, train_data.target, test_vectors, test_data.target, args.clf)
	baseline_acc = 0.87760
	return acc, baseline_acc

def main():
	args = parser.parse_args()
	num_fold = 5
	total_acc = 0
	total_b_acc = 0

	for i in range(num_fold):
		movie_test, movie_train = load(i+1)
		acc, b_acc = main_helper(movie_train,  movie_test, args, i+1)
		print("Fold = ", i, "Topic accuracy is %.5f, baseline acc is %.5f" % (acc, b_acc))
		total_acc += acc
		total_b_acc += b_acc

	ave_accuracy = total_acc / num_fold
	ave_b_acc = total_b_acc / num_fold
	print("Average accuracy: %.5f (topic), %.5f (baseline)" % (ave_accuracy, ave_b_acc))

def end_to_end(runid): # training set of size 25000; test set of size 25000
	if runid == None:
		runid = int(time.time())
		print('===========================================')
		print('Running experiments using cross validated hyperparameters, current experiment id = {}'.format(runid))
		print('Tokenization: Ngram = {}'.format(args.ngram))
		print('Vectorization: {} for topic modelling, {} for clfr training'.format(args.vect_topic, args.vect_clfr))
		print('Topic model: model = {}, num_topic = {}, num_top_topics = {}'.format(args.topic, args.num_topic, args.num_top_topics))
		print('Classifier: {}'.format(args.clf))
		print('===========================================')
	else:
		print('===========================================')
		print('Running (previous) experiment, id = {}'.format(runid))
		print('===========================================')

	args = parser.parse_args()
	print('[LOG] Loading data...')
	train_data, test_data = readData("")
	feature_type = args.vect
	ngram_range = (1, args.ngram)

	try:
		pickle_in = open('final_experiments/saved_model/crossvalidated_topicmodel_{}'.format(runid), 'rb')
		topic_model, features, vectors, vectorizer = pickle.load(pickle_in)
		print('Successfully loaded previous topic model...')
	except:
		print('Previous model does not exist. Creating topic model for the corpus...')
		topic_model, features, vectors, vectorizer = createTopicModel(train_data, feature_type, ngram_range, args.num_feat, args.topic, args.num_topic)
		pickle_out = open('final_experiments/saved_model/crossvalidated_topicmodel_{}'.format(runid), 'wb')
		pickle.dump((topic_model, features, vectors, vectorizer), pickle_out)
	
	# if args.display_topics: 
	# 	num_top_words = 10 # display 10 top words from extracted topics
	# 	display_topics(topic_model, features, num_top_words)

	# if args.display_features:
	# 	# for i in features:
	# 	# 	print(i)
	# 	print(len(features))

	try:
		pickle_in = open('saved_clf/end_to_end_%s_%s_%s_%i_%i' % (args.topic, feature_type, args.clf, args.num_topic, args.num_top_topics), 'rb')
		clfs, clf_accs, vectorizer = pickle.load(pickle_in)
		print('Successfully loaded previously trained classifiers...')
	except:
		assert(False)
		print('Previous classifiers do not exist. Creating topic-specific classifiers...')
		# vectors, features = createFeatureVecForTopic(train_data, feature_type, ngram_range, args.num_feat, args.topic)
		doc_clf_mask = split_by_topic(args, topic_model, vectors)

		print('Training topic-specific classifiers...')
		clfs, clf_accs = train_main(args, doc_clf_mask, train_data, vectors)
		pickle_out = open('saved_clf/end_to_end_%s_%s_%s_%i_%i' % (args.topic, feature_type, args.clf, args.num_topic, args.num_top_topics), 'wb')
		pickle.dump((clfs, clf_accs, vectorizer), pickle_out)

	print('Testing topic-specific classifiers...')
	if feature_type == "custom_bow":
		pickle_in = open('custom_vector_test.pickle', 'rb')
		test_vectors = pickle.load(pickle_in)

	elif feature_type == 'better_custom':
		pickle_in = open('vector-3-0.95-2-dep-True-test.pickle', 'rb')
		test_vectors = pickle.load(pickle_in)

	elif feature_type == "custom_tf":
		pickle_in = open('custom_vector_test.pickle', 'rb')
		test_vectors = pickle.load(pickle_in)
		tf_transformer = TfidfTransformer(use_idf=False).fit(test_vectors)
		test_vectors = tf_transformer.transform(test_vectors)
	elif feature_type == "custom_tfidf":
		pickle_in = open('custom_vector_test.pickle', 'rb')
		test_vectors = pickle.load(pickle_in)
		tfidf_transformer = TfidfTransformer(use_idf=True).fit(test_vectors)
		test_vectors = tfidf_transformer.transform(test_vectors)
	else:
		print('Using regular bow')

		test_vectors = vectorizer.transform(test_data.data)

	if args.test_style == 'R':
		acc = test_main_R(clfs, test_vectors, topic_model, test_data.target, args.num_top_topics) # test based on topic relevance
	elif args.test_style == 'P':
		acc = test_main_P(clfs, clf_accs, test_vectors, topic_model, test_data.target) # test based on topic's ability to do sentiment classification
	elif args.test_style == 'A':
		acc = test_main_A(clfs, test_vectors, topic_model, test_data.target, args.num_top_topics)

	baseline_acc = baseline_train_n_test(vectors, train_data.target, test_vectors, test_data.target, args.clf)
	# baseline_acc = 0.87760
	print('Acc with topic:', acc)
	print('Baseline acc:', baseline_acc)
	return acc

if __name__ == '__main__':
	main()
	# end_to_end()




