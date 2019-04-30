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

from utils import *
from train_val_split import load

from experiments_for_final_report import readData, createTopicModel, split_by_topic, createFeatureVecForTopic, baseline_train_n_test

'''
This script runs final experiments for our 701 project, using cross-validated
hyperparameters, training classifiers on the entire training set, and reporting 
accuracy on the testing set that was untouched during cross validation.
'''

'''
feature_type = 'bow': regular trigram + chop off max and min freq
feature_type = 'better_bow': trigram + rm 'br' and lemmatize (cleaned trigram) + max 1000K features
feature_type = 'custom_bow': cleaned trigram + dependencies + max 1000K features
feature_type = 'better_custom': cleaned trigram + dependencies + max 1000K features? + chop off max and min freq

'''

train_data, test_data = readData("")
ngram_range = (1, 3)

def create_topic_model(vect_topic):
    train_data, test_data = readData("")
    ngram_range = (1, 3)

    topic_model, features, vectors, vectorizer = createTopicModel(train_data, vect_topic, ngram_range, 0, 'LDA', 10)
    pickle_out = open('final_experiments/saved_topic/traintopic_{}'.format(vect_topic), 'wb')
    pickle.dump((topic_model, features, vectors, vectorizer), pickle_out)

    print('Done!')

# create_topic_model('bow')
# create_topic_model('better_bow')

def train_main(doc_clf_mask, train_data, train_feature_vector):
    '''
    Note:
    - train_data.data is a list of strings
    '''
    num_topics = 10

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

        print('Classifier NO.{}: {} samples, {} acc.'.format(clf_i + 1,sum(curr_mask),curr_train_acc))
    return clfs, clf_accs

def train(vect_topic, vect_clfr):
    train_data, test_data = readData("")
    ngram_range = (1, 3)

    num_topic = 10
    num_top_topics = 2

    pickle_in = open('final_experiments/saved_topic/traintopic_{}'.format(vect_topic), 'rb')
    topic_model, features, vectors, vectorizer = pickle.load(pickle_in)
    # split using vectors for topic modelling
    doc_clf_mask = split_by_topic(num_topic, num_top_topics, topic_model, vectors)
    # now create vectors for training 
    vectors, features, vectorizer = createFeatureVecForTopic(train_data, vect_clfr, ngram_range, 0, 'LDA')
    clfs, clf_accs = train_main(doc_clf_mask, train_data, vectors)
    pickle_out = open('final_experiments/saved_model/trained_clfrs_{}_{}'.format(vect_topic, vect_clfr), 'wb')
    pickle.dump((clfs, clf_accs, vectorizer), pickle_out)
    print('Done!')

# vect_topics = ['bow', 'better_bow']
# vect_clfrs = ['bow', 'better_bow', 'custom_bow', 'better_custom']

# for i in range(len(vect_topics)):
#     for j in range(len(vect_clfrs)):
#         train(vect_topics[i], vect_clfrs[j])

def test_main(clfs, test_vectors_topic, test_vectors_clf, topic_model, test_labels):
    doc_topic_distr = topic_model.transform(test_vectors_topic)

    num_samples = test_vectors_topic.shape[0]
    num_topics = len(clfs)

    all_preds = np.zeros((num_samples, num_topics))
    for clf_i, clf in enumerate(clfs):
        all_preds[:, clf_i] = clf.predict(test_vectors_clf)

    weighted_preds = (np.sum(doc_topic_distr * all_preds, axis = 1) > 0.5).astype(int)
    test_acc = np.mean(weighted_preds == test_labels)
    return test_acc

def createFeatureVecForTopic_test(dataset, feature_type):
    if feature_type == 'bow':
        vectorizer = CountVectorizer(max_df=0.95, min_df=2, max_features=None, stop_words='english', ngram_range = ngram_range)
        vectorizer.fit(train_data.data)
        vectors = vectorizer.transform(test_data.data)

    elif feature_type == 'better_bow':
        with open('custom_vector_test_nodep.pickle', 'rb') as f:
            vectors = pickle.load(f)

    elif feature_type == 'custom_bow':
        with open('custom_vector_test.pickle', 'rb') as f:
            vectors = pickle.load(f)

    elif feature_type == 'better_custom':
        with open('vector-3-0.95-2-dep-True-test.pickle', 'rb') as f:
            vectors = pickle.load(f)

    return vectors

def generate_all_relevant_vectorization():
    all_vect = ['bow', 'better_bow', 'custom_bow', 'better_custom']
    for V in all_vect:
        train = createFeatureVecForTopic(train_data, V, (1,3), 0, 'LDA')[0]
        pickle_out = open('final_experiments/saved_vectors/train_{}'.format(V), 'wb')
        pickle.dump(train, pickle_out)
        test = createFeatureVecForTopic_test(test_data, V)
        pickle_out = open('final_experiments/saved_vectors/test_{}'.format(V), 'wb')
        pickle.dump(test, pickle_out)
        

# generate_all_relevant_vectorization()

def test(vect_topic, vect_clfr):

    pickle_in = open('final_experiments/saved_vectors/train_{}'.format(vect_topic), 'rb')
    train_vectors_topic = pickle.load(pickle_in)

    pickle_in = open('final_experiments/saved_vectors/train_{}'.format(vect_clfr), 'rb')
    train_vectors_clf = pickle.load(pickle_in)

    pickle_in = open('final_experiments/saved_vectors/test_{}'.format(vect_topic), 'rb')
    test_vectors_topic = pickle.load(pickle_in)

    pickle_in = open('final_experiments/saved_vectors/test_{}'.format(vect_clfr), 'rb')
    test_vectors_clf = pickle.load(pickle_in)

    # first run baseline (single clfr using vect_clfr vectorization)
    baseline_acc = baseline_train_n_test(train_vectors_clf, train_data.target, test_vectors_clf, test_data.target, 'NB')
    print('Baseline accuracy (single clfr + vect_clfr: {}'.format(baseline_acc))

    # then test our proposed methods
    # load previously generated topic model and classifiers
    pickle_in = open('final_experiments/saved_topic/traintopic_{}'.format(vect_topic), 'rb')
    topic_model = pickle.load(pickle_in)[0]

    pickle_in = open('final_experiments/saved_model/trained_clfrs_{}_{}'.format(vect_topic, vect_clfr), 'rb')
    clfs = pickle.load(pickle_in)[0]

    train_acc = test_main(clfs, train_vectors_topic, train_vectors_clf, topic_model, train_data.target)
    print('Training accuracy: {}'.format(train_acc))
    test_acc = test_main(clfs, test_vectors_topic, test_vectors_clf, topic_model, test_data.target)
    print('Testing accuracy: {}'.format(test_acc))

# vect_topics = ['bow', 'better_bow']
# vect_clfrs = ['bow', 'better_bow', 'custom_bow', 'better_custom']

# for i in range(len(vect_topics)):
#     for j in range(len(vect_clfrs)):
#         print('==================================')
#         print('Testing combination {} for topic modelling + {} for classification.'.format(vect_topics[i], vect_clfrs[j]))
#         print('==================================')
#         test(vect_topics[i], vect_clfrs[j])















