import numpy as np
import argparse, pickle
from baseline_bigram_topic import standardize_topic_distr

parser = argparse.ArgumentParser(description = 'Movie Review Sentiment Analysis')
parser.add_argument('--vect', default = 'tf', type = str, help = 'Specify vectorization method.')
parser.add_argument('--ngram', default = 3, type = int, help = 'Specify ngram range.')
parser.add_argument('--topic', default = 'LDA', type = str, help = 'Specify topic model type.')
parser.add_argument('--clf', default = 'NB', type = str, help = 'Specify choice of classifier.')
parser.add_argument('--num_feat', default = 5000, type = int)
parser.add_argument('--num_topic', default = 20, type = int)
parser.add_argument('--display_topics', default = True, type = bool)
parser.add_argument('--num_top_topics', default = 5, type = int) 
parser.add_argument('--display_features', default = False, type = bool)
parser.add_argument('--test_style', default = 'R', type = str)
parser.add_argument('--load_prev_model', default = False, type = bool)
parser.add_argument('--load_prev_clf', default = False, type = bool)

def main():
    args = parser.parse_args()

    # print('Loading data...')
    # train_data, test_data = readData("")
    # feature_type = args.vect
    # ngram_range = (1, args.ngram)

    print('Loading previous topic model...')
    pickle_in = open('saved_model/%s_%s_%i_%i' % (args.topic, args.vect, args.ngram, args.num_topic), 'rb')
    topic_model, features, vectors, vectorizer = pickle.load(pickle_in)

    doc_topic_distr = topic_model.transform(vectors[:10])

    print(doc_topic_distr[:1])
    print('\n')
    doc_topic_distr = standardize_topic_distr(doc_topic_distr) # standardize vertically to account for rare topics
    doc_topic_distr = doc_topic_distr / np.sum(doc_topic_distr, axis = 1).reshape(-1, 1)

    print(doc_topic_distr[:1])

if __name__ == '__main__':
    main()