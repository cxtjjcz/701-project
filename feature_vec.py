import os, argparse, re, pickle, spacy, json
import numpy as np
import sklearn
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.datasets import load_files
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from scipy import sparse


parser = argparse.ArgumentParser(description = 'Creating Cusotmized Feature Vector')
parser.add_argument('--vect', default = 'bow', type = str, help = 'Specify vectorization method.')
parser.add_argument('--dep', default = False, type =bool, help='Add dependency relationship?')
parser.add_argument('--ngram', default = 3, type = int, help = 'Specify ngram range.')
parser.add_argument('--num_feat', default = 1000000, type = int, help='Specify the maximum number of features')
parser.add_argument('--verbose',default = False, type=bool, help="Print progress?")
parser.add_argument('--maxdf',default = 1, type=float, help="Maximum word frequency cap")
parser.add_argument('--mindf',default = 1, type=int, help="Lowest word presence in document")


# <-------------------------- Read Data ----------------------------> 
def readData(rootPath):
    category = ["pos","neg"]
    #load only labeled data
    movie_train = load_files(rootPath + "aclImdb/train", shuffle=True, categories=category)
    movie_test = load_files(rootPath + "aclImdb/test", shuffle=True, categories=category)
    return [movie_train, movie_test]

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
            doc = doc.decode("utf-8-sig")
            doc = re.sub(r'(\s*<br.*?>)+\s*', " ", doc)
            doc = re.sub("[^a-zA-Z]+", " ", doc)

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
            progress = CustomVectorizer.count/25000*100
            CustomVectorizer.count += 1
            if verbose:
                print (progress,"% processed")

            return(unigram_bigram)

        return(analyser)

class CustomVectorizer_WithDep(CountVectorizer): 
    count = 0
    # overwrite the build_analyzer method, allowing one to
    # create a custom analyzer for the vectorizer
    def build_analyzer(self):
        # load stop words using CountVectorizer's built in method
        stop_words = self.get_stop_words()
        # create the analyzer that will be returned by this method
        def analyser(doc):
            doc = doc.decode("utf-8-sig")
            doc = re.sub(r'(\s*<br.*?>)+\s*', " ", doc)
            doc = re.sub("[^a-zA-Z]+", " ", doc)

            # load spaCy's model for english language
            spacy.load('en')
            
            # instantiate a spaCy tokenizer
            lemmatizer = spacy.lang.en.English()
            
            # apply the preprocessing and tokenzation steps
            doc_clean = (doc).lower()
            tokens = lemmatizer(doc_clean)

            lemmatized_tokens = [token.lemma_ for token in tokens]
            
            # use CountVectorizer's _word_ngrams built in method
            # to remove stop words and extract n-grams

            # for that sentence, find the unigram and bigram
            unigram_bigram = self._word_ngrams(lemmatized_tokens, stop_words)

            count = CustomVectorizer.count
            lst_pair = dependency_dic[str(count)]

            for pair in lst_pair:
                temp_str = "" + str(pair[0]) + " " + str(pair[1])
                unigram_bigram.append(temp_str)

            progress = CustomVectorizer.count/25000*100
            CustomVectorizer.count += 1

            if verbose:
                print (progress,"% processed")
            return(unigram_bigram)

        return(analyser)


def main():
    args = parser.parse_args()
    print("Loading data...")
    train_data, test_data = readData("")
    global verbose
    verbose = args.verbose
    if args.dep:
        # <---------------------train data--------------------->
        global dependency_dic
        with open('train_data.json') as json_file:  
            dependency_dic = json.load(json_file)
        print("Creating customized feature vector on training data...")
        custom_vectorizer = CustomVectorizer_WithDep(ngram_range=(1,args.ngram),
                                stop_words='english',
                                encoding="utf-8-sig",
                                token_pattern=r"(?u)\b\w\w+\b",
                                max_df=args.maxdf, min_df=args.mindf, lowercase=True,
                                max_features=args.num_feat)
        custom_vector = custom_vectorizer.fit_transform(train_data.data)
        print("Dumping training data vector...")
        with open('vector-%s-%s-%s-dep-%s.pickle' % (args.ngram, args.maxdf,
                                                  args.mindf, args.dep), 'wb') as f:
            pickle.dump(custom_vector, f, pickle.HIGHEST_PROTOCOL)

        print("Dumping feature names...")
        with open('vector-%s-%s-%s-dep-%s-feature_name.pickle' % (args.ngram, args.maxdf,
                                                  args.mindf,
                                                  args.dep),"wb") as f:
            pickle.dump(custom_vectorizer.get_feature_names(), f, pickle.HIGHEST_PROTOCOL)
        # <---------------------test data--------------------->
        CustomVectorizer.count = 0
        with open('test_data.json') as json_file:  
            dependency_dic = json.load(json_file)
        print("Fitting customized feature vector on testing data...")
        test_vector = custom_vectorizer.transform(test_data.data)
        print("Dumping test data vector...")
        with open('vector-%s-%s-%s-dep-%s-test.pickle' % (args.ngram, args.maxdf,
                                                  args.mindf, args.dep), 'wb') as f:
            pickle.dump(test_vector, f, pickle.HIGHEST_PROTOCOL)
    else:
        # <---------------------train data--------------------->
        print("Creating customized feature vector on training data...")
        custom_vectorizer = CustomVectorizer(ngram_range=(1,args.ngram),
                                stop_words='english',
                                encoding="utf-8-sig",
                                token_pattern=r"(?u)\b\w\w+\b",
                                max_df=args.maxdf, min_df=args.mindf, lowercase=True,
                                max_features=args.num_feat)
        custom_vector = custom_vectorizer.fit_transform(train_data.data)
        print("Dumping training data vector...")
        with open('vector-%s-%s-%s-dep-%s.pickle' % (args.ngram, args.maxdf,
                                                  args.mindf, args.dep), 'wb') as f:
            pickle.dump(custom_vector, f, pickle.HIGHEST_PROTOCOL)

        print("Dumping feature names...")
        with open('vector-%s-%s-%s-dep-%s-feature_name.pickle' % (args.ngram, args.maxdf,
                                                  args.mindf, args.dep),"wb") as f:
            pickle.dump(custom_vectorizer.get_feature_names(), f, pickle.HIGHEST_PROTOCOL)
        # <---------------------test data--------------------->
        CustomVectorizer.count = 0
        print("Fitting customized feature vector on testing data...")
        test_vector = custom_vectorizer.transform(test_data.data)
        print("Dumping test data vector...")
        with open('vector-%s-%s-%s-dep-%s-test.pickle' % (args.ngram, args.maxdf,
                                                  args.mindf, args.dep), 'wb') as f:
            pickle.dump(test_vector, f, pickle.HIGHEST_PROTOCOL)




if __name__ == '__main__':
    main()

