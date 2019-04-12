import os, argparse
from sklearn.feature_extraction.text import CountVectorizer
import scipy
from nltk.parse import CoreNLPParser
from nltk.stem.porter import *
from nltk import tokenize
from nltk.parse.corenlp import CoreNLPDependencyParser
from sklearn.datasets import load_files
import inflect
import json

parser = argparse.ArgumentParser(description = 'Parse Reviews with Standford Parser')
parser.add_argument('--parse', default = False, type = bool, help = 'Parse?')
parser.add_argument('--parsed', default = False, type = bool, help = 'Already have parsed file?')

def readData(rootPath):
	category = ["pos","neg"]
	#load only labeled data
	movie_train = load_files(rootPath + "aclImdb/train", shuffle=True, categories=category)
	movie_test = load_files(rootPath + "aclImdb/test", shuffle=True, categories=category)
	return [movie_train, movie_test]

def parseReview():
	# Start server by running the following command under the parser directory:
	# java -mx4g -cp "*" edu.stanford.nlp.pipeline.StanfordCoreNLPServer
	dep_parser = CoreNLPDependencyParser(url='http://localhost:9000')
	dependency_relation = ["nsubj", "dobj", "agent", "advmod", "amod", 
							"neg", "prep_of", "acomp", "xcomp"]

	train_data, test_data = readData("")
	review_dependency_feature = dict()
	prev_percent = 0
	all_data = [train_data, test_data]
	names = ["train_data", "test_data"]

	for j in range(len(all_data)):
		dat = all_data[j]
		for i in range(len(dat.data)):
			review = tokenize.sent_tokenize(train_data.data[i].decode("utf-8"))
			review_feature = list()
			for line in review:
				parse, = dep_parser.raw_parse(line)
				for governor, dep, dependent in parse.triples():
					if dep in dependency_relation:
						review_feature.append((governor[0],dependent[0],dep))
			review_dependency_feature[i] = review_feature
			percent = int(i/len(train_data.data)*100)

			if percent == prev_percent + 1:
				prev_percent += 1
				print (percent,"% processed")


		with open(names[j]+'.json', 'w') as outfile:  
			json.dump(review_dependency_feature, outfile)

def cleanDependency(l):
	capWords = lambda l: list(map(lambda w: w.lower(),l))
	p = inflect.engine()

	def convertPluraltoSingular(w):
		result = p.singular_noun(w)
		if result == False:
			return w
		else:
			return result

	depluralize = lambda l: list(map(lambda w: convertPluraltoSingular(w),l))
	
	r = list(map(capWords, l))
	r = list(map(depluralize,r))
	return r


def furtherReduce():
	filenames = ["train_data.json","test_data.json"]
	for filename in filenames:
		with open(filename,"r") as json_file:  
		    data = json.load(json_file)
		for k in data.keys():
			data[k] = cleanDependency(data[k])

		with open(filename,"w") as json_file:  
			json.dump(data,json_file)

		with open(filename,"r") as json_file:  
		    data = json.load(json_file)

def main():
	args = parser.parse_args()
	if args.parse:
		parseReview()

	if args.parsed:
		furtherReduce()
	
	
if __name__ == '__main__':
	main()


