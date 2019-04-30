from sklearn.datasets import load_files
import random

def readData(rootPath = ''):
    category = ["pos","neg"]
    #load only labeled data
    movie_train = load_files(rootPath + "aclImdb/train", shuffle=True, categories=category)
    # movie_test = load_files(rootPath + "aclImdb/test", shuffle=True, categories=category)
    return movie_train

train = readData()
train_text = train.data
random.shuffle(train_text)

for doc in train_text:
    if 'recommend' in str(doc):
        print(doc)
        break

