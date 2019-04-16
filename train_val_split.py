
import random, pickle
import numpy as np
import sklearn
from sklearn.datasets import load_files

def split():
    total = 25000
    all_ids = list(range(total))
    random.shuffle(all_ids)

    num_fold = 5
    num_samples_per_fold = total//num_fold

    for i in range(num_fold):
        start_i = i * num_samples_per_fold
        end_i = min(total, num_samples_per_fold * (i + 1))
        test_ids = np.array(all_ids[start_i:end_i])
        mask = np.zeros(total, dtype = bool)
        mask[test_ids] = True
        pickle_out = open('cross_validation/train_val_split_%i.pickle' % (i+1), 'wb')
        pickle.dump(mask, pickle_out)

def load(test_fold_id):
    pickle_in = open('cross_validation/train_val_split_%i.pickle' % test_fold_id, 'rb')
    mask = pickle.load(pickle_in)
    category = ["pos","neg"]
    all_train = load_files("aclImdb/train", categories=category)

    test_x = np.array(all_train.data)[mask]
    test_y = np.array(all_train.target)[mask]

    train_x = np.array(all_train.data)[~mask]
    train_y = np.array(all_train.target)[~mask]

    movie_test = sklearn.datasets.base.Bunch(data=test_x, target=test_y)
    movie_train = sklearn.datasets.base.Bunch(data=train_x, target=train_y)

    return movie_test, movie_train

if __name__ == '__main__':
    split()

