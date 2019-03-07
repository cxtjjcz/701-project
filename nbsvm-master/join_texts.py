import fileinput
import glob

file_list_pos_train = glob.glob("/Users/xiangtic/PycharmProjects/701-project/aclImdb/train/pos/*.txt")
file_list_neg_train = glob.glob("/Users/xiangtic/PycharmProjects/701-project/aclImdb/train/neg/*.txt")

file_list_pos_test = glob.glob("/Users/xiangtic/PycharmProjects/701-project/aclImdb/test/pos/*.txt")
file_list_neg_test = glob.glob("/Users/xiangtic/PycharmProjects/701-project/aclImdb/test/neg/*.txt")

with open('train_pos.txt', 'w') as file:
    input_lines = fileinput.input(file_list_pos_train)
    file.writelines(input_lines)

with open('train_neg.txt', 'w') as file:
    input_lines = fileinput.input(file_list_neg_train)
    file.writelines(input_lines)

with open('test_pos.txt', 'w') as file:
    input_lines = fileinput.input(file_list_pos_test)
    file.writelines(input_lines)

with open('test_neg.txt', 'w') as file:
    input_lines = fileinput.input(file_list_neg_test)
    file.writelines(input_lines)