from simple_model import NeuralNetwork
from simple_model import Predict
import numpy as np
from get_data import *
import pickle
import re
import glob
from underthesea import word_tokenize
from gensim import corpora, models
import numpy as np
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
# from gensim.mod/home/ngocmaiels import Phrases
# from gensim.models.phrases import Phraser

icon_pos = ["😀","😁","😂","😃","😄","😅","😆","😉","😊","😇","☺","😋","😍","😘","😗","😙","😚","😜","😝",":)",":D","O:)","3:)",";)",">:O",":*","<3","8-)","8|","(y)","❤","😌","v.v", "hehe"]
icon_neg = [":(",":P","o.O","-_-",">:(",":/",":poop:","3:)","😠","😭","😲","😔","😳","😵","😡","😒","😞","😂","😰","😥","😨","😪","😱","😏","😓","😢","😷"]
pattern_symbol = r'''([#~:;^<>.\\\!?,'/()])123456789-'''
regex_symbol = re.compile(pattern_symbol, re.IGNORECASE + re.VERBOSE)

titles_domain = ['Phone', 'Food', 'Clothes']

def load_domain(title_domain):
    content_pos = []
    content_neg = []
    content = []
    label = []
    files_pos = './data/Domain/%s/pos.txt' %title_domain
    files_neg = './data/Domain/%s/neg.txt' %title_domain
    # files_pos = glob.glob(path_pos)
    # files_neg = glob.glob(path_neg)
    data_pos = read_txt(files_pos)
    data_pos = data_pos.lower()
    data_neg = read_txt(files_neg)
    data_neg = data_neg.lower()
    for row in data_pos.split("\n\n"):
        text_line = row.replace('__label__compliment , ', '')
        text_line = text_line.replace('user_name', '')
        text_line = re.sub("[1-9]", "", text_line)
        # text_line = text_line.replace('ạ', '')
        # text_line = text_line.replace('ơi', '')
        text_line = text_line.replace(' , ', ' ')
        text_line = text_line.replace(' . ', ' ')
        text_line = text_line.strip()
        # content_pos.append(text_line)
        content.append(text_line)
        label.append(1)
    for row in data_neg.split("\n\n"):
        text_line = row.replace('__label__decry , ','')
        text_line = text_line.replace('user_name', '')
        text_line = re.sub("[1-9]", "", text_line)
        # text_line = text_line.replace('ạ', '')
        # text_line = text_line.replace('ơi', '')
        text_line = text_line.replace(' , ', ' ')
        text_line = text_line.replace(' . ', ' ')
        text_line = text_line.strip()
        # content_neg.append(text_line)
        content.append(text_line)
        label.append(0)
    return (content, label)

def Load_json():
    data = read_json('../Data/Domain/Food/data_food_2.json')
    # print(data)


def clean_data(data):
    clean_data = []
    for line in data:
        line = regex_symbol.sub('', line)
        line = ' '.join(line.split())
        line = word_tokenize(line)
        for word in line:
            if len(word) <= 2:
                line.remove(word)
        clean_data.append(line)
    return clean_data

def k_gram(clean_data, k):
    array_k_folder = []

    for i in range(len(clean_data)):
        new_row = []
        # for j in range(len(array_1_folder[i]))[0:(len(array_1_folder[i])):k]:
        for j in range(len(clean_data[i])-k+1):
            mul_word = clean_data[i][j:(j+k)]
            new_mul_word = '_'.join(mul_word)
            new_row.append(new_mul_word)
        array_k_folder.append(new_row)
    return array_k_folder

def tfidfModel(array_k_folder):
    dictionary = corpora.Dictionary(array_k_folder)
    temp_corpus_bow = [dictionary.doc2bow(line) for line in array_k_folder]
    print("temp_corpus_bow", temp_corpus_bow)
    tfidf = models.TfidfModel(temp_corpus_bow)
    temp_corpus_tfidf = tfidf[temp_corpus_bow]

    content_tfidf = np.zeros((len(array_k_folder), len(dictionary.keys())))
    dict_tfidf = dict.fromkeys(dictionary, 0)

    # extract_word = dict.fromkeys(dictionary.keys(), 0)
    extract_word = {}.fromkeys('', 0)
    for i in range(len(temp_corpus_tfidf)):
        for item in temp_corpus_tfidf[i]:
            content_tfidf[i][item[0]] = item[1]
            dict_tfidf[item[0]] = item[1]
    for key, value in sorted(dict_tfidf.items(), key=lambda item: item[1]):
        extract_word[key] = value
        # if value > 0.568:
        #     extract_word[key] = value
    return(content_tfidf, extract_word, dictionary)

def vecttest(text):
    clean_text = clean_data([text])
    bi_gram = k_gram(clean_text, 2)
    three_gram = k_gram(clean_text, 3)

    vect_text = clean_text[0]
    # vect_text = clean_text[0] + bi_gram[0]
    # vect_text = clean_text[0] + bi_gram[0] + three_gram[0]

    return vect_text

def tf_idf_test(vect_test, dict):
    temp_corpus_bow = [dict.doc2bow(vect_test[0])]
    print("temp_corpus", temp_corpus_bow)
    tfidf = models.TfidfModel(temp_corpus_bow)
    temp_corpus_tfidf = tfidf[temp_corpus_bow]

    content_tfidf = np.zeros((len(vect_test), len(dict.keys())))
    for i in range(len(temp_corpus_tfidf)):
        for item in temp_corpus_tfidf[i]:
            content_tfidf[i][item[0]] = item[1]
    return content_tfidf

# Domain = Phone
# data_C = load_domain("Clothes")
# print(data_C)
if __name__ == "__main__":
    data_domain = load_domain("Phone")
    data = data_domain[0]
    targe = data_domain[1]
    clean_text = clean_data(data)
    data_bi_gram = k_gram(clean_text,2)
    data_tree_gram = k_gram(clean_text,3)
    data_1feature = clean_text
    tf_idf_data = tfidfModel(data_1feature)
    X_train = tf_idf_data[0]
    print("x_train", X_train.shape)
    dictionary = tf_idf_data[2]
    y_train = np.array(targe)

    text = "Hàng kém quá"
    text = text.lower()
    vectext = [vecttest(text=text)]
    # print(type(vectext), vectext)
    # print(dictionary.doc2bow(['hàng', 'cty', 'cấp', 'cấp', 'xong', 'vứt', 'nhà', 'xài', 'con', 'xornet', 'cho', 'mỏi', 'tay']))
    y_test = tf_idf_test(vectext, dictionary)

    for i in range(30000):
        neural_network = NeuralNetwork(X_train, y_train)
    w1 = neural_network.weights1
    w2 = neural_network.weights2

    test_predict = Predict(y_test, w1, w2)
    print(test_predict.predict())


