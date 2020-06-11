from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd 
from sklearn.utils import shuffle
import numpy as np
import math
import re
from math import log
from tqdm import trange
from tqdm import tqdm

# return feature words from the corpus, selected by occuring counts and relative entropy
def get_feature_words(corpus_list, stop_words = 'english', lowercase = True, max_features = 5000, top_entropy = 1000, vocabulary = None):
    '''
    usage:
    return the minimum cross entropy words list in the corpus_list
    '''
    # use sklearn Vectorizer to tokenlize and count the words
    vectorizer = CountVectorizer(stop_words=stop_words, lowercase=lowercase, vocabulary=vocabulary)
    count_matrix = vectorizer.fit_transform(corpus_list).toarray()
    word_list = vectorizer.get_feature_names()
    
    spam_count = count_matrix[0]
    ham_count = count_matrix[1]
    word_count_list = list(zip(word_list, spam_count, ham_count))

    # sort by the occurrence counts in spam and ham, which may get the most representative words
    word_count_list.sort(key=lambda x: (x[1] + x[2]), reverse = True)
    word_count_list = word_count_list[0:max_features]
    word_new_list = [x[0] for x in word_count_list]

    # calculate the relative entropy(modified by renyi), which can measure the significance of the feature words to some extent
    entropy_list = list()
    for x in word_count_list:
        p1 = (x[1] + 1) / (len(word_count_list) + 2) # occurring probabability in the spam
        p2 = (x[2] + 1) / (len(word_count_list) + 2) # occurring probabability in the ham
        entropy_list.append(-abs(math.log2(p1/p2)))
    
    word_entropy_list = list(zip(word_new_list, entropy_list))
    word_entropy_list.sort(key=lambda x:x[1]) # sort by modified relative entropy, list by the significance of the feature words
    return [x[0] for x in word_entropy_list[0:top_entropy]]


# get the conditional probability dictionary for the feature words selected before
def get_feature_dict(features_list, corpus_list, stop_words = 'english', lowercase = True):
    # use sklearn CountVectorizer to count the occurence of those feature words in the mails
    vectorizer = CountVectorizer(stop_words = 'english', lowercase=lowercase, vocabulary=features_list)
    count_matrix = vectorizer.fit_transform(corpus_list).toarray()
    mail_count = len(count_matrix)
    prob_list = list()
    print("getting probability dict...")

    smooth = 1e-50 # smoothing constant for laplace smoothing
    for i in trange(len(features_list)):
        cnt = 0
        for j in range(len(corpus_list)):
            if count_matrix[j][i] > 0:
                cnt += 1
        # Laplace Smoothing, avoid 0 probability
        # prob_list.append((cnt  + smooth) / (mail_count + 2 * smooth))
        prob_list.append((cnt) / (mail_count))
        
    return dict(zip(features_list, prob_list))


# start the Naive Bayer process, get the conditional probability of the feature words
def check_text_prob(target_dict, content, mail_count, smooth):
    result = 0
    # use sklearn Vectorizer to split the words
    vectorizer = CountVectorizer()
    try:
        vectorizer.fit_transform([content])
    except ValueError: # it means no valid words in the content
        return 0
    
    
    word_list = vectorizer.get_feature_names() # get the words in the content
    for word in word_list:
        if word in target_dict and target_dict[word] != 0:
            result += log(target_dict[word])
        else:
            result += log(smooth) # just add a small constant to smooth
            # result += log(smooth / mail_count + 2 * smooth) # or use laplace smoothing
    return result

# extract the domain of the email address using regex
def addr_extract(addr_list):
    pattern = re.compile(r'\w+[@]([a-zA-Z0-9_-]+\.[a-zA-Z0-9_\.-]+)')
    addr_extract = list()
    for x in addr_list:
        try:
            suffix = re.findall(pattern, x)
        except TypeError: # the email address is nan or some other invalid things
            suffix = []
        addr_extract.append(suffix)
    return addr_extract

# get the email address domain feature, and return the conditional probability dict for the feature
def get_addr_feature(spam_list, ham_list):
    # extract domain
    spam_extract = addr_extract(spam_list)
    ham_extract = addr_extract(ham_list)
    all_pieces = set() # contains all the features in the domain
    for x in spam_extract:
        if len(x) != 0:
            pieces = x[0].split('.')
            for p in pieces:
                all_pieces.add(p)
    for x in ham_extract:
        if len(x) != 0:
            pieces = x[0].split('.')
            for p in pieces:
                all_pieces.add(p)
    spam_dict = dict()
    ham_dict = dict()
    print("getting address probability dict...")
    # calculate the conditional probability for the feature doumain
    for _, x in enumerate(tqdm(all_pieces)):
        spam_cnt = 0
        ham_cnt = 0
        for i in spam_extract:
            if len(i) != 0:
                if x in i[0]:
                    spam_cnt += 1
        for i in ham_extract:
            if len(i) != 0:
                if x in i[0]:
                    ham_cnt += 1
        spam_dict[x] = spam_cnt / len(spam_list)
        ham_dict[x] = ham_cnt / len(ham_list)
    return spam_dict, ham_dict

# process of naive bayes in email address, check the address conditional probability
def check_addr(target_dict, addr):
    pattern = re.compile(r'\w+[@]([a-zA-Z0-9_-]+\.[a-zA-Z0-9_\.-]+)')
    result = 0
    try:
        suffix = re.findall(pattern, addr)
    except TypeError:
        suffix = []
    if len(suffix) != 0:
        pieces = suffix[0].split('.')
        for p in pieces:
            if p in target_dict and target_dict[p] != 0:
                result += log(target_dict[p])
            else:
                result += log(1e-50)
    else:
        return log(1e-50)
    return result
