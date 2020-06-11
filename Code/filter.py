import pandas as pd 
from sklearn.utils import shuffle
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
import math
import re
from math import log
from tqdm import trange
from sklearn.metrics import f1_score, recall_score, precision_score, accuracy_score
from core_function import *
from sklearn.model_selection import RepeatedStratifiedKFold

# Get prepared with data
email_df_all = pd.read_csv('emails_info_ascii.csv')


in_use_proportion = 0.1 # the proportion of data among the whole dataset
email_df_all = shuffle(email_df_all) # the random number generator is the RandomState instance used by np.random, see more at https://scikit-learn.org/stable/modules/generated/sklearn.utils.shuffle.html?highlight=shuffle#sklearn.utils.shuffle
email_df = email_df_all[0:int(in_use_proportion * len(email_df_all))]
email_df.fillna('', inplace = True)

folds, repeats = 5, 5
rskf = RepeatedStratifiedKFold(n_splits = folds, n_repeats = repeats) # the random number generator is the RandomState instance used by np.random, see more at https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.RepeatedStratifiedKFold.html?highlight=stratified#sklearn.model_selection.RepeatedStratifiedKFold
iteration = 0
result = pd.DataFrame(columns=['Accuracy', 'FPR', 'Precision', 'Recall', 'F1'])

# get english words list
with open('words.txt') as f:
    useful_words = f.read().split(' ')

# cross validation with stratified sampling
for train_index, test_index in rskf.split(email_df.values, email_df['label']):
    fold = (iteration % folds) + 1
    repeat = int(iteration / folds) + 1
    iteration += 1

    train_df = email_df.iloc[train_index]
    test_df = email_df.iloc[test_index]
    mail_count = len(train_df)
    print('repeat = %d of %d, fold = %d of %d' % (repeat, repeats, fold, folds))
    print("training set size : ", len(train_df))
    print("test set size : ", len(test_df))

    spam_df = train_df[train_df['label'] == 1]
    ham_df = train_df[train_df['label'] == 0]

    spam_content = ' '.join(spam_df['content'].tolist())
    ham_content = ' '.join(ham_df['content'].tolist())
    spam_subject = ' '.join(spam_df['subject'].tolist())
    ham_subject = ' '.join(ham_df['subject'].tolist())

    # get the feature words of content and subject
    content_feature_words = get_feature_words([spam_content, ham_content], vocabulary=useful_words)
    subject_feature_words = get_feature_words([spam_subject, ham_subject], vocabulary=useful_words)

    # storing P(feature_word | spam) and P(feature_word | ham) from mail text
    spam_content_dict = get_feature_dict(content_feature_words, spam_df['content'].tolist())
    ham_content_dict = get_feature_dict(content_feature_words, ham_df['content'].tolist())

    # storing P(feature_word | spam) and P(feature_word | ham) from mail suject
    spam_subject_dict = get_feature_dict(subject_feature_words, spam_df['subject'].tolist())
    ham_subject_dict = get_feature_dict(subject_feature_words, ham_df['subject'].tolist())

    # get from address feature
    spam_from_dict, ham_from_dict = get_addr_feature(spam_df['from'].tolist(), ham_df['from'].tolist())
    
    # get to address feature
    spam_to_dict, ham_to_dict = get_addr_feature(spam_df['to'].tolist(), ham_df['to'].tolist())

    # the prior probability for spam and ham
    train_spam_prob = train_df['label'].value_counts()[1] / len(train_df)
    train_ham_prob = train_df['label'].value_counts()[0] / len(train_df)

    # get prepared with test data
    test_size = len(test_df)
    test_from = test_df['from'].tolist()
    test_to = test_df['to'].tolist()
    test_content = test_df['content'].tolist()
    test_subject = test_df['subject'].tolist()
    test_label = test_df['label'].tolist()

    train_spam_count = train_df['label'].value_counts()[1]
    train_ham_count = train_df['label'].value_counts()[0]


    predict_labels = list()
    true_labels = test_label
    print("testing repeat %d of %d fold %d of %d ..." % (repeat, repeats, fold, folds))
    
    smooth_constant = 1e-50
    for i in trange(test_size):

        # use log probability to avoid underflow
        spam_prob = log(train_spam_prob)
        ham_prob = log(train_ham_prob)

        # calculate spam_prob
        spam_prob += check_text_prob(spam_content_dict, test_content[i].lower(), train_spam_count, smooth_constant)
        spam_prob += check_text_prob(spam_subject_dict, test_subject[i].lower(), train_spam_count, smooth_constant)
        spam_prob += check_addr(spam_from_dict, test_from[i])
        spam_prob += check_addr(spam_to_dict, test_to[i])

        # calculate ham_prob
        ham_prob += check_text_prob(ham_content_dict, test_content[i].lower(), train_ham_count, smooth_constant)
        ham_prob += check_text_prob(ham_subject_dict, test_subject[i].lower(), train_ham_count, smooth_constant)
        ham_prob += check_addr(ham_from_dict, test_from[i])
        ham_prob += check_addr(ham_to_dict, test_to[i])
        
        # if spam probability > ham probability, we believe it's a spam email, then give 1 to the predicted result, vice versa.
        predict = (spam_prob > ham_prob)
        predict_labels.append(predict)

    # calculate the metrics
    f1 = f1_score(true_labels, predict_labels, average='binary')
    recall = recall_score(true_labels, predict_labels, average='binary')
    precision = precision_score(true_labels, predict_labels, average='binary')
    accuracy = accuracy_score(true_labels, predict_labels)


    decimal = 4
    local_result = pd.DataFrame({'Accuracy':round(accuracy, decimal) ,\
    'FPR':round((1 - precision), decimal), 'Precision':round(precision, decimal),\
    'Recall': round(recall, decimal),'F1': round(f1, decimal)}, \
    index = ['fold %d repeat %d'%(fold, repeat)])

    # append now result to the dataframe and print it every iteration
    result = result.append(local_result)
    print(result)

# add max, min, average metrics to the result and print it
max_result, min_result, avg_result = result.max(), result.min(), result.mean()
max_result.name, min_result.name, avg_result.name = 'max', 'min', 'average'
result = result.append(max_result)
result = result.append(min_result)
result = result.append(avg_result)
print(result)
