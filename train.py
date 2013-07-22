#!/usr/bin/env python
from json import loads
from json import dumps
from sklearn.svm import LinearSVC
from sklearn.feature_extraction import DictVectorizer

f = open('Data/answered_data_10k.in')
f2 = open('Data/answered_data_10k.out')
N = int(f.readline())
training_data_list = []
test_data_list = []
test_out_list = []

for i in xrange(N):
    json_string = f.readline()
    training_data_list.append(loads(json_string))

T = int(f.readline())
for i in xrange(T):
    json_string = f.readline()
    test_data_list.append(loads(json_string))

for i in xrange(T):
    json_string = f2.readline()
    test_out_list.append(loads(json_string))

f.close()
f2.close()
features = []
label = []
features_new = []
label_new = []
for train_ele in training_data_list:
    features.append({'sum':sum([x['followers'] for x in \
                                    train_ele['topics']])})
    label.append({'__ans__':1 if train_ele['__ans__'] == True else 0})

for test_ele, test_out in zip(test_data_list, test_out_list):
    features_new.append({'sum':sum(x['followers'] for x in \
                                       test_ele['topics'])})
    label_new.append({'__ans__':1 if test_out['__ans__'] == \
                          True else 0})

vec = DictVectorizer()
X = vec.fit_transform(features).toarray()
y = vec.fit_transform(label).toarray()
X_new = vec.fit_transform(features_new).toarray()
y_new = vec.fit_transform(label_new).toarray()

clf = LinearSVC()
clf = clf.fit(X, y)
print clf.score(X_new, y_new)
