#!/usr/bin/env python
from json import loads
from json import dumps
from sklearn.svm import LinearSVC
from sklearn.feature_extraction import DictVectorizer

f = open('Data/answered_data_10k.in')
N = int(f.readline())
training_data_list = []
test_data_list = []

for i in xrange(N):
    json_string = f.readline()
    training_data_list.append(loads(json_string))

T = int(f.readline())
for i in xrange(T):
    json_string = f.readline()
    test_data_list.append(loads(json_string))
    
features = []
label = []
features_new = []
for train_ele in training_data_list:
    features.append({'sum':sum([x['followers'] for x in \
                                    train_ele['topics']])})
    label.append({'__ans__':1 if train_ele['__ans__'] == True else 0})

for test_ele in test_data_list:
    features_new.append({'sum':sum(x['followers'] for x in \
                                       test_ele['topics'])})

vec = DictVectorizer()
X = vec.fit_transform(features).toarray()
y = vec.fit_transform(label).toarray()
X_new = vec.fit_transform(features_new).toarray()

clf = LinearSVC()
clf = clf.fit(X, y)
y_new = clf.predict(X_new)

out = []
for test_ele, l in zip(test_data_list, y_new):
    out.append({'__ans__':str(l==1), 'question_key':test_ele['question_key']})

for o in out:
    print dumps(o)
