'''
Iancu Florentina-Mihaela
grupa 361
'''

import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import re
from nltk.corpus import stopwords
from collections import Counter

import langid
langid.set_languages(['da', 'de', 'es', 'it', 'nl'])  # ISO 639-1 codes

# CITIREA SI PREPROCESAREA DATELOR DE ANTRENAMENT

data_path = 'C:/Users/Asus tuf/Desktop/University/ANUL3_S1/Inteligenta Artificiala'
train_data_df = pd.read_csv(os.path.join(data_path, 'train_data.csv'))

etichete_unice = train_data_df['label'].unique()
label2id = {}
id2label = {}
for idx, eticheta in enumerate(etichete_unice):
    label2id[eticheta]=idx
    id2label[idx]=eticheta

labels = []
for i in train_data_df['label']:
  labels.append(label2id[i])

def proceseaza(text):
    text_final = []
    text = re.sub("[-–|.,;:!?\"\'\/()_*=`¿¡«»\n0123456789]", "", text)
    language, score = langid.classify(text)
    
    if language == 'da':
        lang.append('danish')
    elif language == 'de':
        lang.append('german')
    elif language == 'es':
        lang.append('spanish')
    elif language == 'it':
        lang.append('italian')
    elif language == 'nl':
        lang.append('dutch')
    else:
        print("Eroare: Limba textului este gresita!")
        
    text_in_cuvinte = text.strip().split(' ')
    
    for i in range(len(text_in_cuvinte)):
        if text_in_cuvinte[i]!='' and text_in_cuvinte[i]!='  ':
            text_final.append(text_in_cuvinte[i].lower())
    
    return text_final

texts_da = train_data_df[train_data_df['language'] == 'dansk']
texts_ge = train_data_df[train_data_df['language'] == 'Deutsch']
texts_sp = train_data_df[train_data_df['language'] == 'español']
texts_it = train_data_df[train_data_df['language'] == 'italiano']
texts_ol = train_data_df[train_data_df['language'] == 'Nederlands']

nr_texte = len(texts_da)

#impartire pe limbi

lang = []

data_da = []
for text in texts_da['text']:
  data_da.append(proceseaza(text))

data_ge = []
for text in texts_ge['text']:
  data_ge.append(proceseaza(text))

data_sp = []
for text in texts_sp['text']:
  data_sp.append(proceseaza(text))

data_it = []
for text in texts_it['text']:
  data_it.append(proceseaza(text))

data_ol = []
for text in texts_ol['text']:
  data_ol.append(proceseaza(text))

#labels pentru fiecare limba

labels_da = []
labels_ge = []
labels_sp = []
labels_it = []
labels_ol = []

for i in range(len(labels)):
    if i < nr_texte:
        labels_da.append(labels[i])
    elif (i >= nr_texte) and (i < nr_texte*2):
        labels_ge.append(labels[i])
    elif (i >= nr_texte*2) and (i < nr_texte*3):
        labels_sp.append(labels[i])
    elif (i >= nr_texte*3) and (i < nr_texte*4):
        labels_it.append(labels[i])
    else:
        labels_ol.append(labels[i])

#lematizare

from simplemma import lemmatize

for i in range(len(data_da)):
    for j in range(len(data_da[i])):
        data_da[i][j] = lemmatize(data_da[i][j], lang='da').lower()

for i in range(len(data_ge)):
    for j in range(len(data_ge[i])):
        data_ge[i][j] = (lemmatize(data_ge[i][j], lang='de')).lower()
        
for i in range(len(data_sp)):
    for j in range(len(data_sp[i])):
        data_sp[i][j] = lemmatize(data_sp[i][j], lang='es').lower()
        
for i in range(len(data_it)):
    for j in range(len(data_it[i])):
        data_it[i][j] = lemmatize(data_it[i][j], lang='it').lower()
        
for i in range(len(data_ol)):
    for j in range(len(data_ol[i])):
        data_ol[i][j] = lemmatize(data_ol[i][j], lang='nl').lower()


stopwords_da = set(stopwords.words('danish'))
# new_words = ['europæisk','eu','parlament','hr','så','formand','ved','vores','kommission','kommissær','parlamentet','europæiske','eus','europa']
# stopwords_da.update(new_words)

def remove_stopwords_da(data):
    temp_list = []
    for word in data:
        if word not in stopwords_da:
            temp_list.append(word)
    return temp_list

stopwords_ge = set(stopwords.words('german'))
new_words = ['er|es|sie']
# new_words = ['er|es|sie','herr','europäisch','kommission','parlament','präsident','eu','europa']
stopwords_ge.update(new_words)

def remove_stopwords_ge(data):
    temp_list = []
    for word in data:
        if word not in stopwords_ge:
            temp_list.append(word)
    return temp_list

stopwords_sp = set(stopwords.words('spanish'))
# new_words = ['europeo','comisión','presidente','señor','parlamento','europa','ue']
# stopwords_sp.update(new_words)

def remove_stopwords_sp(data):
    temp_list = []
    for word in data:
        if word not in stopwords_sp:
            temp_list.append(word)
    return temp_list

stopwords_it = set(stopwords.words('italian'))
# new_words = ['ue','europa','governo','parlamento','presidente','commissione','europeo']
# stopwords_it.update(new_words)

def remove_stopwords_it(data):
    temp_list = []
    for word in data:
        if word not in stopwords_it:
            temp_list.append(word)
    return temp_list

stopwords_ol = set(stopwords.words('dutch'))
# new_words = ['europees','commissie','parlement','commissaris','europa','heer','eu']
# stopwords_ol.update(new_words)

def remove_stopwords_ol(data):
    temp_list = []
    for word in data:
        if word not in stopwords_ol:
            temp_list.append(word)
    return temp_list


#se ruleaza o singura data

for i in range(len(data_da)):
    data_da[i] = remove_stopwords_da(data_da[i])

for i in range(len(data_ge)):
    data_ge[i] = remove_stopwords_ge(data_ge[i])

for i in range(len(data_sp)):
    data_sp[i] = remove_stopwords_sp(data_sp[i])

for i in range(len(data_it)):
    data_it[i] = remove_stopwords_it(data_it[i])

for i in range(len(data_ol)):
    data_ol[i] = remove_stopwords_ol(data_ol[i])


# def count_most_common(how_many, data):
#     ctr = Counter(data[0])
#     for i in range(1,len(data)):
#         x = Counter(data[i])
#         ctr = ctr + x
#     most_occur = ctr.most_common(how_many)
#     return most_occur

def count_most_common(how_many, texte_preprocesate):
    counter = Counter()
    
    for text in texte_preprocesate:
        counter.update(text)
    cuvinte_caracteristice = []
    for cuvant, frecventa in counter.most_common(how_many):
        if cuvant.strip():
            cuvinte_caracteristice.append(cuvant)
    return cuvinte_caracteristice

def build_id_word_dicts(cuvinte_caracteristice):
    word2id = {}
    id2word = {}
    for idx, cuv in enumerate(cuvinte_caracteristice):
      word2id[cuv] = idx
      id2word[idx] = cuv
      
    return word2id, id2word

def featurize(text_preprocesat, id2word):
    ctr = Counter(text_preprocesat)
    features = np.zeros(len(id2word))
    
    for idx in range(0, len(features)):
      cuvant = id2word[idx]
      features[idx] = ctr[cuvant]
    return features

def featurize_multi(texte, id2word):
    all_features = []
    for text in texte:
        all_features.append(featurize(text, id2word))
    return np.array(all_features)

# CITIRE SI PREPROCESARE DATE DE TEST

test_data_df = pd.read_csv(os.path.join(data_path, 'test_data.csv'))

#print(test_data_df)

lang = []

test_data = []
for text in test_data_df['text']:
  test_data.append(proceseaza(text))

nr_texte_test=len(test_data)


#impartire test_data in functie de limba

test_data_da = []
test_data_ge = []
test_data_sp = []
test_data_it = []
test_data_ol = []

data2df_da = []
data2df_ge = []
data2df_sp = []
data2df_it = []
data2df_ol = []

for i in range(len(test_data)):
    if lang[i]=='danish':
        data2df_da.append(i)
        test_data_da.append(test_data[i])
    elif lang[i]=='german':
        data2df_ge.append(i)
        test_data_ge.append(test_data[i])
    elif lang[i]=='spanish':
        data2df_sp.append(i)
        test_data_sp.append(test_data[i])
    elif lang[i]=='italian':
        data2df_it.append(i)
        test_data_it.append(test_data[i])
    else:
        data2df_ol.append(i)
        test_data_ol.append(test_data[i])

#lemmatize

for i in range(len(test_data_da)):
    for j in range(len(test_data_da[i])):
        test_data_da[i][j] = lemmatize(test_data_da[i][j], lang='da').lower()

for i in range(len(test_data_ge)):
    for j in range(len(test_data_ge[i])):
        test_data_ge[i][j] = (lemmatize(test_data_ge[i][j], lang='de')).lower()
        
for i in range(len(test_data_sp)):
    for j in range(len(test_data_sp[i])):
        test_data_sp[i][j] = lemmatize(test_data_sp[i][j], lang='es').lower()
        
for i in range(len(test_data_it)):
    for j in range(len(test_data_it[i])):
        test_data_it[i][j] = lemmatize(test_data_it[i][j], lang='it').lower()
        
for i in range(len(test_data_ol)):
    for j in range(len(test_data_ol[i])):
        test_data_ol[i][j] = lemmatize(test_data_ol[i][j], lang='nl').lower()

#stopwords

for i in range(len(test_data_da)):
    test_data_da[i] = remove_stopwords_da(test_data_da[i])

for i in range(len(test_data_ge)):
    test_data_ge[i] = remove_stopwords_ge(test_data_ge[i])

for i in range(len(test_data_sp)):
    test_data_sp[i] = remove_stopwords_sp(test_data_sp[i])

for i in range(len(test_data_it)):
    test_data_it[i] = remove_stopwords_it(test_data_it[i])

for i in range(len(test_data_ol)):
    test_data_ol[i] = remove_stopwords_ol(test_data_ol[i])


def word_dict(nr_words, train_data, test_data):
    common_words = count_most_common(nr_words, train_data)
    print(len(common_words))
    word2id, id2word = build_id_word_dicts(common_words)
    train = featurize_multi(train_data, id2word)
    test = featurize_multi(test_data, id2word)
    
    return word2id, id2word, train, test

#folosim word_dict() pe fiecare set de date

da_word2id, da_id2word, da_train, da_test = word_dict(800, data_da, test_data_da)
ge_word2id, ge_id2word, ge_train, ge_test = word_dict(800, data_ge, test_data_ge)
sp_word2id, sp_id2word, sp_train, sp_test = word_dict(800, data_sp, test_data_sp)
it_word2id, it_id2word, it_train, it_test = word_dict(800, data_it, test_data_it)
ol_word2id, ol_id2word, ol_train, ol_test = word_dict(800, data_ol, test_data_ol)

from sklearn import svm
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score, cross_val_predict

def show_accuracy_and_matrix_5fold(true, pred,x):
    for i in range(5):
        print('Accuracy: ',accuracy_score(true[x*i : x*i+x], pred[x*i : x*i+x]))#5-fold CrossValidation
        print(confusion_matrix(true[x*i : x*i+x], pred[x*i : x*i+x]))

#5-fold CrossValidation

model = svm.SVC(kernel='rbf', C=0.5)
    
preds_da = cross_val_predict(model, da_train, labels_da, cv=5)
preds_ge = cross_val_predict(model, ge_train, labels_ge, cv=5)
preds_sp = cross_val_predict(model, sp_train, labels_sp, cv=5)
preds_it = cross_val_predict(model, it_train, labels_it, cv=5)
preds_ol = cross_val_predict(model, ol_train, labels_ol, cv=5)

#Afisare acuratete si matrice de confuzie

x = int(len(da_train)/5)+1
    # x reprezinta lungimea unui fold din KFold

print('\nRezultate pt limba daneza: ')
show_accuracy_and_matrix_5fold(labels_da, preds_da, x)

print('\nRezultate pt limba germana: ')
show_accuracy_and_matrix_5fold(labels_da, preds_da, x)

print('\nRezultate pt limba spaniola: ')
show_accuracy_and_matrix_5fold(labels_da, preds_da, x)

print('\nRezultate pt limba italiana: ')
show_accuracy_and_matrix_5fold(labels_da, preds_da, x)

print('\nRezultate pt limba olandeza: ')
show_accuracy_and_matrix_5fold(labels_da, preds_da, x)

# Nu pot obtine confusion matrix

# scores = cross_val_score(model, da_train, labels_da, cv=5)
# print('Accuracy pt limba daneza: ',scores)

# scores = cross_val_score(model, ge_train, labels_ge, cv=5)
# print('Accuracy pt limba germana: ',scores)

# scores = cross_val_score(model, sp_train, labels_sp, cv=5)
# print('Accuracy pt limba spaniola: ',scores)

# scores = cross_val_score(model, it_train, labels_it, cv=5)
# print('Accuracy pt limba italiana: ',scores)

# scores = cross_val_score(model, ol_train, labels_ol, cv=5)
# print('Accuracy pt limba olandeza: ',scores)


# predict pt test data

model = svm.SVC(kernel='rbf', C=0.5)

model.fit(da_train, labels_da)
preds_da = model.predict(da_test)

model.fit(ge_train, labels_ge)
preds_ge = model.predict(ge_test)

model.fit(sp_train, labels_sp)
preds_sp = model.predict(sp_test)

model.fit(it_train, labels_it)
preds_it = model.predict(it_test)

model.fit(ol_train, labels_ol)
preds_ol = model.predict(ol_test)

# Obtinerea rezultatelor

f_data = [None]*nr_texte_test
for i in range(len(preds_da)):
    f_data[data2df_da[i]] = id2label[preds_da[i]]
    
for i in range(len(preds_ge)):
    f_data[data2df_ge[i]] = id2label[preds_ge[i]]
    
for i in range(len(preds_sp)):
    f_data[data2df_sp[i]] = id2label[preds_sp[i]]
    
for i in range(len(preds_it)):
    f_data[data2df_it[i]] = id2label[preds_it[i]]
    
for i in range(len(preds_ol)):
    f_data[data2df_ol[i]] = id2label[preds_ol[i]]


numbers = list(range(1, 13861))

res = {'id':numbers, 'label':f_data}
results = pd.DataFrame(res)

results.to_csv('results_svm.csv',index=False)

