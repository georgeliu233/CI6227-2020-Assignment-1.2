from sklearn.ensemble import RandomForestClassifier
from sklearn import tree
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('darkgrid')

import graphviz,pydotplus
from tqdm import tqdm
from six import StringIO
from time import time

from sklearn.impute import SimpleImputer as imputer
imputer=imputer(strategy='most_frequent',missing_values=-1)
import os
os.environ['PATH'] += os.pathsep+'C:/Program Files/Graphviz/bin/'

def train_model():
    path = 'C:/Users/24829/Desktop/data_minign/assignment-1/adult_dataset'
    train = pd.read_csv(path+'/train.csv')
    attr = [at for at in train.columns if at !='label']
    X = train[attr].values
    imputer.fit(X)
    X = imputer.fit_transform(X)

    Y = train['label'].values
    print(X.shape,Y.shape)
    st = time()
    classifier = RandomForestClassifier(n_estimators=10,max_depth=20)
    classifier.fit(X,Y)
    print(time()-st)
    
    return classifier

def test_params():
    path = 'C:/Users/24829/Desktop/data_minign/assignment-1/adult_dataset'
    train = pd.read_csv(path+'/train.csv')
    attr = [at for at in train.columns if at !='label']
    X = train[attr].values
    imputer.fit(X)
    X = imputer.fit_transform(X)

    Y = train['label'].values

    path = 'C:/Users/24829/Desktop/data_minign/assignment-1/adult_dataset'
    test = pd.read_csv(path+'/test.csv')
    attr = [at for at in test.columns if at !='label']
    test_x = test[attr].values
    imputer.fit(test_x)
    test_x = imputer.fit_transform(test_x)
    test_y = test['label'].values

    res_list=[]
    x_list = [3,5,10,20,50,100]
    for esti_num in x_list:
        classifier = RandomForestClassifier(n_estimators=10,max_depth=esti_num)
        #start=time()
        classifier.fit(X,Y)
        pred_y = classifier.predict(test_x)
        outcome = test_metric(test_y,pred_y)
        res_list.append(list(outcome))
    
    plt.figure()
    for i in range(4):
        plt.plot(x_list,[res[i] for res in res_list])
        plt.scatter(x_list,[res[i] for res in res_list])
    plt.legend(['Accuracy','Precision','Recall','F1'])
    plt.savefig(path+'/metric_depth.png')

    

def plot_tree(classifier):
    path = 'C:/Users/24829/Desktop/data_minign/assignment-1/adult_dataset'
    p = 'C:/Users/24829/Desktop/data_minign/assignment-1/imgs/'
    test = pd.read_csv(path+'/test.csv')
    ind = 0
    attr = [at for at in test.columns if at !='label']
    classn = ['<50K','>=50k']
    for model in classifier.estimators_:
        #data = StringIO()
        tree.export_graphviz(
            model,feature_names=attr,class_names=classn,
            filled=True,rounded=True,special_characters=True,
            out_file='tree.dot'
        )
        with open('tree.dot',encoding='utf-8') as f:
            data=f.read()
        graph = graphviz.Source(data)
        graph.view()
        #print(data)
        # graph = pydotplus.graph_from_dot_data(data.getvalue())
        # graph.write_pdf(p+str(ind)+'.pdf')
        ind+=1

def test_model(classifier):
    path = 'C:/Users/24829/Desktop/data_minign/assignment-1/adult_dataset'
    test = pd.read_csv(path+'/test.csv')
    attr = [at for at in test.columns if at !='label']
    test_x = test[attr].values
    imputer.fit(test_x)
    test_x = imputer.fit_transform(test_x)
    test_y = test['label'].values
    
    st = time()
    pred_y = classifier.predict(test_x)
    print(time()-st)
    test_metric(test_y,pred_y)
    # acc = 1-np.sum(np.abs(test_y-pred_y))/len(test_y)
    # print(acc)

def test_metric(ground_t,pred):
    cnt_dict = {
        'TP':0,'FN':0,'FP':0,'TN':0
    }
    for y,y_pred in zip(ground_t,pred):
        if y==1:
            if y_pred==1:
               cnt_dict['TP']+=1
            else:
                cnt_dict['FN']+=1
        else:
            if y_pred==1:
               cnt_dict['FP']+=1
            else:
                cnt_dict['TN']+=1
    
    acc = (cnt_dict['TP']+cnt_dict['TN'])/(cnt_dict['TP']+cnt_dict['TN']+cnt_dict['FP']+cnt_dict['FN'])
    precision = cnt_dict['TP']/(cnt_dict['TP']+cnt_dict['FP'])
    recall = cnt_dict['TP']/(cnt_dict['TP']+cnt_dict['FN'])
    F = 2*precision*recall/(precision+recall)

    print('acc:',acc,"|precision:",precision,'|recall:',recall,'|F1-score:',F)
    return acc,precision,recall,F


if __name__=="__main__":
    classifier = train_model()
    test_model(classifier)
    #plot_tree(classifier)
    # test_params()