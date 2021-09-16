import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import random
import json

import keras
from keras import layers,models
from time import time

global count
def data_generator(X,Y,batch_size,shuffle=True):
    
    count  = 1
    while True:

        if count==1 and shuffle:
            candidate = np.arange(X.shape[0])
            random.shuffle(candidate)	
        if count*batch_size-1>=X.shape[0]:
            count = 1
            if shuffle:
                candidate = np.arange(X.shape[0])
                random.shuffle(candidate)

        if shuffle:
            batch_x,batch_y=[],[]
            for i in candidate[(count-1)*batch_size:count*batch_size]:
                #assert len(X[i])==97,X[i]
                batch_x.append(X[i])
                batch_y.append(Y[i])
            batch_x,batch_y=np.array(batch_x),np.array(batch_y)
        else:
            batch_x = X[(count-1)*batch_size:count*batch_size,:]
            batch_y = Y[(count-1)*batch_size:count*batch_size]
        
        count += 1
        yield batch_x , batch_y

class Deep_Ensemble_model(object):
    def __init__(self,generator,num_of_ensenmbles,continuous_dim,hidden_dim,build_embed=False):
        self.generator = generator
        self.num_of_ensenmbles=num_of_ensenmbles
        self.hidden_dim = hidden_dim
        self.build_embed=build_embed
        self.build_models()

    def build_embeddings(self):

        self.workclass = tf.placeholder(tf.int64,[None],name='workclass')
        wk_var = tf.get_variable('wk',shape=[9,self.hidden_dim],initializer=tf.random_uniform_initializer(-1,1))
        wk_embedd = tf.nn.embedding_lookup(wk_var,self.workclass)

        self.education = tf.placeholder(tf.int64,[None],name='education')
        edu_var = tf.get_variable('edu',shape=[17,self.hidden_dim],initializer=tf.random_uniform_initializer(-1,1))
        edu_embedd = tf.nn.embedding_lookup(edu_var,self.education)

        self.martial = tf.placeholder(tf.int64,[None],name='martial')
        mar_var = tf.get_variable('mar',shape=[8,self.hidden_dim],initializer=tf.random_uniform_initializer(-1,1))
        mar_embedd = tf.nn.embedding_lookup(mar_var,self.martial)

        self.occupation = tf.placeholder(tf.int64,[None],name='occu')
        occ_var = tf.get_variable('occ',shape=[15,self.hidden_dim],initializer=tf.random_uniform_initializer(-1,1))
        occ_embedd = tf.nn.embedding_lookup(occ_var,self.occupation)

        self.relation = tf.placeholder(tf.int64,[None],name='rela')
        rel_var = tf.get_variable('rel',shape=[7,self.hidden_dim],initializer=tf.random_uniform_initializer(-1,1))
        rel_embedd = tf.nn.embedding_lookup(rel_var,self.relation)

        self.race = tf.placeholder(tf.int64,[None],name='race')
        re_var = tf.get_variable('rac',shape=[6,self.hidden_dim],initializer=tf.random_uniform_initializer(-1,1))
        rece_embedd = tf.nn.embedding_lookup(re_var,self.race)

        self.sex = tf.placeholder(tf.int64,[None],name='sex')
        se_var = tf.get_variable('se',shape=[3,self.hidden_dim],initializer=tf.random_uniform_initializer(-1,1))
        sex_embedd = tf.nn.embedding_lookup(se_var,self.sex)

        self.native = tf.placeholder(tf.int64,[None],name='native')
        ne_var = tf.get_variable('na',shape=[42,self.hidden_dim],initializer=tf.random_uniform_initializer(-1,1))
        native_embedd = tf.nn.embedding_lookup(ne_var,self.native)

        return wk_embedd+edu_embedd+mar_embedd+occ_embedd+rel_embedd+rece_embedd+sex_embedd+native_embedd
    
    def build_model(self):
        if self.build_embed:
            discrete_out = self.build_embeddings()
            self.input_tensor=(tf.float32,[None,6])
            
        else:
            self.input_tensor=(tf.float32,[None,88])


        
def get_data():
    path = 'C:/Users/24829/Desktop/data_minign/assignment-1/adult_dataset'
    from sklearn.impute import SimpleImputer as imputer
    imputer=imputer(strategy='most_frequent',missing_values=np.NaN)

    train = pd.read_csv(path+'/train.csv')
    train.dropna(axis=0)
    
    imputer.fit(train)
    train = pd.read_csv(path+'/train.csv')
    imputer.fit_transform(train)

    test = pd.read_csv(path+'/test.csv')
    train.dropna(axis=0)
    imputer.fit(test)
    test = pd.read_csv(path+'/test.csv')
    imputer.fit_transform(test)

    

    category_col_1 =['workclass', 'education', 'occupation',"marital-status",
               'relationship','native-country','race','sex'] 
    
    new = pd.concat([train,test])
    new_df = pd.get_dummies(new, columns=category_col_1, drop_first=True)
    # train_df = pd.get_dummies(train, columns=category_col_1, drop_first=True)
    # test_df = pd.get_dummies(test, columns=category_col_1, drop_first=True)
    # train_df.to_csv(path+'/nn_train.csv',index=False)
    # test_df.to_csv(path+'/nn_test.csv',index=False)
    train_df = pd.DataFrame(new_df.values[:train.values.shape[0],:],columns=new_df.columns)
    test_df = pd.DataFrame(new_df.values[train.values.shape[0]:,:],columns=new_df.columns)

    attr = [at for at in train_df.columns if at not in ['label','fnlwgt']]

    train_x = train_df[attr].values
    train_y = train_df['label'].values
    attr = [at for at in test_df.columns if at not in ['label','fnlwgt']]
    test_x = test_df[attr].values
    test_y = test_df['label'].values

    print(train_x.shape,test_x.shape)    

    return train_x,train_y,test_x,test_y

def train_test():
    all_set = pd.read_csv('C:/Users/24829/Desktop/Census-classifier-comparison-master/adult_kearsNN.csv')
    attr = [at for at in all_set.columns if at not in ['income']]
    X = all_set[attr].values
    Y = all_set['income'].values
    train_x,test_x = X[:32561],X[32561:]
    train_y,test_y = Y[:32561],Y[32561:]
    return train_x,train_y,test_x,test_y
def train_and_test_nn(num_of_ensenmbles=1):
    path = 'C:/Users/24829/Desktop/data_minign/assignment-1/adult_dataset'
    train_x,train_y,test_x,test_y = train_test()

    input_layer = tf.placeholder(tf.float32,[None,88],'input_layer')
    gt_layer = tf.placeholder(tf.float32,[None,1],'ground_truth_layer')

    logits_list = []
    for _ in range(num_of_ensenmbles):
        feat = tf.layers.dense(input_layer,units=64,activation=tf.nn.relu,kernel_initializer=tf.random_uniform_initializer(-1,1))
        feat = tf.layers.dense(input_layer,units=32,activation=tf.nn.relu,kernel_initializer=tf.random_uniform_initializer(-1,1))
        #feat = tf.layers.dense(feat,units=32,activation=tf.nn.tanh,kernel_initializer=tf.random_uniform_initializer(-1,1))
        logits = tf.layers.dense(feat,units=1,kernel_initializer=tf.random_uniform_initializer(-1,1))
        logits_list.append(logits)

    voted_logits  = tf.reduce_mean(tf.squeeze(tf.transpose(logits_list,(1,0,2))),axis=-1)
    prediction = tf.nn.sigmoid(voted_logits)
    loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=gt_layer,logits=voted_logits))
    optimizer = tf.train.AdamOptimizer(learning_rate=3e-4)
    train_op = optimizer.minimize(loss)

    batch_size=512
    epochs=50
    step = 0
    max_steps = train_x.shape[0]//batch_size*epochs

    loss_cnt = []
    vaild_loss = []

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        start = time()
        for x,y in data_generator(train_x,train_y,batch_size):
            step+=1
            feed = {
                input_layer:x,
                gt_layer:np.reshape(y,(-1,1))
            }
            l,_ = sess.run([loss,train_op],feed_dict=feed)
            if step%10==0:
                loss_cnt.append((np.asscalar(l),step))
            
            if step%100==0:
                print('loss:',l,'|steps:',step)
            
            if step%(train_x.shape[0]//batch_size)==0 and False:
                print('testing..')
                for x,y in data_generator(test_x,test_y,batch_size):
                    feed = {
                        input_layer:x,
                        gt_layer:np.reshape(y,(-1,1))
                    }
                    l,pred = sess.run([loss,prediction],feed_dict=feed)
                    pred = np.squeeze(pred)
                    pred = pred>0.5
                    acc,precision,recall,F=test_metric(y,pred)
                    l = np.asscalar(l)
                    # acc = np.asscalar(acc)
                    # precision = np.asscalar(precision)
                    # recall = np.asscalar(recall)
                    # F = np.asscalar(F)
                    print('acc:',acc,"|precision:",precision,'|recall:',recall,'|F1-score:',F,'|loss:',l)
                    vaild_loss.append((acc,precision,recall,F,l,step))
                    break
                with open(path+'/res2.json','w',encoding='utf-8') as writer:
                    writer.write(json.dumps([loss_cnt,vaild_loss],ensure_ascii=False,indent=4))
            
            if step>=max_steps:
                break
        a = start-time()
        print('final-loss:',l,'start testing..',a)

        test_steps=0
        max_test_steps = test_x.shape[0]//batch_size
        test_mat = []
        start=time()
        for x,y in data_generator(test_x,test_y,batch_size,shuffle=False):
            feed = {
                input_layer:x
            }
            pred = sess.run(prediction,feed_dict=feed)
            pred = np.squeeze(pred)
            pred = pred>0.5
            test_mat.append(pred)
            test_steps+=1
            if test_steps>=max_test_steps:
                b = start-time()
                print(b)
                break
        
        test_mat = np.reshape(np.array(test_mat),(-1,1))
        acc,precision,recall,F=test_metric(test_y,np.squeeze(test_mat))
        print(acc,precision,recall,F)
        return (acc,precision,recall,F,a,b,l)

def print_acc():
    acc = [0.8503150201612904, 0.8888104838709677, 0.8863634072580645, 0.8927797379032258, 0.8974344758064516]
    train_time = [-27.252010822296143, -43.73055100440979, -92.96853923797607, -134.28169775009155, -321.25241255760193]
    test_time = [-0.07864871978759766, -0.15561509132385254, -0.3365809917449951, -0.38812732696533203, -1.1123011112213135]
    plt.figure()
    x = [3,5,10,20,50]
    sns.set_style('darkgrid')
    plt.subplot(1,3,1)
    plt.plot(x,acc)
    plt.title('Acc')
    plt.subplot(1,3,2)
    plt.plot(x,-np.array(train_time))
    plt.title('Train-time/s')
    plt.subplot(1,3,3)
    plt.plot(x,-np.array(test_time))
    plt.title('Test-time/s')
    plt.show()



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
    if cnt_dict['TP']+cnt_dict['FP']==0:
        precision=0.0
    else:
        precision = cnt_dict['TP']/(cnt_dict['TP']+cnt_dict['FP'])
    if cnt_dict['TP']+cnt_dict['FN']==0:
        recall=0
    else:
        recall = cnt_dict['TP']/(cnt_dict['TP']+cnt_dict['FN'])
    
    if recall==0 and precision==0:
        F=0
    else:
        F = 2*precision*recall/(precision+recall)

    #print('acc:',acc,"|precision:",precision,'|recall:',recall,'|F1-score:',F)
    return acc,precision,recall,F

def smooth(scalar,weight=0.9):
    last = scalar[0]
    smoothed = []
    for point in scalar:
        smoothed_val = last * weight + (1 - weight) * point
        smoothed.append(smoothed_val)
        last = smoothed_val
    return smoothed

def plot_curve():
    sns.set_style('darkgrid')
    path = 'C:/Users/24829/Desktop/data_minign/assignment-1/adult_dataset'
    with open(path+'/res2.json','r',encoding='utf-8') as reader:
        loss_cnt,vaild_loss = json.load(reader)

    train_steps = [p[1] for p in loss_cnt]
    train_loss = [p[0] for p in loss_cnt]

    test_steps = [p[-1] for p in vaild_loss]
    test_loss = [p[-2] for p in vaild_loss]

    acc = [p[0] for p in vaild_loss]
    pre = [p[1] for p in vaild_loss]
    rec = [p[2] for p in vaild_loss]
    f1 = [p[3] for p in vaild_loss]
    # plt.figure()
    # plt.plot(train_steps,smooth(train_loss))
    # plt.scatter(train_steps,smooth(train_loss),linewidths=0.5)

    # plt.plot(test_steps,test_loss)
    # plt.scatter(test_steps,test_loss,linewidths=0.5)
    # plt.legend(['Train','Test'])
    # plt.ylim([0,2])
    # plt.show()
    plt.figure()
    s = 0.4
    plt.plot(test_steps,smooth(acc,s))
    plt.scatter(test_steps,smooth(acc,s))

    plt.plot(test_steps,smooth(pre,s))
    plt.scatter(test_steps,smooth(pre,s))

    plt.plot(test_steps,smooth(rec,s))
    plt.scatter(test_steps,smooth(rec,s))

    plt.plot(test_steps,smooth(f1,s))
    plt.scatter(test_steps,smooth(f1,s))

    plt.legend(['Accuracy','Precision','Recall','F1'])
    plt.show()


def test_ensembles():
    acc_list = []
    pre_list = []
    recall_list = []
    F_list = []
    a_list = []
    b_list = []
    l_list = []
    for en in [3,5,10,20,50]:
        acc,precision,recall,F,a,b,l = train_and_test_nn(num_of_ensenmbles=en)
        acc_list.append(acc)
        pre_list.append(precision)
        recall_list.append(recall)
        F_list.append(F)
        a_list.append(a)
        b_list.append(b)
        l_list.append(l)

    print(acc_list)
    print(pre_list)
    print(recall_list)
    print(F_list)

    print(a_list)
    print(b_list)
    print(l_list)

    plt.figure()
    x = [3,5,10,20,50]
    plt.plot(x,acc_list)
    plt.plot(x,pre_list)
    plt.plot(x,recall_list)
    plt.plot(x,F_list)
    plt.legend(['Accuracy','Precision','Recall','F1'])
    plt.show()



    

if __name__=="__main__":
    # train_and_test_nn()
    # plot_curve()
    # test_ensembles()
    print_acc()