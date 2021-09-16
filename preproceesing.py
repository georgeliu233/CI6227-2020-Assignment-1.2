import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from multiprocessing import Process,Queue
from time import time
import seaborn as sns

import math
import json

from tqdm import tqdm

sns.set_style('darkgrid')

def analyze_dataset(path):

    line = [str(i) for i in range(15)]
    train_data = pd.read_csv(path+'/adult.data',sep=',',names=line,index_col=False)
    test_data = pd.read_csv(path+'/adult.test',sep=',',names=line,index_col=False)

    plt.figure(figsize=(40,8))
    i=1
    for col,name in zip([0,2,4,10,11,12],['Age','fnlwgt',"Edu-num",'capital-gain','capital-loss','hours-per-week']):
        data =  [int(i) for i in train_data[str(col)].values] + [int(i) for i in test_data[str(col)].values]
        n_list = []
        for j in tqdm(data):
            if j!='?':
                n_list.append(j)
        print('cal...',len(data),len(n_list))
        plt.subplot(2,3,i)
        plt.title(name)
        plt.hist(n_list,bins=50)
        i+=1
    plt.savefig(path+'/hist.png')

class EntropyBased_Discrete(object):

    def __init__(self,max_splits,ent_threshold,multiprocessing=False,workers=10):
        self.max_splits = max_splits
        self.ent_threshold = ent_threshold
        self.multiprocessing = multiprocessing
        self.workers = workers
    
    def entropy(self,pairs):
        N_pairs = len(pairs)
        label_cnts = {}
        for pair in pairs:
            label = pair[-1]
            if label_cnts.get(label,-1)==-1:
                label_cnts[label]=0
            label_cnts[label]+=1
        ent = 0.0

        # one_prob = np.sum([p[-1] for p in pairs])/N_pairs
        # if one_prob==0 or one_prob==1:
        #     return 0.0
        # return -one_prob * math.log(one_prob,2) - (1-one_prob) * math.log(1-one_prob,2)
        for k in label_cnts:
            p = float(label_cnts[k])/N_pairs
            ent -= p*math.log2(p)
        return ent
    
    def multi_split_onetime(self,pairs):
        self.final_res = Queue()

        worker_dict = {}
        pair_range = np.arange(len(pairs))
        true_workers = min(len(pairs),self.workers)
        range_length = len(pairs)//true_workers
        for w in range(true_workers):
            if w==true_workers-1:
                worker_dict[w] = pair_range[w*range_length:]
            else:
                worker_dict[w] = pair_range[w*range_length:(w+1)*range_length]
        
        Process_list = []
        st = time()
        pairs = pairs[np.argsort(pairs[:,0])]

        for w in range(true_workers):
            Process_list.append(Process(target=self.single_worker,args=(pairs,worker_dict[w])))
        for process in Process_list:
            process.start()
        for process in Process_list:
            process.join()
        
        min_ent = np.inf
        min_pair = tuple()
        #print(time()-st)
        res = [self.final_res.get() for process in Process_list]
        #print(res)
        for pair in res:
            if pair[-1]<min_ent:
                min_pair = pair
                min_ent = pair[-1]

        pre_Ent,post_Ent,split_id,min_ent = min_pair
        pre_dict = {
            'ent':pre_Ent,
            'pairs':pairs[:split_id+1]
        }

        post_dict = {
            'ent':post_Ent,
            'pairs':pairs[split_id+1:]
        }
        return pre_dict,post_dict,min_ent

    def single_worker(self,pairs,length_list):
        min_ent = np.inf
        split_id = -1
        length = len(pairs)
        pre_Ent,post_Ent = -1,-1

        for i in tqdm(length_list):
            pre_data,post_data = pairs[:i+1],pairs[i+1:]
            pre_ent,post_ent = self.entropy(pre_data),self.entropy(post_data)
            avg_ent = (pre_ent*len(pre_data) + post_ent*len(post_data))/length
            if avg_ent < min_ent:
                min_ent = avg_ent
                split_id = i
                pre_Ent = pre_ent
                post_Ent = post_ent
        self.final_res.put((pre_Ent,post_Ent,split_id,min_ent))
    def split_onetime(self,pairs):
        """
        find the best split of the record
        According to weighted avg info-entropy
        """
        min_ent = np.inf
        split_id = -1
        length = len(pairs)
        pre_Ent,post_Ent = -1,-1
        sort_pairs = pairs[np.argsort(pairs[:,0])]

        for i in tqdm(range(length)):
            pre_data,post_data = sort_pairs[:i+1],sort_pairs[i+1:]
            pre_ent,post_ent = self.entropy(pre_data),self.entropy(post_data)
            avg_ent = pre_ent*len(pre_data)/len(sort_pairs) + post_ent*len(post_data)/len(sort_pairs)
            if avg_ent < min_ent:
                min_ent = avg_ent
                split_id = i
                pre_Ent = pre_ent
                post_Ent = post_ent
        
        pre_dict = {
            'ent':pre_Ent,
            'pairs':sort_pairs[:split_id+1]
        }

        post_dict = {
            'ent':post_Ent,
            'pairs':sort_pairs[split_id+1:]
        }
        return pre_dict,post_dict,min_ent
    
    def split(self,data):
        """
        Make the entropy-based discretization
        """
        #sorted_data = data[np.argsort(data[:,0])]
        print('sorted! start splitting...')
        self.final_split = {
            0:{}
        }
        self.final_split[0]['ent'] = np.inf
        self.final_split[0]['pairs'] = data
        split_ind =[0]
        classes = 1

        for i in split_ind:
            if self.multiprocessing:
               pre_split,post_split,entropy = self.multi_split_onetime(self.final_split[i]['pairs'])
            else: 
                pre_split,post_split,entropy = self.split_onetime(self.final_split[i]['pairs'])
            if entropy > self.ent_threshold and classes < self.max_splits:
                self.final_split[i] = pre_split
                next_key = max(self.final_split.keys())+1
                self.final_split[next_key] = post_split
                split_ind.extend([i])
                split_ind.extend([next_key])
                classes += 1
            else:
                break
    
def test_spliter():
    data = np.array(
              [
                  [56,1],[87,1],[129,0],[23,0],[342,1],
                  [641,1],[63,0],[2764,1],[2323,0],[453,1],
                  [10,1],[9,0],[88,1],[222,0],[97,0],
              ]
          )
    spliter = EntropyBased_Discrete(6,0.5)
    spliter.split(data)
    print(spliter.final_split)

def test_single_continuous_dis():

    path = 'C:/Users/24829/Desktop/data_minign/assignment-1/adult_dataset'
    line = [str(i) for i in range(15)]
    attr = 0

    train_data = pd.read_csv(path+'/adult.data',sep=',',names=line,index_col=False)
    test_data = pd.read_csv(path+'/adult.test',sep=',',names=line,index_col=False)
    
    age_data = []
    for i,ind in zip(train_data[str(attr)].values,train_data[str(14)].values):
        if ind==' <=50K':
            age_data.append([int(i),0])
        else:
            age_data.append([int(i),1])
    for i,ind in zip(test_data[str(attr)].values,test_data[str(14)].values):
        if ind==' <=50K.':
            age_data.append([int(i),0])
        else:
            age_data.append([int(i),1])

    spliter = EntropyBased_Discrete(12,0.5,multiprocessing=True,workers=20)
    spliter.split(np.array(age_data))
    print('complete')
    split_set = []
    i=1
    for val in tqdm(spliter.final_split.values()):
        single_set = set([ind[0] for ind in val['pairs']])
        split_set.append(single_set)
        print(i,len(single_set),single_set)
        i+=1
    new_col_train = []
    new_col_test = []
    for ind in train_data[str(attr)]:
        for j,discre_set in enumerate(split_set):
            if int(ind) in discre_set:
                new_col_train.append(j)
                break
    
    for ind in test_data[str(attr)]:
        for j,discre_set in enumerate(split_set):
            if int(ind) in discre_set:
                new_col_test.append(j)
                break
    
    # with open(path+'/age.json','w',encoding='utf-8') as writer:
    #     writer.write(json.dumps([new_col_train,new_col_test],ensure_ascii=False,indent=4))

def get_dict():

    attr_dict = {
        "workclass" : [' Private',' Self-emp-not-inc',' Self-emp-inc',' Federal-gov',' Local-gov',' State-gov'," Without-pay"," Never-worked"],
        "education" : [' Bachelors',' Some-college',' 11th',' HS-grad',' Prof-school',' Assoc-acdm',' Assoc-voc',' 9th',' 7th-8th',' 12th',' Masters',' 1st-4th',' 10th',' Doctorate',' 5th-6th',' Preschool'],
        "marital-status": [' Married-civ-spouse',' Divorced',' Never-married',' Separated',' Widowed',' Married-spouse-absent',' Married-AF-spouse'],
        "occupation" :' Tech-support, Craft-repair, Other-service, Sales, Exec-managerial, Prof-specialty, Handlers-cleaners, Machine-op-inspct, Adm-clerical, Farming-fishing, Transport-moving, Priv-house-serv, Protective-serv, Armed-Forces'.split(','),
        "relationship" :' Wife, Own-child, Husband, Not-in-family, Other-relative, Unmarried'.split(','),
        "race":' White, Asian-Pac-Islander, Amer-Indian-Eskimo, Other, Black'.split(','),
        "sex":[' Female',' Male'],
        "native-country":' United-States, Cambodia, England, Puerto-Rico, Canada, Germany, Outlying-US(Guam-USVI-etc), India, Japan, Greece, South, China, Cuba, Iran, Honduras, Philippines, Italy, Poland, Jamaica, Vietnam, Mexico, Portugal, Ireland, France, Dominican-Republic, Laos, Ecuador, Taiwan, Haiti, Columbia, Hungary, Guatemala, Nicaragua, Scotland, Thailand, Yugoslavia, El-Salvador, Trinadad&Tobago, Peru, Hong, Holand-Netherlands'.split(','),
        "label":[' <=50K',' >50K']
    }
    V = []
    for k,v in attr_dict.items():
        print(k,len(v))
        V.append(len(v))
    plt.figure(figsize=(10,5))
    plt.bar(np.arange(len(V)),V)
    plt.xticks(np.arange(len(V)),list(attr_dict.keys()))
    for x,y in zip(np.arange(len(V)),V):
        plt.text(x,y+1,str(y))
    plt.show()
    attr2id_dict={}
    for k,v in attr_dict.items():
        attr2id_dict[k] = dict((j,i) for i,j in enumerate(v))
        attr2id_dict[k][' ?']=-1
    
    #print(attr2id_dict)
    return attr2id_dict

def preprocess(path):
    line = ['age',"workclass",'fnlwgt','education','education-num',
            'marital-status','occupation','relationship','race','sex','capital-gain','capital-loss','hours-per-week',
            'native-country','label']
    # key_to_lines = dict((j,i) for i,j in enumerate(line))
    
    train_data = pd.read_csv(path+'/adult.data',sep=',',names=line,index_col=False)
    test_data = pd.read_csv(path+'/adult.test',sep=',',names=line,index_col=False)
    print(train_data.head(10))
    attr2id_dict = get_dict()
    key_list = set(list(attr2id_dict.keys()))
    new_train_data = dict()
    new_test_data = dict()

    #tokenize the discrete attributes and label, for missing values, we give it -1
    for l in tqdm(line):

        new_train_data[l]=[]
        if l in key_list:
            for data in train_data[l].values:
                new_train_data[l].append(attr2id_dict[l][data])
        else:
           new_train_data[l]=train_data[l].values
        
        new_test_data[l]=[]
        if l in key_list:
            for data in test_data[l].values:
                if l=='label':
                    data = data[:-1]
                new_test_data[l].append(attr2id_dict[l][data])
        else:
           new_test_data[l]=test_data[l].values
            

    new_train_df = pd.DataFrame(data=new_train_data)
    new_test_df = pd.DataFrame(data=new_test_data)

    new_train_df.to_csv(path+'/train.csv',index=False)
    new_test_df.to_csv(path+'/test.csv',index=False)
    #process train:
    





    



if __name__=="__main__":
    # analyze_dataset('C:/Users/24829/Desktop/data_minign/assignment-1/adult_dataset')
    #test_spliter()
    
    #test_single_continuous_dis()
    preprocess('C:/Users/24829/Desktop/data_minign/assignment-1/adult_dataset')
    #a = get_dict()