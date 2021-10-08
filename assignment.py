import numpy as np 
import matplotlib.pyplot as plt 
from itertools import combinations
from time import time
import pickle
import pandas as pd

from sys import argv
from fim import apriori, eclat, fpgrowth, fim
from time import time

def generate_candidates(num):
    ls = np.arange(num)
    out_items = list()
    for i in range(num+1):
        combines = list(combinations(ls,i))
        for c in combines:
            out_items.append(set(list(c)))
    return out_items

def generate_rules(list_in):
    out_items = list()
    for i in range(len(list_in)+1):
        combines = list(combinations(list_in,i))
        for c in combines:
            out_items.append([set(list(c)),set(list_in)-set(list(c))])
    return out_items

def generate_random_dataset(num,length_list=[1000,2000,5000,10000,20000,50000,100000],width=5):
    
    ls = np.arange(num)
    data_list = []
    for length in length_list:
        dataset = []
        for _ in range(length):
            ind = np.random.randint(2,width+1)
            lines = list(combinations(ls,ind))
            rand_num = np.random.choice(np.arange(len(lines)))
            line = lines[rand_num]
            dataset.append(line)
        data_list.append(dataset)
    with open('C:/Users/24829/Desktop/data_minign/datasets.pkl','wb') as writer:
        pickle.dump(data_list,writer)
    

def brute_force_mining(data,top_k=3):
    
    itemsets = generate_candidates(5)
    cnt = np.zeros(32)
    start = time()
    for line in data:
        for i,item in enumerate(itemsets):
            line_set = set(line)
            if item.issubset(line_set):
                cnt[i]+=1
    top_indices = np.argsort(cnt)[-top_k:]
    rules = []
    for ind in top_indices:
        itemset = itemsets[ind]
        rules = rules + generate_rules(list(itemset))
    comp = time()-start
    print(comp)
    return comp

def test_brute_force():
    with open('C:/Users/24829/Desktop/data_minign/datasets.pkl','rb') as reader:
        datas = pickle.load(reader)
    t = []
    for _ in range(10):
        time_list = []
        for data in datas:
            time_list.append(brute_force_mining(data))
        t.append(time_list)
    with open('C:/Users/24829/Desktop/data_minign/brute.pkl','wb') as writer:
        pickle.dump(t,writer)
        
def plot_curve():
    import seaborn
    seaborn.set_style('darkgrid')
    with open('C:/Users/24829/Desktop/data_minign/aprori.pkl','rb') as reader:
        datas = pickle.load(reader)
    datas = np.array(datas)
    # plt.figure(figsize=(20,5))
    # plt.figure()
    plt.xticks(np.arange(7),[1000,2000,5000,10000,20000,50000,100000])
    plt.xlabel('Data size')
    plt.ylabel('Times/s')
    m_list = []
    std_list = [] 
    mean = np.mean(datas,0)
    std = np.std(datas,0)
    m_list.append(mean)
    std_list.append(std)
    plt.plot(np.arange(7),mean)
    plt.fill_between(np.arange(7),mean+std,mean-std,alpha=0.3)
    plt.savefig('C:/Users/24829/Desktop/data_minign/comp-3.pdf')
    # plt.show()

    with open('C:/Users/24829/Desktop/data_minign/brute.pkl','rb') as reader:
        datas = pickle.load(reader)
    datas = np.array(datas)
    mean = np.mean(datas,0)
    std = np.std(datas,0)

    m_list.append(mean)
    std_list.append(std)
    plt.figure()
    plt.plot(np.arange(7),mean)
    plt.fill_between(np.arange(7),mean+std,mean-std,alpha=0.3)
    # plt.legend(['Apriori','Brute-force'])

    # plt.show()
    plt.savefig('C:/Users/24829/Desktop/data_minign/comp-2.pdf')

    full_data = np.concatenate((m_list,std_list),0)
    df = pd.DataFrame(data=full_data,index=['Mean_bf','Mean_apri','Std_bf','Std_apri'],
    columns=[1000,2000,5000,10000,20000,50000,100000])
    df.to_csv('C:/Users/24829/Desktop/data_minign/res.csv')

plot_curve()

def test_apriori():
    with open('C:/Users/24829/Desktop/data_minign/datasets.pkl','rb') as reader:
        datas = pickle.load(reader)
    times = []
    for _ in range(10):
        t = []
        for data in datas:
            
            start = time()
            res = apriori(data, supp=-3, zmin=2)
            comp = time()-start
            print(len(data),comp)
            t.append(comp)
        times.append(t)

    with open('C:/Users/24829/Desktop/data_minign/aprori.pkl','wb') as writer:
        pickle.dump(times,writer) 