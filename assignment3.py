import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets.samples_generator import make_blobs
import seaborn as sns
import pickle
from sklearn.cluster import KMeans

sns.set_style('darkgrid')

def build_dataset():
    center=[[0,1],[-1,-1],[1,-1]]
    cluster_std=0.5
    X,labels=make_blobs(n_samples=300,centers=center,n_features=2,
                        cluster_std=cluster_std,random_state=0)
    print('X.shape',X.shape)
    print("labels",set(labels))

    unique_lables=set(labels)
    plt.figure()
    colors=plt.cm.Spectral(np.linspace(0,1,len(unique_lables)))
    for c in center:
        plt.scatter(c[0], c[1],c='red',marker='*',s=150)
    for k,col in zip(unique_lables,colors):
        x_k=X[labels==k]
        plt.plot(x_k[:,0],x_k[:,1],'o',markersize=5)#,markerfacecolor=col,markeredgecolor="k")
    plt.title('generated_dataset')
    plt.show()
    
    # plt.show()
    with open('C:/Users/24829/Desktop/data_minign/datasets_3.pkl','wb') as writer:
        pickle.dump(X,writer)

def train_and_vis(iters=5,K=3,random_choice=False):
    with open('C:/Users/24829/Desktop/data_minign/datasets_3.pkl','rb') as reader:
        X = pickle.load(reader)
    len_x = X.shape[0]
    random_init = np.random.randint(0,len_x,size=(K,))
    print(random_init,len_x)

    centroid = X[random_init]
    if not random_choice:
        centroid = np.array([
            [-2,-2.5],[1,2.5],[-2,2.5]
        ])
    for i in range(iters):
        
        #compute dist
        dis_list = []
        for k in range(K):
            dis_list.append(np.reshape(np.linalg.norm(X-centroid[k],axis=-1),(-1,1)))
        dis = np.concatenate(dis_list,axis=-1)
        # print(dis.shape)
        labels = np.argmin(dis,axis=-1)
        # print(np.unique(labels))
        #update centorid
        new_centroid = []
        for k in range(K):
            new_centroid.append(np.mean(
                X[labels==k],axis=0
            ))
        centroid = new_centroid
        #plot each time

        unique_lables=np.unique(labels)
        colors = ['orange','g','b']
        plt.figure()
        for c in centroid:
            plt.scatter(c[0], c[1],c='red',marker='*',s=150)
        for k,col in zip(unique_lables,colors):
            x_k=X[labels==k]
            plt.plot(x_k[:,0],x_k[:,1],'o',markersize=5,color=col,alpha=0.5)#,,markeredgecolor="k")
        plt.title('K-means-iter:'+str(1+i))

        if random_choice:
            s = 'A'
        else:
            s = 'B'

        plt.savefig('C:/Users/24829/Desktop/data_minign/'+s+str(1+i)+'.png')
# build_dataset()
train_and_vis()
