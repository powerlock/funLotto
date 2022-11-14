import scipy.stats as ss
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import AgglomerativeClustering
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import recall_score
# generate random numbers, this generation is to make noise to the data, the geneation is NOT normal distributed
# to simulate the similar property of the lottery number, assuming every number has the similar probability.
scale = MinMaxScaler()
# import data
file_path = "../data/powerball.xlsx"
df = pd.read_excel(file_path)
# label data with agglomerative clustering method
scale = MinMaxScaler()
X = df.drop('DrawDate', axis=1)
X_norm = scale.fit_transform(X)
Agg = AgglomerativeClustering(n_clusters=2).fit(X_norm) # use two clusters to classify current data
df['cluster_agg'+str(2)] = Agg.labels_
df_class = df.drop('DrawDate', axis=1)
print(df_class.columns)
def generate_noise(start,end,std,size): 

    x = np.arange(start,end)
    xU, xL = x + 0.5, x - 0.5
    prob = ss.norm.cdf(xU, scale = std) - ss.norm.cdf(xL, scale = std)
    prob = prob / prob.sum() # normalize the probabilities so their sum is 1
    nums = np.random.choice(x, size = size, p = prob)
    #plt.hist(nums, bins = len(x)) # plot the nums to check the distribtution, here it is not a normal distribution
    return nums

# find each numbers frequency in each feature. col is the columns of the data set, typically the field name of all 6 numbers
def find_frequency(df,col, l=list(range(70)), p=list(range(26))): # predefine lottery numbers and powerball number range, 70 and 26
    i = 1
    j = 0
    k = 0
    Frequency = []
    temp = []
    while i < len(col)-1:
        print("Calculating feature ", i)
        while j < len(l):
            fr = df.loc[df[col[i]]==j].value_counts().sum()
            temp.append(fr)
            j += 1
        Frequency.append(temp) # all numbers 1-70 frequency
        temp = []
            
        i +=1
        j = 0
    print('Frequency length is, ', len(Frequency))
    print(Frequency)
    P_fre = []
    while k < len(p):
        pfr = df.loc[df[col[len(col)-1]]==k].value_counts().sum()
        P_fre.append(pfr) # all powerball freqency
        k += 1
    print("powerball freqency.....",P_fre)

    return Frequency,P_fre
# use the frequency, to find which numbers have the high frequencey in the dataset
def find_max(arr,number_arr):
    ind = []
    val = max(arr)
    print("Highest frequency is  ", val)
    for i in range(len(arr)):
        if arr[i]==val:
            ind.append(i)
    res = []
    for n in ind:
        res.append(number_arr[n])
    print("The number with highest frequency is ", res)
    return res
# Generate noise data
def noise_data(df_class,std=50,size=5,sample=50):
    lot = []
    power = []
    for i in range(sample):
        random_lot= generate_noise(1,70,std=std,size=size)
        random_power = generate_noise(1,26,std=std,size=1)
        lot.append(sorted(random_lot))
        power.append(random_power)
    df_lot = pd.DataFrame(lot)
    df_power = pd.DataFrame(power)
    noise = pd.concat([df_lot,df_power],axis=1)
    noise.columns = df_class.columns.to_list()[0:6] # finish create dataset with noise and two other classes, in here only class 2 is noise
    noise['cluster_agg2'] = 2
    noise_df = pd.concat([df_class,noise], axis=0).drop_duplicates()
    #print("Data generation is done")
    return noise_df,noise
def training():
    i = 1
    recall_v = 1
    n = 1
    counter = 0
    while recall_v > 0.3 or n==0: # if recall is equal or less than 0.5, we can not tell it is true value or random value, use 0.3 as the worst

        noise_df,noise = noise_data(df_class,std=50,size=5,sample=65)
        # use pretrained model classify the data, to find the worst prediction
        noise_df.dropna(inplace=True)
        X_new = noise_df.drop(['cluster_agg2'],axis=1)
        y_new = noise_df['cluster_agg2']
        X_train_new, X_test_new, y_train_new, y_test_new = train_test_split(X_new, y_new, test_size=0.2, random_state=42)
        X_train_new_ = scale.fit_transform(X_train_new)
        X_test_new_ = scale.fit_transform(X_test_new)
        lr = LogisticRegression(max_iter=1000)
        model = lr.fit(X_train_new_, y_train_new)
        y_pred_x = model.predict(X_test_new_)

        recall = recall_score(y_test_new, y_pred_x,average=None)
        recall_v = recall[2]
        if i%100==0:
            print("**********Generating numbers************")
            print(recall)
            print(noise.head(1))
            print('Original data with noise')
            print(X_train_new.head(1))
        if recall_v <= 0.3:
            # select data and do the prediction for noise, filter those data with predictions are 1 or 0
            filtered = noise.drop('cluster_agg2', axis=1)
            filtered_ = scale.fit_transform(filtered)
            y_pred_filtered = model.predict(filtered_)
            #print(y_pred_filtered)
            #print("                                     ")
            y_ = pd.DataFrame({'label':y_pred_filtered})
            y_f = y_.loc[y_['label']==0].index.to_list()
            final = noise.loc[y_f,:]

            if final.empty:
                n = 0
            else:
                test = final.copy()
                test_ = test[test.columns.to_list()[0:6]]
                
                final_list = test_.transpose().reset_index(drop=True)
                col = final_list.columns.to_list()
                L =[]
                for c in col:
                    l = final_list[c].unique()
                    if len(l)<6:
                        L.append(c)
                
                final_list.drop(L, axis = 1, inplace=True)
                test_ = final_list.transpose()
                test_.columns = test.columns.to_list()[0:6]
                if test_.empty:
                    n=0
                else:
                    res = test_
                    print("======================")
                    print(test_.head())
        i += 1
        counter += 1
       
    return res, counter, recall,i


