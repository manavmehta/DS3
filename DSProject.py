#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 12 17:54:24 2019

@author: manavmehta
"""

from google.colab import files
uploaded = files.upload()
import io
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.mixture import GaussianMixture as GMix
from sklearn.metrics import confusion_matrix,accuracy_score
from sklearn.naive_bayes import GaussianNB
from sklearn import metrics
## Basics : Date is the first column, Y is the class(Occupancy) and the rest attributes are X :)

def replacing_median(df):
    X=df.drop(columns=['date','Occupancy'])
    #print(X.keys())
    col=list(X.columns)
    for i in col:
        q1=X[i].quantile(0.25)
        q3=X[i].quantile(0.75)
        iqr=q3-q1
        lower_bound = q1-1.5*iqr
        upper_bound = q3+1.5*iqr
        median=X[i].median()
        X[i]= np.where(X[i]<=lower_bound,median,X[i])
        X[i] = np.where(X[i]>=upper_bound,median,X[i])
    return X

def replacing_mean(df):     #DONE
    X=df.drop(columns=['date','Occupancy'])
    #print(X.keys())
    col=list(X.columns)
    for i in col:
        q1=X[i].quantile(0.25)
        q3=X[i].quantile(0.75)
        iqr=q3-q1
        lower_bound = q1-1.5*iqr
        upper_bound = q3+1.5*iqr
        mean=X[i].mean()
        X[i]= np.where(X[i]<=lower_bound,mean,X[i])
        X[i] = np.where(X[i]>=upper_bound,mean,X[i])
    return X


def PCAA(date,X,Y,d):     #DONE
    print("For Dimensions : ",d)
    X = StandardScaler().fit_transform(X)
    pca = PCA(n_components=d)
    
    Xi=['X'+str(i+1) for i in range(d)]
    
    X_PCA = pca.fit_transform(X)
    X_PCA = pd.DataFrame(data = X_PCA, columns = Xi)
    finaldf=pd.concat([date,X_PCA,Y], axis=1)
    return finaldf
    
def standardise(df):         #DONE
    std_df = df.copy()
    std_df=std_df.drop(columns=['date','Occupancy'])
    col = list(std_df.columns)
    for i in col:
        std_df[i] = (std_df[i] - std_df[i].mean())/std_df[i].std()
    return std_df


def min_max(df):     #DONE
    min_df = df.copy()
    min_df=min_df.drop(columns=['date','Occupancy'])
    col = list(min_df.columns)
    for i in col:
        min_df[i] = (min_df[i] - min_df[i].min())/(min_df[i].max() - min_df[i].min())
        
    return min_df


def GMM_Analysis(x,y):
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42, shuffle=True)
    x0_train = x_train[y_train==0]
    x1_train = x_train[y_train==1]
    Q=[1,2,4,8,16]
    print("\n\nGMM ANALYSIS\n\n")
    for q in Q:
        gmm0,gmm1 = GMix(n_components=q).fit(x0_train),GMix(n_components=q).fit(x1_train)
        y_pred=[int(gmm0.score_samples(x_test)[i]<gmm1.score_samples(x_test)[i]) for i in range(len(x_test))]
        print("For Q = ", q, " : ")
        print("\nConfusion Matrix : \n",[list(confusion_matrix(y_test,y_pred)[i]) for i in range(len(confusion_matrix(y_test,y_pred)))])
        print('\nAccurcy Score : ',accuracy_score(y_test,y_pred))
        print()


def x_train_test_split(X,Y,size_test):
    x_train, x_test,x_label_train, x_label_test = train_test_split(X,Y, test_size=size_test, random_state=42, shuffle= True)
    return x_train,x_test,x_label_train,x_label_test

def knn_analysis(dataframe,Y):
    X=dataframe
    X_label= Y
    X_train, X_test, X_label_train, X_label_test =train_test_split(X, X_label, test_size=0.3, random_state=42,shuffle=True)
    accuracy=[]
    klist=[1, 3, 5, 7, 9, 11, 13, 15, 17, 21]
    for k in klist:
        knn = KNeighborsClassifier(n_neighbors=k)
        knn.fit(X_train, X_label_train)
        y_pred = knn.predict(X_test)
        print("Confusion Matrix for k=",k,"\n",confusion_matrix(X_label_test, y_pred))
        print("Accuracy for k=:",k,"  ",accuracy_score(X_label_test, y_pred))
        accuracy.append(accuracy_score(X_label_test, y_pred))
    print("\nBest k : ",klist[accuracy.index(max(accuracy))])
    plt.plot(klist,accuracy)
    plt.xlabel("K Values")
    plt.ylabel("Accuracy")
    plt.show()

def Bayes_Classifier(dataframe,clas):
     X=dataframe
     col=X.columns
     X=X[col[1:len(col)-1]]
     
     X_label=dataframe[clas]
     c=dataframe.columns
     X=dataframe[c[:len(c)-1]]
     X_train, X_test, X_label_train, X_label_test =train_test_split(X, X_label, test_size=0.3, random_state=42,shuffle=True)
     accuracy=[]
     model=GaussianNB()
     model.fit(X_train, X_label_train)
     y_pred=model.predict(X_test)
     #y_pred = model.predict(X_test)
     print("Confusion Matrix \n",confusion_matrix(X_label_test, y_pred))
     print("Accuracy  ",metrics.accuracy_score(X_label_test, y_pred))
     accuracy.append(metrics.accuracy_score(X_label_test, y_pred))
     '''
     plt.plot(klist,accuracy)
     plt.xlabel("K Values")
     plt.ylabel("Accuracy")
     plt.show()'''


#### MAIN
df = pd.read_csv(io.BytesIO(uploaded['Batch11.csv']))
#df = pd.read_csv(r'C:\Users\LENOVO\Desktop\Batch11.csv')
#df = pd.read_csv('Batch11.csv')
X = df.drop(columns=['date','Occupancy'])
date, Y = df['date'], df['Occupancy']
col = list(X.columns)
keys = X.keys()
'''
##1. Descriptive Ananlysis + Correlation Analysis
print("Descriptive Analysis\n")
print("\n\n Mean \n",X.mean(),    "\n\n\n\n Median \n",X.median(),    "\n\n\n\n Mode \n",X.mode(),    "\n\n\n\n Std Dev \n",X.std()    )
print('\n\n\n\n Correlation between different attributes \n',X.corr())
print(100*"_")
'''


##2. PreProcessing 
print("\n Removing Outliers \n")

X_std = standardise(df)          ## Standardised X
X_norm = min_max(df)             ## Min Max Normalised X
X_u=replacing_mean(df)           ## Mean replaced X
X_med=replacing_mean(df)         ## Median replaced X



'''
processes=['original','repl_mean','repl_median']

for i in range(len(keys)):
    dff = pd.concat([X[col[i]],   X_u[col[i]],    X_med[col[i]]],    axis=1)
    dff.columns = processes
    dff.to_csv('dff.csv')
    plt.title(col[i])
    dff.boxplot()
    plt.show()

print(100*"_")

'''



'''

print("\n Scaling Data \n")

scales=['original','standardise','min_max']
for i in range(5):
    dff=pd.concat([X[col[i]],   X_std[col[i]],  X_norm[col[i]]],    axis=1)
    dff.columns=scales
    plt.title(col[i])
    dff.boxplot()
    plt.show()

print(100*"_")
'''



print("\n Performing PCA and Classification \n")

##  PCA
for i in range(1,6):
  X_PCA = PCAA(date,X,Y,i)
  X_PCA1=X_PCA.drop(columns=['date','Occupancy'])
  #print(X_PCA1)
  #knn_analysis(X_PCA1,Y)
  #GMM_Analysis(X_PCA1,Y)
  X_PCA1["Occupancy"]=Y
  Bayes_Classifier(X_PCA1,"Occupancy")

print(100*"_")

'''
## KNN 
print("KNN on Standardised Data : ")
knn_analysis(X_std,Y)
print("KNN on Normalised Data : ")
knn_analysis(X_norm,Y)

print(100*"_")
'''

##GMM
#print("Performing GMM Analysis on Raw/Unprocessed Data : ")
#GMM_Analysis(X,Y)
#print("Performing GMM Analysis on Standardised Data : ")
#GMM_Analysis(X_std,Y)
#print("Performing GMM Analysis on Normalised Data : ")
#GMM_Analysis(X_norm,Y)


'''
##GaussianNB
print("GaussianNB on Raw/Unprocessed Data : ")
X["Occupancy"]=Y
Bayes_Classifier(X,"Occupancy")
'''