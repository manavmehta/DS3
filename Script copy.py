#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 12 17:54:24 2019

@author: manavmehta
"""
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split as Split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.mixture import GaussianMixture as GMix
from sklearn.metrics import confusion_matrix,accuracy_score
from sklearn.naive_bayes import GaussianNB

## Basics : Date is the first column, Y is the class(Occupancy) and the rest attributes are X :)

def Replace(df, arg):
    X   = df.drop(columns=['date','Occupancy'])
    col = list(X.columns)
    if arg == 'mean':
        tendency = X.mean()
    elif arg == 'median':
        tendency = X.median()
    for i in col:
        q1, q3  = X[i].quantile(0.25), X[i].quantile(0.75)
        iqr     = q3-q1
        lo, hi  = q1-1.5*iqr, q3+1.5*iqr
        X[i]    = np.where(X[i]<=lo,tendency[i],X[i])
        X[i]    = np.where(X[i]>=hi,tendency[i],X[i])
    return X

    
def Standardise(data):         #DONE
    df=data.drop(columns=['date','Occupancy'])
    keys = list(df.keys())
    for i in keys:
        df[i] = (df[i] - df[i].mean())/df[i].std()
    return df


def MinMax(data):     #DONE
    df=data.drop(columns=['date','Occupancy'])
    keys = list(df.keys())
    for i in keys:
        df[i] = (df[i] - df[i].min())/(df[i].max() - df[i].min())  
    return df

def Do_PCA(date,X,Y,d):     #DONE
    Xi=['X'+str(i+1) for i in range(d)]
    print("For Dimensions = ",d)
    X = StandardScaler().fit_transform(X)
    pca = PCA(n_components=d)
    X_PCA = pca.fit_transform(X)
    X_PCA = pd.DataFrame(data = X_PCA, columns = Xi)
    finaldf=pd.concat([date,X_PCA,Y], axis=1)
    return finaldf

def Do_KNN(X,Y):
    X_train, X_test, Y_train, Y_test = Split(X, Y, test_size=0.3, random_state=42,shuffle=True)
    accuracy = []
    klist    = [1, 3, 5, 7, 9, 11, 13, 15, 17, 21]
    for k in klist:
        KNN = KNeighborsClassifier(n_neighbors=k)
        KNN.fit(X_train, Y_train)
        Y_pred = KNN.predict(X_test)
        print("Confusion Matrix for k=",k,"\n",confusion_matrix(Y_test, Y_pred))
        print("Accuracy for k=:",k,"  ",accuracy_score(Y_test, Y_pred))
        accuracy.append(accuracy_score(Y_test, Y_pred))
    print("\nBest k : ",klist[accuracy.index(max(accuracy))])
    plt.plot(klist,accuracy)
    plt.xlabel("K Values")
    plt.ylabel("Accuracy")
    plt.show()

def Do_GMM(X,Y):
    X_train, X_test, Y_train, Y_test = Split(X, Y, test_size=0.3, random_state=42, shuffle=True)
    X0_train, X1_train = X_train[Y_train==0], X_train[Y_train==1]
    Q=[1,2,4,8,16]
    print("\n\nGMM ANALYSIS\n\n")
    for q in Q:
        gmm0,gmm1 = GMix(n_components=q).fit(X0_train),GMix(n_components=q).fit(X1_train)
        Y_pred=[int(gmm0.score_samples(X_test)[i]<gmm1.score_samples(X_test)[i]) for i in range(len(X_test))]
        print("For Q = ", q, " : ")
        print("\nConfusion Matrix : \n",[list(confusion_matrix(Y_test,Y_pred)[i]) for i in range(len(confusion_matrix(Y_test,Y_pred)))])
        print('\nAccurcy Score : ',accuracy_score(Y_test,Y_pred))
        print()

def Do_Bayes(df,clasif):
     X=df
     keys=X.keys()
     X=X[keys[1:len(keys)-1]]
     
     Y=df[clasif]
     c=df.keys()
     X=df[c[:len(c)-1]]
     X_train, X_test, X_label_train, X_label_test =Split(X, Y, test_size=0.3, random_state=42,shuffle=True)
     accuracy=[]
     model=GaussianNB()
     model.fit(X_train, X_label_train)
     y_pred=model.predict(X_test)
     #y_pred = model.predict(X_test)
     print("Confusion Matrix \n",confusion_matrix(X_label_test, y_pred))
     print("Accuracy  ",accuracy_score(X_label_test, y_pred))
     accuracy.append(accuracy_score(X_label_test, y_pred))
     '''
     plt.plot(klist,accuracy)
     plt.xlabel("K Values")
     plt.ylabel("Accuracy")
     plt.show()'''


#### MAIN
df = pd.read_csv('Batch11.csv')
X  = df.drop(columns=['date','Occupancy'])
date, Y = df['date'], df['Occupancy']
keys = X.keys()

##1. Descriptive Ananlysis + Correlation Analysis
print("Descriptive Analysis\n")
print("\n\n Mean \n",X.mean(),    "\n\n\n\n Median \n",X.median(),    "\n\n\n\n Mode \n",X.mode(),    "\n\n\n\n Std Dev \n",X.std()    )
print('\n\n\n\n Correlation between different attributes \n',X.corr())
print(100*"_")


##2. PreProcessing 
print("\n Removing Outliers \n")

X_std  = Standardise(df)              ## Standardised X
X_norm = MinMax(df)                   ## Min Max Normalised X
X_u    = Replace(df, 'mean')          ## Mean replaced X
X_med  = Replace(df, 'median')        ## Median replaced X

processes=['Original','Mean Replaced','Median Replaced']

for i in range(len(keys)):
    dff = pd.concat([X[keys[i]],   X_u[keys[i]],    X_med[keys[i]]],    axis=1)
    dff.columns = processes
    dff.to_csv('dff.csv')
    plt.title(keys[i])
    dff.boxplot()
    plt.show()

print(100*"_")

print("\n Scaling Data \n")

scales=['Original','Standard','MinMax']
for i in range(5):
    dff=pd.concat([X[keys[i]],   X_std[keys[i]],  X_norm[keys[i]]],    axis=1)
    dff.columns=scales
    plt.title(keys[i])
    dff.boxplot()
    plt.show()

print(100*"_")



print("\n Performing PCA and Classification \n")

##3.  PCA
for i in range(1,6):
  X_PCA  = Do_PCA(date,X,Y,i)
  X_PCA1 = X_PCA.drop(columns=["date"])
  Do_Bayes(X_PCA1,"Occupancy")

print(100*"_")

##4. Classification
## KNN 
print("KNN on Standardised Data : ")
Do_KNN(X_std,Y)
print("KNN on Normalised Data : ")
Do_KNN(X_norm,Y)

print(100*"_")

##GMM
print("Performing GMM Analysis on Raw/Unprocessed Data : ")
Do_GMM(X,Y)
print("Performing GMM Analysis on Standardised Data : ")
Do_GMM(X_std,Y)
print("Performing GMM Analysis on Normalised Data : ")
Do_GMM(X_norm,Y)

print(100*"_")

##Gaussian Naive Bayes
print("GaussianNB on Raw Data : ")
X["Occupancy"]=Y
Do_Bayes(X,"Occupancy")

