#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import scipy.stats as st
from sklearn.metrics import r2_score
from sklearn import linear_model
from sklearn.metrics import mean_squared_error as RMSE

import matplotlib.pyplot as plt
import seaborn as sns
plt.style.use('fivethirtyeight')

import os
for dirname, _, filenames in os.walk('/1kodikon'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# In[4]:


data = pd.read_csv("Cropdata.csv")


# In[5]:


x = data.loc[:, ['N','P','K','temperature','ph','humidity','rainfall']].values
print(x.shape)
x_data  = pd.DataFrame(x)
x_data.head()


# In[6]:


from sklearn.cluster import KMeans
plt.rcParams['figure.figsize'] = (10, 4)

wcss = []
for i in range(1, 11):
    km = KMeans(n_clusters = i, init = 'k-means++', max_iter = 500, n_init = 10, random_state = 0)
    km.fit(x)
    wcss.append(km.inertia_)


# In[7]:


# K Means - Clustering analysis
km = KMeans(n_clusters = 4, init = 'k-means++', max_iter = 300, n_init = 10, random_state = 0)
y_means = km.fit_predict(x)

a = data['label']
y_means = pd.DataFrame(y_means)
z = pd.concat([y_means, a], axis = 1)
z = z.rename(columns = {0: 'cluster'})


# In[ ]:





# In[8]:


from sklearn.cluster import AgglomerativeClustering  
hc= AgglomerativeClustering(n_clusters=4, affinity='euclidean', linkage='ward')  
y_her= hc.fit_predict(x)  

b = data['label']
y_herr = pd.DataFrame(y_her)
w = pd.concat([y_herr, b], axis = 1)
w= w.rename(columns = {0: 'cluster'})


# In[9]:


counts = z[z['cluster'] == 0]['label'].value_counts()
d = z.loc[z['label'].isin(counts.index[counts >= 50])]
d = d['label'].value_counts()
counts = z[z['cluster'] == 1]['label'].value_counts()
d = z.loc[z['label'].isin(counts.index[counts >= 50])]
d = d['label'].value_counts()
counts = z[z['cluster'] == 2]['label'].value_counts()
d = z.loc[z['label'].isin(counts.index[counts >= 50])]
d = d['label'].value_counts()
counts = z[z['cluster'] == 3]['label'].value_counts()
d = z.loc[z['label'].isin(counts.index[counts >= 50])]
d = d['label'].value_counts()


# In[10]:


plt.rcParams['figure.figsize'] = (15, 8)


# In[12]:


y = data['label']
x = data.drop(['label'], axis = 1)

print("Shape of x:", x.shape)
print("Shape of y:", y.shape)


# In[13]:


from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 0)
from imblearn.over_sampling import SMOTE
sm = SMOTE(random_state=12)
x_train, y_train = sm.fit_resample(x_train, y_train)


# print("The Shape of x train:", x_train.shape)
# print("The Shape of x test:", x_test.shape)
# print("The Shape of y train:", y_train.shape)
# print("The Shape of y test:", y_test.shape)

# In[15]:


from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import AdaBoostClassifier
from xgboost import XGBClassifier
import xgboost as xgb
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score,confusion_matrix,roc_auc_score
from mlxtend.plotting import plot_confusion_matrix


# In[16]:


def evaluator(y_test, y_pred):    
    
    # Accuracy:
    print('Accuracy is: ', accuracy_score(y_test,y_pred))
    print('')
    # Classification Report:
    print('Classification Report: \n',classification_report(y_test,y_pred))

    print('Confusion Matrix: \n\n')
    plt.style.use("ggplot")
    cm = confusion_matrix(y_test,y_pred)
    plot_confusion_matrix(conf_mat = cm,figsize=(10,10),show_normed=True)
    plt.title('Confusion Matrix for Logistic Regression', fontsize = 15)
    plt.show()


# In[24]:


model_accuracy = pd.DataFrame(columns=['Model','Accuracy'])
models = {
          "KNN" : KNeighborsClassifier(),
          "DT" : DecisionTreeClassifier(),
          'RFC' : RandomForestClassifier(),
          'GBC' : GradientBoostingClassifier(),
          'XGB' : XGBClassifier()
          }


# In[25]:


model_accuracy.sort_values(ascending=False, by = 'Accuracy')


# In[19]:


from sklearn.neighbors import KNeighborsClassifier

kn_classifier = KNeighborsClassifier()

kn_classifier.fit(x_train,y_train)


# In[26]:


pred_kn = kn_classifier.predict(x_test)

evaluator(y_test, pred_kn)


# In[29]:


def pred(n,p,k,temp,hum,ph,rain):
    prediction = kn_classifier.predict((np.array([[n,p,k,temp,hum,ph,rain]])))
    return prediction.tolist()


# In[30]:


print(pred(1,1,1,1,1,1,1))


# In[ ]:




