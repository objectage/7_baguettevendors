#!/usr/bin/env python
# coding: utf-8

# In[1]:


# for manipulations
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import scipy.stats as st
from sklearn.metrics import r2_score
from sklearn import linear_model
from sklearn.metrics import mean_squared_error as RMSE

# for data visualizations
import matplotlib.pyplot as plt
import seaborn as sns
plt.style.use('fivethirtyeight')

# for interactivity
import ipywidgets
from ipywidgets import interact

import os
for dirname, _, filenames in os.walk('/1kodikon'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# In[2]:


data = pd.read_csv("Cropdata.csv")
print(data.shape)
data.head()


# ## Description for each of the columns in the Dataset
# 
# N - ratio of Nitrogen content in soil
# P - ratio of Phosphorous content in soil
# K - ration of Potassium content in soil
# temperature - temperature in degree Celsius
# humidity - relative humidity in %
# ph - ph value of the soil
# rainfall - rainfall in mm

# In[3]:


data.isnull().sum()


# #Dwscriptive stats:

# In[4]:


print("Average Ratio of Nitrogen in the Soil : {0:.2f}".format(data['N'].mean()))
print("Average Ratio of Phosphorous in the Soil : {0:.2f}".format(data['P'].mean()))
print("Average Ratio of Potassium in the Soil : {0:.2f}".format(data['K'].mean()))
print("Average Tempature in Celsius : {0:.2f}".format(data['temperature'].mean()))
print("Average Relative Humidity in % : {0:.2f}".format(data['humidity'].mean()))
print("Average PH Value of the soil : {0:.2f}".format(data['ph'].mean()))
print("Average Rainfall in mm : {0:.2f}".format(data['rainfall'].mean()))


# In[5]:


data['label'].value_counts()


# In[6]:


@interact
def summary(crops = list(data['label'].value_counts().index)):
    x = data[data['label'] == crops]
    print("---------------------------------------------")
    print("Nitrogen-")
    print("min:", x['N'].min())
    print("avg:", x['N'].mean())
    print("max:", x['N'].max()) 
    print("---------------------------------------------")
    print("Phosphorous-")
    print("min:", x['P'].min())
    print("avg:", x['P'].mean())
    print("max:", x['P'].max()) 
    print("---------------------------------------------")
    print("Potassium-")
    print("min:", x['K'].min())
    print("avg:", x['K'].mean())
    print("max:", x['K'].max()) 
    print("---------------------------------------------")
    print("Temperature-")
    print("min: {0:.2f}".format(x['temperature'].min()))
    print("avg: {0:.2f}".format(x['temperature'].mean()))
    print("max: {0:.2f}".format(x['temperature'].max()))
    print("---------------------------------------------")
    print("Humidity-")
    print("min: {0:.2f}".format(x['humidity'].min()))
    print("avg: {0:.2f}".format(x['humidity'].mean()))
    print("max: {0:.2f}".format(x['humidity'].max()))
    print("---------------------------------------------")
    print("PH-")
    print("min: {0:.2f}".format(x['ph'].min()))
    print("avg: {0:.2f}".format(x['ph'].mean()))
    print("max: {0:.2f}".format(x['ph'].max()))
    print("---------------------------------------------")
    print("Rainfall-")
    print("min: {0:.2f}".format(x['rainfall'].min()))
    print("avg: {0:.2f}".format(x['rainfall'].mean()))
    print("max: {0:.2f}".format(x['rainfall'].max()))

rice={'N':{'min':60,'avg':79.80,'max':99},'P':{'min':35,'avg':47.58,'max':60},'K':{'min':35,'avg':39.87,'max':45},'temp':{'min':20.05,'avg':23.69,'max':36.93},'hum':{'min':80.12,'avg':82.27,'max':84.47},'ph':{'min':5.01,'avg':6.43,'max':7.87},'rain':{'min':182.56,'avg':236.18,'max':298.56}}
maize={'N':{'min'60:,'avg':77.76,'max':100},'P':{'min':35,'avg':48.44,'max':60},'K':{'min':15,'avg':19.79,'max':25},'temp':{'min':18.04,'avg':22.39,'max':26.55},'hum':{'min':55.28,'avg':65.09,'max':74.83},'ph':{'min':5.51,'avg':6.25,'max':7.00},'rain':{'min':60.65,'avg':84.77,'max':109.75}}
jute={'N':{'min':60,'avg':78.4,'max':100},'P':{'min':35,'avg':46.86,'max':60},'K':{'min':35,'avg':39.99,'max':45},'temp':{'min':23.09,'avg':24.96,'max':26.99},'hum':{'min':70.88,'avg':79.64,'max':89.89},'ph':{'min':6.00,'avg':6.73,'max':},'rain':{'min':,'avg':,'max':}}
cotton={'N':{'min':,'avg':,'max':},'P':{'min':,'avg':,'max':},'K':{'min':,'avg':,'max':},'temp':{'min':,'avg':,'max':},'hum':{'min':,'avg':,'max':},'ph':{'min':,'avg':,'max':},'rain':{'min':,'avg':,'max':}}
coconut={'N':{'min':,'avg':,'max':},'P':{'min':,'avg':,'max':},'K':{'min':,'avg':,'max':},'temp':{'min':,'avg':,'max':},'hum':{'min':,'avg':,'max':},'ph':{'min':,'avg':,'max':},'rain':{'min':,'avg':,'max':}}
papaya={'N':{'min':,'avg':,'max':},'P':{'min':,'avg':,'max':},'K':{'min':,'avg':,'max':},'temp':{'min':,'avg':,'max':},'hum':{'min':,'avg':,'max':},'ph':{'min':,'avg':,'max':},'rain':{'min':,'avg':,'max':}}
orange={'N':{'min':,'avg':,'max':},'P':{'min':,'avg':,'max':},'K':{'min':,'avg':,'max':},'temp':{'min':,'avg':,'max':},'hum':{'min':,'avg':,'max':},'ph':{'min':,'avg':,'max':},'rain':{'min':,'avg':,'max':}}
apple={'N':{'min':,'avg':,'max':},'P':{'min':,'avg':,'max':},'K':{'min':,'avg':,'max':},'temp':{'min':,'avg':,'max':},'hum':{'min':,'avg':,'max':},'ph':{'min':,'avg':,'max':},'rain':{'min':,'avg':,'max':}}
muskmelon={'N':{'min':,'avg':,'max':},'P':{'min':,'avg':,'max':},'K':{'min':,'avg':,'max':},'temp':{'min':,'avg':,'max':},'hum':{'min':,'avg':,'max':},'ph':{'min':,'avg':,'max':},'rain':{'min':,'avg':,'max':}}
watermelon={'N':{'min':,'avg':,'max':},'P':{'min':,'avg':,'max':},'K':{'min':,'avg':,'max':},'temp':{'min':,'avg':,'max':},'hum':{'min':,'avg':,'max':},'ph':{'min':,'avg':,'max':},'rain':{'min':,'avg':,'max':}}
grapes={'N':{'min':,'avg':,'max':},'P':{'min':,'avg':,'max':},'K':{'min':,'avg':,'max':},'temp':{'min':,'avg':,'max':},'hum':{'min':,'avg':,'max':},'ph':{'min':,'avg':,'max':},'rain':{'min':,'avg':,'max':}}
mango={'N':{'min':,'avg':,'max':},'P':{'min':,'avg':,'max':},'K':{'min':,'avg':,'max':},'temp':{'min':,'avg':,'max':},'hum':{'min':,'avg':,'max':},'ph':{'min':,'avg':,'max':},'rain':{'min':,'avg':,'max':}}
banana={'N':{'min':,'avg':,'max':},'P':{'min':,'avg':,'max':},'K':{'min':,'avg':,'max':},'temp':{'min':,'avg':,'max':},'hum':{'min':,'avg':,'max':},'ph':{'min':,'avg':,'max':},'rain':{'min':,'avg':,'max':}}
pomegranate={'N':{'min':,'avg':,'max':},'P':{'min':,'avg':,'max':},'K':{'min':,'avg':,'max':},'temp':{'min':,'avg':,'max':},'hum':{'min':,'avg':,'max':},'ph':{'min':,'avg':,'max':},'rain':{'min':,'avg':,'max':}}
lentil={'N':{'min':,'avg':,'max':},'P':{'min':,'avg':,'max':},'K':{'min':,'avg':,'max':},'temp':{'min':,'avg':,'max':},'hum':{'min':,'avg':,'max':},'ph':{'min':,'avg':,'max':},'rain':{'min':,'avg':,'max':}}
blackgram={'N':{'min':,'avg':,'max':},'P':{'min':,'avg':,'max':},'K':{'min':,'avg':,'max':},'temp':{'min':,'avg':,'max':},'hum':{'min':,'avg':,'max':},'ph':{'min':,'avg':,'max':},'rain':{'min':,'avg':,'max':}}
mungbean={'N':{'min':,'avg':,'max':},'P':{'min':,'avg':,'max':},'K':{'min':,'avg':,'max':},'temp':{'min':,'avg':,'max':},'hum':{'min':,'avg':,'max':},'ph':{'min':,'avg':,'max':},'rain':{'min':,'avg':,'max':}}
mothbeans={'N':{'min':,'avg':,'max':},'P':{'min':,'avg':,'max':},'K':{'min':,'avg':,'max':},'temp':{'min':,'avg':,'max':},'hum':{'min':,'avg':,'max':},'ph':{'min':,'avg':,'max':},'rain':{'min':,'avg':,'max':}}
pigeonpeas={'N':{'min':,'avg':,'max':},'P':{'min':,'avg':,'max':},'K':{'min':,'avg':,'max':},'temp':{'min':,'avg':,'max':},'hum':{'min':,'avg':,'max':},'ph':{'min':,'avg':,'max':},'rain':{'min':,'avg':,'max':}}
kidneybeans={'N':{'min':,'avg':,'max':},'P':{'min':,'avg':,'max':},'K':{'min':,'avg':,'max':},'temp':{'min':,'avg':,'max':},'hum':{'min':,'avg':,'max':},'ph':{'min':,'avg':,'max':},'rain':{'min':,'avg':,'max':}}
chickpea={'N':{'min':,'avg':,'max':},'P':{'min':,'avg':,'max':},'K':{'min':,'avg':,'max':},'temp':{'min':,'avg':,'max':},'hum':{'min':,'avg':,'max':},'ph':{'min':,'avg':,'max':},'rain':{'min':,'avg':,'max':}}
coffee={'N':{'min':,'avg':,'max':},'P':{'min':,'avg':,'max':},'K':{'min':,'avg':,'max':},'temp':{'min':,'avg':,'max':},'hum':{'min':,'avg':,'max':},'ph':{'min':,'avg':,'max':},'rain':{'min':,'avg':,'max':}}
# In[8]:


## the Average Requirement for each crops with average conditions

@interact
def compare(conditions = ['N','P','K','temperature','ph','humidity','rainfall']):
    print("Average Value for", conditions,"is {0:.2f}".format(data[conditions].mean()))
    print("----------------------------------------------")
    print("Rice : {0:.2f}".format(data[(data['label'] == 'rice')][conditions].mean()))
    print("Black Grams : {0:.2f}".format(data[data['label'] == 'blackgram'][conditions].mean()))
    print("Banana : {0:.2f}".format(data[(data['label'] == 'banana')][conditions].mean()))
    print("Jute : {0:.2f}".format(data[data['label'] == 'jute'][conditions].mean()))
    print("Coconut : {0:.2f}".format(data[(data['label'] == 'coconut')][conditions].mean()))
    print("Apple : {0:.2f}".format(data[data['label'] == 'apple'][conditions].mean()))
    print("Papaya : {0:.2f}".format(data[(data['label'] == 'papaya')][conditions].mean()))
    print("Muskmelon : {0:.2f}".format(data[data['label'] == 'muskmelon'][conditions].mean()))
    print("Grapes : {0:.2f}".format(data[(data['label'] == 'grapes')][conditions].mean()))
    print("Watermelon : {0:.2f}".format(data[data['label'] == 'watermelon'][conditions].mean()))
    print("Kidney Beans: {0:.2f}".format(data[(data['label'] == 'kidneybeans')][conditions].mean()))
    print("Mung Beans : {0:.2f}".format(data[data['label'] == 'mungbean'][conditions].mean()))
    print("Oranges : {0:.2f}".format(data[(data['label'] == 'orange')][conditions].mean()))
    print("Chick Peas : {0:.2f}".format(data[data['label'] == 'chickpea'][conditions].mean()))
    print("Lentils : {0:.2f}".format(data[(data['label'] == 'lentil')][conditions].mean()))
    print("Cotton : {0:.2f}".format(data[data['label'] == 'cotton'][conditions].mean()))
    print("Maize : {0:.2f}".format(data[(data['label'] == 'maize')][conditions].mean()))
    print("Moth Beans : {0:.2f}".format(data[data['label'] == 'mothbeans'][conditions].mean()))
    print("Pigeon Peas : {0:.2f}".format(data[(data['label'] == 'pigeonpeas')][conditions].mean()))
    print("Mango : {0:.2f}".format(data[data['label'] == 'mango'][conditions].mean()))
    print("Pomegranate : {0:.2f}".format(data[(data['label'] == 'pomegranate')][conditions].mean()))
    print("Coffee : {0:.2f}".format(data[data['label'] == 'coffee'][conditions].mean()))


# In[9]:


@interact
def compare(conditions = ['N','P','K','temperature','ph','humidity','rainfall']):
    print("Crops which require greater than average", conditions,'\n')
    print(data[data[conditions] > data[conditions].mean()]['label'].unique())
    print("----------------------------------------------")
    print("Crops which require less than average", conditions,'\n')
    print(data[data[conditions] <= data[conditions].mean()]['label'].unique())


# ### Analyzing Agricultural Conditions

# In[10]:


x=data['N'].tolist()
y=data['ph'].tolist()
plt.plot(x, y, 'ro')
b1,b0=np.polyfit(x,y,deg=1)
yp=b0+b1*np.array(data['N'])
plt.plot(x,yp)
plt.show()


# In[11]:


x=data['P'].tolist()
y=data['ph'].tolist()
plt.plot(x, y, 'ro')
b1,b0=np.polyfit(x,y,deg=1)
yp=b0+b1*np.array(data['P'])
plt.plot(x,yp)
plt.show()


# In[12]:


x=data['K'].tolist()
y=data['ph'].tolist()
plt.plot(x, y, 'ro')
b1,b0=np.polyfit(x,y,deg=1)
yp=b0+b1*np.array(data['K'])
plt.plot(x,yp)
plt.show()


# In[13]:


x = data['ph'].tolist()

n, bins, patches = plt.hist(x, 20, density = 1, facecolor = 'blue', alpha = 0.5)

y=st.norm.pdf(bins,data['ph'].mean(),data['ph'].std())

plt.plot(bins,y,'r--')
plt.xlabel('pH')
plt.ylabel('Probabilty')


# In[14]:


#distribution of Agricultural Conditions :

plt.rcParams['figure.figsize'] = (15, 7)

plt.subplot(2, 4, 1)
sns.distplot(data['N'], color = 'lightgrey')
plt.xlabel('Ratio of Nitrogen', fontsize = 12)
plt.grid()

plt.subplot(2, 4, 2)
sns.distplot(data['P'], color = 'red')
plt.xlabel('Ratio of Phosphorous', fontsize = 12)
plt.grid()

plt.subplot(2, 4, 3)
sns.distplot(data['K'], color ='yellow')
plt.xlabel('Ratio of Potassium', fontsize = 12)
plt.grid()

plt.subplot(2, 4, 4)
sns.distplot(data['temperature'], color = 'violet')
plt.xlabel('Temperature', fontsize = 12)
plt.grid()

plt.subplot(2, 4, 5)
sns.distplot(data['rainfall'], color = 'darkblue')
plt.xlabel('Rainfall', fontsize = 12)
plt.grid()

plt.subplot(2, 4, 6)
sns.distplot(data['humidity'], color = 'lightblue')
plt.xlabel('Humidity', fontsize = 12)
plt.grid()

plt.subplot(2, 4, 7)
sns.distplot(data['ph'], color = 'darkgreen')
plt.xlabel('pH Level', fontsize = 12)
plt.grid()

plt.suptitle('Distribution for Agricultural Conditions', fontsize = 20)
plt.show()


# In[15]:


print("Crops - High Ratio of Nitrogen Content in Soil:", data[data['N'] > 120]['label'].unique())
print("Crops - High Ratio of Phosphorous Content in Soil:", data[data['P'] > 100]['label'].unique())
print("Crops - High Ratio of Potassium Content in Soil:", data[data['K'] > 200]['label'].unique())
print("Crops - High Rainfall:", data[data['rainfall'] > 200]['label'].unique())
print("Crops - Low Temperature :", data[data['temperature'] < 10]['label'].unique())
print("Crops - High Temperature :", data[data['temperature'] > 40]['label'].unique())
print("Crops - Low Humidity:", data[data['humidity'] < 20]['label'].unique())
print("Crops - Low pH:", data[data['ph'] < 4]['label'].unique())
print("Crops - High pH:", data[data['ph'] > 9]['label'].unique())


# In[16]:


print("Crops in Karnataka: ")
print("Summer Crops")
print(data[(data['temperature'] > 32) & (data['humidity'] > 62)]['label'].unique())
print("Winter Crops")
print(data[(data['temperature'] < 25) & (data['humidity'] > 30)]['label'].unique())
print("Monsoon Crops")
print(data[(data['rainfall'] > 200) & (data['humidity'] > 50)]['label'].unique())


# In[17]:


x = data.loc[:, ['N','P','K','temperature','ph','humidity','rainfall']].values
print(x.shape)
x_data  = pd.DataFrame(x)
x_data.head()


# In[18]:


from sklearn.cluster import KMeans
plt.rcParams['figure.figsize'] = (10, 4)

wcss = []
for i in range(1, 11):
    km = KMeans(n_clusters = i, init = 'k-means++', max_iter = 500, n_init = 10, random_state = 0)
    km.fit(x)
    wcss.append(km.inertia_)

plt.plot(range(1, 11), wcss)
plt.title('The Elbow Method', fontsize = 20)
plt.xlabel('No. of Clusters')
plt.ylabel('wcss')
plt.show()


# In[19]:


# K Means - Clustering analysis
km = KMeans(n_clusters = 4, init = 'k-means++', max_iter = 300, n_init = 10, random_state = 0)
y_means = km.fit_predict(x)

a = data['label']
y_means = pd.DataFrame(y_means)
z = pd.concat([y_means, a], axis = 1)
z = z.rename(columns = {0: 'cluster'})

#Clusters of each Crops
print("1st Cluster:", z[z['cluster'] == 0]['label'].unique())
print('\n')
print("2nd Cluster:", z[z['cluster'] == 1]['label'].unique())
print('\n')
print("3rd Cluster:", z[z['cluster'] == 2]['label'].unique())
print('\n')
print("4th Cluster:", z[z['cluster'] == 3]['label'].unique())


# In[20]:


import scipy.cluster.hierarchy as shc  
dendro = shc.dendrogram(shc.linkage(x, method="ward"))


# In[21]:


#training the hierarchical model  
from sklearn.cluster import AgglomerativeClustering  
hc= AgglomerativeClustering(n_clusters=4, affinity='euclidean', linkage='ward')  
y_her= hc.fit_predict(x)  

b = data['label']
y_herr = pd.DataFrame(y_her)
w = pd.concat([y_herr, b], axis = 1)
w= w.rename(columns = {0: 'cluster'})

print("Hierachical Clustering Analysis : \n")
print("Zero Cluster:", w[w['cluster'] == 0]['label'].unique())
print("First Cluster:", w[w['cluster'] == 1]['label'].unique())
print("Second Cluster:", w[w['cluster'] == 2]['label'].unique())
print("Third Cluster:", w[w['cluster'] == 3]['label'].unique())


# In[22]:


# Hard Clustering

print("Results for Hard Clustering\n")
counts = z[z['cluster'] == 0]['label'].value_counts()
d = z.loc[z['label'].isin(counts.index[counts >= 50])]
d = d['label'].value_counts()
print("Crops in Cluster 1:", list(d.index))
print("--------------------------------------------------")
counts = z[z['cluster'] == 1]['label'].value_counts()
d = z.loc[z['label'].isin(counts.index[counts >= 50])]
d = d['label'].value_counts()
print("Crops in Cluster 2:", list(d.index))
print("--------------------------------------------------")
counts = z[z['cluster'] == 2]['label'].value_counts()
d = z.loc[z['label'].isin(counts.index[counts >= 50])]
d = d['label'].value_counts()
print("Crops in Cluster 3:", list(d.index))
print("--------------------------------------------------")
counts = z[z['cluster'] == 3]['label'].value_counts()
d = z.loc[z['label'].isin(counts.index[counts >= 50])]
d = d['label'].value_counts()
print("Crops in Cluster 4:", list(d.index))


# In[23]:


data.head()


# In[24]:


plt.rcParams['figure.figsize'] = (15, 8)


# In[25]:


plt.subplot(2, 4, 1)
sns.barplot(data['N'], data['label'])
plt.ylabel(' ')
plt.xlabel('Ratio of Nitrogen', fontsize = 10)
plt.yticks(fontsize = 10)


# In[26]:


plt.subplot(2, 4, 1)
sns.barplot(data['N'], data['label'])
plt.ylabel(' ')
plt.xlabel('Ratio of Nitrogen', fontsize = 10)
plt.yticks(fontsize = 10)


# In[27]:


plt.subplot(2, 4, 4)
sns.barplot(data['temperature'], data['label'])
plt.ylabel(' ')
plt.xlabel('Temperature', fontsize = 10)
plt.yticks(fontsize = 10)


# In[28]:


plt.subplot(2, 4, 5)
sns.barplot(data['humidity'], data['label'])
plt.ylabel(' ')
plt.xlabel('Humidity', fontsize = 10)
plt.yticks(fontsize = 10)


# In[29]:


plt.subplot(2, 4, 6)
sns.barplot(data['ph'], data['label'])
plt.ylabel(' ')
plt.xlabel('pH of Soil', fontsize = 10)
plt.yticks(fontsize = 10)


# In[30]:


plt.subplot(2, 4, 7)
sns.barplot(data['rainfall'], data['label'])
plt.ylabel(' ')
plt.xlabel('Rainfall', fontsize = 10)
plt.yticks(fontsize = 10)


# In[31]:


y = data['label']
x = data.drop(['label'], axis = 1)

print("Shape of x:", x.shape)
print("Shape of y:", y.shape)


# In[76]:


from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 0)
from imblearn.over_sampling import SMOTE
sm = SMOTE(random_state=12)
x_train, y_train = sm.fit_resample(x_train, y_train)

print("The Shape of x train:", x_train.shape)
print("The Shape of x test:", x_test.shape)
print("The Shape of y train:", y_train.shape)
print("The Shape of y test:", y_test.shape)


# In[77]:


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


# In[78]:


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


# In[79]:


# lets create a Predictive Models

model_accuracy = pd.DataFrame(columns=['Model','Accuracy'])
models = {
          "KNN" : KNeighborsClassifier(),
          "DT" : DecisionTreeClassifier(),
          'RFC' : RandomForestClassifier(),
          'GBC' : GradientBoostingClassifier(),
          'XGB' : XGBClassifier()
          }

for test, clf in models.items():
    clf.fit(x_train, y_train)
    y_pred = clf.predict(x_test)
    acc = accuracy_score(y_test,y_pred)
    train_pred = clf.predict(x_train)
    train_acc = accuracy_score(y_train, train_pred)
    print("\n", test + ' scores')
    print(acc)
    print(classification_report(y_test,y_pred))
    print(confusion_matrix(y_test,y_pred))
    print('*' * 100,"\n")
    model_accuracy = model_accuracy.append({'Model': test, 'Accuracy': acc, 'Train_acc': train_acc}, ignore_index=True)


# In[ ]:


model_accuracy.sort_values(ascending=False, by = 'Accuracy')


# In[80]:


from sklearn.neighbors import KNeighborsClassifier

kn_classifier = KNeighborsClassifier()

kn_classifier.fit(x_train,y_train)


# In[81]:


pred_kn = kn_classifier.predict(x_test)

evaluator(y_test, pred_kn)


# In[82]:



prediction = kn_classifier.predict((np.array([[90,
                                       40,
                                       40,
                                       20,
                                       80,
                                       7,
                                       200]])))
print("The Suggested Crop for Given Climatic Condition is :", prediction)


# In[83]:


def pred(n,p,k,temp,hum,ph,rain):
    prediction = kn_classifier.predict((np.array([[n,p,k,temp,hum,ph,rain]])))
    print("The Suggested Crop for Given Climatic Condition is :", prediction)
    return prediction


# In[84]:


pred(90,40,40,20,80,7,200)


# In[74]:





# In[ ]:





# In[ ]:





# In[ ]:




