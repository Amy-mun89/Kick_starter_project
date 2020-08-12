#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 10 05:06:20 2020

@author: Hyemi Mun
"""

#### Import libraries
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns 
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn import tree
import graphviz
from sklearn.tree import DecisionTreeClassifier 
from sklearn import metrics 
from wordcloud import WordCloud


#### Get working directory and prepare the data
os.getcwd()
os.chdir("/Users/appleuser/Downloads/Bitwala_SQL_R_Challenge")
db = pd.read_csv("kickstarter_projects_2018.csv")
pd.set_option('display.max_columns', None)
db.head(10)
db.info()
db['launched']= pd.to_datetime(db['launched'],format='%Y-%m-%d %H:%M:%S')
db['deadline']= pd.to_datetime(db['deadline'],format='%Y-%m-%d %H:%M:%S')
db.info()

#### Column removal based on project understanding
db = db.drop(['ID','currency','pledged','usd pledged'], axis=1)
db.info()

#### Check the missing value
db.isnull().sum()
db = db.fillna('name') # fill the missing value 'name' since only name column has missing value

#### Category Analysis and set the target 
db['state'].value_counts()
successful =['successful']
db['state'] = db['state'].where(db['state'].isin(successful), 'other')
db['state'] = pd.Categorical(db.state)
db['category'].value_counts()
db['main_category'].value_counts()
db['country'].value_counts()

datatype = db.dtypes
cat_column = datatype[ datatype == 'category'].index.tolist()
db_encoded = pd.get_dummies(data=db, columns= cat_column, drop_first=True)
db_encoded.head(10)
db_encoded.info()

#### check the correlation
corrmat = db_encoded.corr() 
print(corrmat)
f, ax = plt.subplots(figsize =(7 , 7)) 
sns.heatmap(corrmat, ax = ax, cmap ="YlGnBu", linewidths = 0.1, annot=True) 


#### relationship between backers and usd_pledged_real
a = sns.scatterplot(x=db.backers, y=db.usd_pledged_real/1e6, hue=db.state, style=db.state, alpha=0.5, hue_order=['other','successful'])
#set the xlim to 50000 backers to get a better overview
a.set(ylim=(0,7.5), xlim=(-1, 50000))
a.set(xlabel='Backers', ylabel='USD Pledged in Million', title= 'Backer vs Pledge')
plt.show()

#compare with original dataset
db0 = pd.read_csv("kickstarter_projects_2018.csv")
a2 = sns.scatterplot(x=db0.backers, y=db0.usd_pledged_real/1e6, hue=db0.state, alpha=0.5)
a2.set(ylim=(0,7.5), xlim=(-1, 50000))
a2.set(xlabel='Backers', ylabel='USD Pledged in Million', title= 'Backer vs Pledge')
plt.show()

#### main category analysis
c0 = sns.countplot(x=db.main_category, order = db['main_category'].value_counts().index)
c0.set(xlabel='Main Category', ylabel='application numbers', title= 'Categories analysis')
plt.show()
c = sns.countplot(x=db.main_category, hue=db.state, order =db['main_category'].value_counts().index)
c.set(xlabel='Main Category', ylabel='application numbers', title= 'Categories analysis')
plt.show()

#### launching hour analysis
db['launched_hour'] =  db.launched.apply(lambda x: x.hour)
db['launched_month'] = db.launched.apply(lambda x: x.month)
h = sns.countplot(x=db.launched_hour, hue = db.state)
m = sns.countplot(x=db.launched_month, hue = db.state)
plt.show()

db.max()

#### goal and pledged money
g = sns.scatterplot(x=db.usd_goal_real/1e6, y=db.usd_pledged_real/1e6, hue=db.state, style=db.state, alpha=0.5, hue_order=['other','successful'])
#set the xlim to 50000 backers to get a better overview
g.set(ylim=(0,None), xlim=(0,None))
g.set(xlabel='Pledged goal', ylabel='USD Pledged in Million', title= 'Backer vs Pledge')
plt.show()

#### name analysis
def show_wordcloud(data, title = None):
    wordcloud = WordCloud(
        background_color='white',
        max_words=100,
        max_font_size=40, 
        scale=3,
        random_state=1 # chosen at random by flipping a coin it was heads
    ).generate(str(data))

    fig = plt.figure(1, figsize=(12, 12))
    plt.axis('off')
    if title: 
        fig.suptitle(title, fontsize=20)
        fig.subplots_adjust(top=2.3)

    plt.imshow(wordcloud)
    plt.show()

show_wordcloud(db[db.state == 'successful']['name'])
show_wordcloud(db[db.state == 'other']['name'])

#### numbers of words in the name 
db['name_length'] = db['name'].str.len()
db['name_words'] = db.name.apply(lambda x: len(str(x).split(' ')))
db.head(5)
w = sns.countplot(x=db.name_words, hue = db.state)
plt.show()
w = sns.countplot(x=db.name_words, hue = db.state)
w.set(ylim=(0,None), xlim=(-1,None))
w.set(xlabel='Numbers of words', ylabel='application counts', title= 'words numbers and success')
plt.show()


#### lengths of the projects
db['project_time'] = (db['deadline'] -db['launched']).astype('timedelta64[D]').astype(int)
db['project_weeks']= round(db['project_time']/7)
db.head(5)
t = sns.countplot(db.project_weeks, hue= db.state)
t.set(ylim=(0,None), xlim=(-1,10))
t.set(xlabel='length of projects', ylabel='application counts', title= 'project time and success')
plt.show()


#### build up a prediction model - Decision Tree
db_tree = db.drop(['backers','usd_pledged_real','name','category','deadline','goal','launched','country','name_length','project_time','usd_goal_real'], axis=1)
db_tree.info()
db_tree['main_category'] = pd.Categorical(db.main_category)
db_tree['main_category'].value_counts()
datatype = db_tree.dtypes
cat_columns = datatype[ datatype == 'category'].index.tolist()


db_tree_encoded = pd.get_dummies(data=db_tree, columns= cat_columns, drop_first=True)
db_tree_encoded.head(10)

dbt = db_tree_encoded
clf = DecisionTreeClassifier(max_depth=5)
dbt.head()

x = dbt.drop(['state_successful'], axis = 1)
y = dbt['state_successful']

#Split dataset into training set and test set 
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=1)
# Train Decision Tree Classifer
clf = clf.fit(x_train,y_train)
#Predict the response for test dataset
y_pred = clf.predict(x_test)
dot_data = tree.export_graphviz(clf, class_names=['fail', 'success'],
                                out_file=None,filled=True, rounded=True, proportion=True, label="root",
                                feature_names = x.columns)
graph = graphviz.Source(dot_data)
graph 


#### model evaluation
print("<Model Evaluation>","\n",metrics.classification_report(y_test, y_pred),"\n"
      "<AUC Score>","\n",metrics.roc_auc_score(y_test,y_pred))
fpr, tpr, thresholds = metrics.roc_curve(y_test,y_pred)
auc=metrics.roc_auc_score(y_test,y_pred)
print(plt.plot([0, 1], [0, 1], linestyle='--'),plt.plot(fpr, tpr, marker='.'),
plt.plot(fpr, tpr, marker='.'),
plt.title('ROC Curve'),
plt.xlabel('TPR'),
plt.ylabel('FPR'),
plt.grid(),
plt.legend(["AUC=%.3f"%auc]))
confusion_matrix(y_test, y_pred)

