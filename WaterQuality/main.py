import pandas as pd
from sklearn.metrics import confusion_matrix
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
import seaborn as sns

from sklearn.ensemble import RandomForestClassifier

from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeClassifier


import matplotlib.pyplot as plt


from sklearn.preprocessing import StandardScaler
from sklearn import metrics

from sklearn import tree
from sklearn import svm
import warnings

from sklearn.model_selection import train_test_split

warnings.filterwarnings('ignore')


#################################################################################################


data = pd.read_csv("water_potability.csv")
print(data.isnull().sum())
sns.countplot(x="Potability", data=data)
plt.show()

print("*********")

from sklearn.impute import SimpleImputer




#pre-proccessing using mean
"""def preprocessing():
    imp= SimpleImputer(strategy= 'mean')
    r= imp.fit_transform(data[['ph']])
    s= imp.fit_transform(data[['Sulfate']])
    t= imp.fit_transform(data[['Trihalomethanes']])

    data['ph']=r
    data['Sulfate']= s
    data['Trihalomethanes']=t
    return data
"""

"""def preprocessing():
    data['ph'].fillna(value=data['ph'].mean(), inplace=True)
    data['Sulfate'].fillna(value=data['Sulfate'].mean(), inplace=True)
    data['Trihalomethanes'].fillna(value=data['Trihalomethanes'].mean(), inplace=True)
    return data"""

#pre-proccessing using median
"""def preprocessing():
    data['ph'].fillna(value=data['ph'].median(), inplace=True)
    data['Sulfate'].fillna(value=data['Sulfate'].median(), inplace=True)
    data['Trihalomethanes'].fillna(value=data['Trihalomethanes'].median(), inplace=True)
    return data
"""
"""def preprocessing():
 nulls = data.dropna()
 return nulls"""



#preprocessing using prediction

def preprocessing():

    dataset = pd.read_csv("water_potability.csv")



    test_data = dataset[dataset["ph"].isnull()]
    test_data.isnull().sum()

    test_data1 = test_data.drop(['Sulfate', 'Trihalomethanes'], axis=1)
    x_test = test_data1.drop('ph', axis=1)
    ph_dataset = dataset.drop(['Sulfate', 'Trihalomethanes'], axis=1)
    ph_dataset.dropna(inplace=True)
    ph_dataset.isnull().sum()

    ph_train = ph_dataset['ph']
    x_train = ph_dataset.drop('ph', axis=1)
    lr = LinearRegression()
    lr.fit(x_train, ph_train)
    ph_pred = lr.predict(x_test)
    test_data.loc[test_data.ph.isnull(), 'ph'] = ph_pred
    test_data.isnull().sum()

    data1 = dataset[dataset['ph'].notna()]
    df = pd.concat([data1, test_data])



    test_data_sulfate = df[df["Sulfate"].isnull()]
    test_data_sulfate.isnull().sum()

    test_data2 = test_data_sulfate.drop(['ph','Trihalomethanes'], axis=1)
    s_dataset = df.drop(['ph', 'Trihalomethanes'], axis=1)
    s_dataset.dropna(inplace=True)
    s_dataset.isnull().sum()

    s_train = s_dataset['Sulfate']
    x_train =s_dataset.drop('Sulfate', axis=1)
    lr = LinearRegression()
    x_test = test_data2.drop('Sulfate', axis=1)
    lr.fit(x_train, s_train)
    s_pred = lr.predict(x_test)
    test_data_sulfate.loc[test_data_sulfate.Sulfate.isnull(), 'Sulfate'] = s_pred
    test_data_sulfate.isnull().sum()

    data2 = df[df['Sulfate'].notna()]
    df1 = pd.concat([data2, test_data_sulfate])


    test_data_tri = df1[df1["Trihalomethanes"].isnull()]
    test_data_tri.isnull().sum()

    test_data3 = test_data_tri.drop(['ph','Sulfate'], axis=1)
    tri_dataset = df1.drop(['ph', 'Sulfate'], axis=1)
    tri_dataset.dropna(inplace=True)
    tri_dataset.isnull().sum()

    t_train = tri_dataset['Trihalomethanes']
    x_train =tri_dataset.drop('Trihalomethanes', axis=1)
    lr = LinearRegression()
    x_test = test_data3.drop('Trihalomethanes', axis=1)
    lr.fit(x_train, t_train)
    t_pred = lr.predict(x_test)
    test_data_tri.loc[test_data_tri.Trihalomethanes.isnull(), 'Trihalomethanes'] = t_pred
    test_data_tri.isnull().sum()

    data3 = df1[df1['Trihalomethanes'].notna()]
    df2 = pd.concat([data3, test_data_tri])

    return df2
#############################


#filtered_Data = preprocessing()
# df_groupin=filteredData.groupby()

#filtered_Data = preprocessing()
#filteredData = filtered_Data.groupby('Potability', group_keys=False).apply(lambda x: x.sample(frac=0.6))


filteredData = preprocessing()
# x= filteredData.drop('Potability', axis=1).copy()
x = filteredData.drop('Potability', axis=1)
# col = ['ph' , 'Hardness' ,'Solids' ,'Chloramines' , 'Sulfate' , 'Conductivity' ,'Organic_carbon' , 'Trihalomethanes' ,'Turbidity']
# x = filteredData[col]
#y= filteredData['Potability'].copy()
y = filteredData.pop('Potability')




#######################################################################



#ways of data splitting
#x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2,random_state=5, stratify=y)
#x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.25)
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.80, random_state=10, shuffle=True)




#Normalize the features
ssc = StandardScaler()

x_trains = ssc.fit_transform(x_train)

x_tests = ssc.transform(x_test)


########################################################################################################


clf = DecisionTreeClassifier(random_state=5, max_depth=10)
clf = clf.fit(x_trains, y_train)
y_pred = clf.predict(x_tests)
print("Decision Tree accuracy:", metrics.accuracy_score(y_pred, y_test) * 100)
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred) * 100)
#Generate the confusion matrix
cf_matrix = confusion_matrix(y_test, y_pred)
print(cf_matrix)

ax = sns.heatmap(cf_matrix/np.sum(cf_matrix), annot=True,  fmt='.2%', cmap='Blues')

ax.set_xlabel('\nPredicted Values')
ax.set_ylabel('Actual Values ')
ax.xaxis.set_ticklabels(['False','True'])
ax.yaxis.set_ticklabels(['False','True'])
plt.show()
print("                    ")
print("***")
print("                    ")
tree.plot_tree(clf)
plt.show()

########################################################################################################################################


gnb = GaussianNB()
gnb.fit(x_trains, y_train)

y_preds = gnb.predict(x_tests)
print("Gaussian Naive Bayes accuracy:", metrics.accuracy_score(y_test, y_preds) * 100)
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_preds) * 100)
#Generate the confusion matrix
cf_matrix = confusion_matrix(y_test, y_pred)
print(cf_matrix)

ax = sns.heatmap(cf_matrix/np.sum(cf_matrix), annot=True,  fmt='.2%', cmap='Blues')

ax.set_xlabel('\nPredicted Values')
ax.set_ylabel('Actual Values ')
ax.xaxis.set_ticklabels(['False','True'])
ax.yaxis.set_ticklabels(['False','True'])
plt.show()
print("                    ")
print("***")
print("                    ")


###########################################################################################################


classifier = KNeighborsClassifier(n_neighbors=5)
classifier.fit(x_trains, y_train)
y_pred = classifier.predict(x_tests)
print("KNN accuracy :", metrics.accuracy_score(y_pred, y_test) * 100)
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred) * 100)
#Generate the confusion matrix
cf_matrix = confusion_matrix(y_test, y_pred)
print(cf_matrix)

ax = sns.heatmap(cf_matrix/np.sum(cf_matrix), annot=True,  fmt='.2%', cmap='Blues')

ax.set_xlabel('\nPredicted Values')
ax.set_ylabel('Actual Values ')
ax.xaxis.set_ticklabels(['False','True'])
ax.yaxis.set_ticklabels(['False','True'])
plt.show()
print("                    ")
print("***")
print("                    ")


###########################################################################################################



RF=RandomForestClassifier(random_state = 42,n_estimators=100)

RF.fit(x_trains,y_train)
y_predt=RF.predict(x_tests)
print("Accuracy of RandomForrest:",metrics.accuracy_score( y_test,y_predt)*100)
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_predt) * 100)
#Generate the confusion matrix
cf_matrix = confusion_matrix(y_test, y_pred)
print(cf_matrix)

ax = sns.heatmap(cf_matrix/np.sum(cf_matrix), annot=True, fmt='.2%', cmap='Blues')

ax.set_xlabel('\nPredicted Values')
ax.set_ylabel('Actual Values ')
ax.xaxis.set_ticklabels(['False','True'])
ax.yaxis.set_ticklabels(['False','True'])
plt.show()

print("                    ")
print("***")
print("                    ")


#######################################################################################################################################


clas = svm.SVC(kernel='linear')
clas.fit(x_trains, y_train)
y_pred = clas.predict(x_tests)
print("Accuracy of SVM:",metrics.accuracy_score( y_test,y_pred)*100)
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred) * 100)
#Generate the confusion matrix
cf_matrix = confusion_matrix(y_test, y_pred)
print(cf_matrix)

ax = sns.heatmap(cf_matrix/np.sum(cf_matrix), annot=True,  fmt='.2%', cmap='Blues')

ax.set_xlabel('\nPredicted Values')
ax.set_ylabel('Actual Values ')
ax.xaxis.set_ticklabels(['False','True'])
ax.yaxis.set_ticklabels(['False','True'])
plt.show()

print("*******")
print(filteredData.isnull().sum())