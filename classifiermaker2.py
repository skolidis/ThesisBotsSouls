import pandas as pd
from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
import pickle
from pymongo import MongoClient
from sklearn.model_selection import train_test_split
from sklearn import neighbors,metrics,svm,linear_model,tree
from sklearn.cluster import KMeans
from sklearn.naive_bayes import GaussianNB,MultinomialNB,ComplementNB,BernoulliNB,CategoricalNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import LabelEncoder
import numpy as np
from imblearn.over_sampling import ADASYN
from sklearn.model_selection import GridSearchCV

client = MongoClient('mongodb://localhost:27017/')
db = client.tweets
ilias = db.iliasfeatures

test=pd.DataFrame(list(ilias.find({},{"_id": 0, "user_id": 0,'botornot':0,'user_name':0,"user_screen_name":0,"max_appearance_of_punc_mark":0})))
target=pd.DataFrame(list(ilias.find({},{"_id": 0,'botornot':1})))
sm = ADASYN()
test, target = sm.fit_resample(test, target)

sum=0
for i in range(1,31):
    xtrain, xtest, ytrain, ytest=train_test_split(test,target,test_size=0.2)

    labelMappin = {"bot": 0, "human": 1}
    ytest['botornot'] = ytest['botornot'].map(labelMappin)
    ytest = np.array(ytest)

    ytrain['botornot'] = ytrain['botornot'].map(labelMappin)
    ytrain = np.array(ytrain)

    xtrain.loc[:, xtrain.dtypes.eq('bool')] = xtrain.loc[:, xtrain.dtypes.eq('bool')].astype(np.int8)
    xtrain=np.array(xtrain)

    xtest.loc[:, xtest.dtypes.eq('bool')] = xtest.loc[:, xtest.dtypes.eq('bool')].astype(np.int8)
    xtest=np.array(xtest)

    ##model=neighbors.KNeighborsClassifier(n_neighbors=57,weights='distance')
    ##model=svm.SVC()
    #model=linear_model.LogisticRegression(penalty= 'l1', solver= 'liblinear')
    #model=tree.DecisionTreeClassifier(criterion= 'gini', max_depth= 9, splitter='best')
    ##model=CategoricalNB()
    ##model= KMeans(n_clusters=4, random_state=0)
    #model= RandomForestClassifier(criterion= 'entropy', max_features= 'sqrt', n_estimators=200)
    model = GradientBoostingClassifier(n_estimators=50, learning_rate=0.1, max_depth=2, random_state=0)

    model.fit(xtrain, np.ravel(ytrain))
    prediction = model.predict(xtest)
    accuracy = metrics.accuracy_score(ytest, prediction)
    conf = metrics.confusion_matrix(ytest, prediction)
    f1 = metrics.f1_score(ytest, prediction)
    sum += accuracy
    print(i, accuracy)
    print('conf:', conf)
    print('f1:', f1)
print('avg acc:',sum/30)
