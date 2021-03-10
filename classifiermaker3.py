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
ebm = db.ebm

test=pd.DataFrame(list(ebm.find({},{"_id": 0, "user_id": 0,'botornot':0,'user':0})))
target=pd.DataFrame(list(ebm.find({},{"_id": 0,'botornot':1})))
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

    xtrain = np.array(xtrain)

    xtest = np.array(xtest)


    ##model=neighbors.KNeighborsClassifier(n_neighbors=55,weights='uniform')
    ##model=svm.SVC(kernel="sigmoid")
    #model=linear_model.LogisticRegression(penalty='none', solver= 'lbfgs')
    #model=tree.DecisionTreeClassifier(criterion= 'gini', max_depth= 9, splitter='best')
    ##model=BernoulliNB()
    ##model= KMeans(n_clusters=3, random_state=0)
    #model= RandomForestClassifier(criterion='entropy', max_features='auto', n_estimators=150)
    model = GradientBoostingClassifier(learning_rate= 0.5, loss='deviance', n_estimators=150)

    model.fit(xtrain, np.ravel(ytrain))
    prediction = model.predict(xtest)
    accuracy = metrics.accuracy_score(ytest, prediction)
    conf = metrics.confusion_matrix(ytest, prediction)
    f1=metrics.f1_score(ytest, prediction)
    sum += accuracy
    print(i, accuracy)
    print('conf:', conf)
    print('f1:', f1)
print('avg acc:', sum / 30)


'''
names=list(db.names.find({}))
for x in names:
    emo=list(db.emotional.find({"user_id":x["userid"]},{"_id":0,"botornot":0}))
    ibm=list(db.ibmusers.find({"user":x["name"]},{"_id":0}))
    print(len(emo),len(ibm))
    if len(emo)>0 and len(ibm)>0:
        for k, v in emo[0].items():
            emo[0][k] = float(v)
        peepee={**emo[0],**ibm[0]}
        db.ebm.insert_one(peepee)
'''