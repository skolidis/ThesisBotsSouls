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
import numpy as np
from imblearn.over_sampling import ADASYN
from sklearn.model_selection import GridSearchCV

client = MongoClient('mongodb://localhost:27017/')
db = client.tweets

# Emotional Features Model
'''
emo = db.emotional
test=pd.DataFrame(list(emo.find({},{"_id": 0, "user_id": 0,'botornot':0})))
target=pd.DataFrame(list(emo.find({},{"_id": 0,'botornot':1})))
'''

# Conventional Features Model
'''
ilias = db.iliasfeatures
test=pd.DataFrame(list(ilias.find({},{"_id": 0, "user_id": 0,'botornot':0,'user_name':0,"user_screen_name":0,"verified":0, "max_appearance_of_punc_mark":0})))
target=pd.DataFrame(list(ilias.find({},{"_id": 0,'botornot':1})))
'''

# Emotional Features and IBM Model
'''
ebm = db.ebm
test=pd.DataFrame(list(ebm.find({},{"_id": 0, "user_id": 0,'botornot':0,'user':0})))
target=pd.DataFrame(list(ebm.find({},{"_id": 0,'botornot':1})))
'''

#All Features Model
all = db.alltogether
test=pd.DataFrame(list(all.find({},{"_id": 0, "user_id": 0,'botornot':0,'user_name':0,"verified":0,"user_screen_name":0})))
target=pd.DataFrame(list(all.find({},{"_id": 0,'botornot':1})))

sm = ADASYN()
test, target = sm.fit_resample(test, target)
print(len(test))
xtrain, xtest, ytrain, ytest = train_test_split(test, target, test_size=0.2)
labelMappin = {"bot": 0, "human": 1}
ytest['botornot'] = ytest['botornot'].map(labelMappin)
ytest = np.array(ytest)

ytrain['botornot'] = ytrain['botornot'].map(labelMappin)
ytrain = np.array(ytrain)

xtrain = np.array(xtrain)
xtrain = xtrain.astype(float)

xtest = np.array(xtest)
xtest = xtest.astype(float)

# Classification Accuracy

sum=0
for i in range(1,31):
    xtrain, xtest, ytrain, ytest=train_test_split(test,target,test_size=0.2)

    labelMappin={"bot":0,"human":1}
    ytest['botornot']=ytest['botornot'].map(labelMappin)
    ytest=np.array(ytest)

    ytrain['botornot']=ytrain['botornot'].map(labelMappin)
    ytrain=np.array(ytrain)

    xtrain=np.array(xtrain)
    xtrain=xtrain.astype(float)

    xtest=np.array(xtest)
    xtest=xtest.astype(float)

    ##model=neighbors.KNeighborsClassifier(n_neighbors=57,weights='uniform')
    ##model=svm.SVC()
    #model=linear_model.LogisticRegression(penalty='l1', solver= 'liblinear')
    #model=tree.DecisionTreeClassifier(criterion='gini', max_depth= 9, splitter='best')
    ##model=GaussianNB()
    ##model= KMeans(n_clusters=3, random_state=0)
    #model= RandomForestClassifier(criterion='gini', max_features= 'auto',n_estimators=100)
    model=GradientBoostingClassifier(learning_rate= 1.0, loss= 'exponential', n_estimators=150)

    model.fit(xtrain,np.ravel(ytrain))
    prediction=model.predict(xtest)
    accuracy=metrics.accuracy_score(ytest,prediction)
    conf=metrics.confusion_matrix(ytest,prediction)
    f1 = metrics.f1_score(ytest, prediction)
    sum+=accuracy
    print(i,accuracy)
    print('conf:',conf)
    print('f1:',f1)
print('avg acc:',sum/30)

# GridSearch for Classification
'''
parameters = {'penalty':('l1', 'l2', 'elasticnet', 'none'), 'solver':('newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga')}
#parameters = {'criterion':('gini', 'entropy'), 'splitter':('best', 'random'), 'max_depth':list(range(1,10))}
#parameters = {'criterion':('gini', 'entropy'), 'max_features':('auto', 'sqrt', 'log2'), 'n_estimators':[100,150,200]}
#parameters = {'loss':('deviance', 'exponential'), 'n_estimators':[50,100,150], 'learning_rate':[0.1,0.5,1.0]}
lr = GradientBoostingClassifier()
clf = GridSearchCV(lr, parameters)
clf.fit(xtrain,np.ravel(ytrain))
df=pd.DataFrame(clf.cv_results_)
pd.set_option('display.width', 5000)
pd.set_option('display.max_columns', 20)
print(df.sort_values(by=['rank_test_score']))
'''