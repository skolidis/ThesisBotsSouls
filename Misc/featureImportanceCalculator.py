from pymongo import MongoClient
import pandas as pd
from imblearn.over_sampling import ADASYN
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
from sklearn import neighbors,metrics
import warnings
warnings.filterwarnings("ignore")

client = MongoClient('mongodb://localhost:27017/')
db = client.tweets
all = db.alltogether

test=pd.DataFrame(list(all.find({},{"_id": 0, "user_id": 0,'botornot':0,'user_name':0,"verified":0,"user_screen_name":0})))
target=pd.DataFrame(list(all.find({},{"_id": 0,'botornot':1})))
sm= ADASYN()
test, target = sm.fit_resample(test, target)

todrop=[]
numbers=[]
n=360
flag=False
while len(test.columns)>0:

    # iterate to calculate accuracy
    sum = 0
    for i in range(1, 31):
        xtrain, xtest, ytrain, ytest = train_test_split(test, target, test_size=0.2)
        labelMappin = {"bot": 0, "human": 1}
        ytest['botornot'] = ytest['botornot'].map(labelMappin)
        ytest = np.array(ytest)

        ytrain['botornot'] = ytrain['botornot'].map(labelMappin)
        ytrain = np.array(ytrain)

        xtrain.loc[:, xtrain.dtypes.eq('bool')] = xtrain.loc[:, xtrain.dtypes.eq('bool')].astype(np.int8)
        xtrain = np.array(xtrain)

        xtest.loc[:, xtest.dtypes.eq('bool')] = xtest.loc[:, xtest.dtypes.eq('bool')].astype(np.int8)
        xtest = np.array(xtest)


        model= RandomForestClassifier(criterion= 'entropy', max_features= 'sqrt', n_estimators= 100)


        model.fit(xtrain, np.ravel(ytrain))
        prediction = model.predict(xtest)

        accuracy = metrics.accuracy_score(ytest, prediction)
        sum += accuracy

    # find the currently most important features
    important_features_dict = {}
    for idx, val in enumerate(model.feature_importances_):
        important_features_dict[idx] = val

    important_features_list = sorted(important_features_dict,
                                     key=important_features_dict.get,
                                     reverse=True)

    # Iterate for the last number of Features one by one
    '''
    if flag:
        todrop = important_features_list[:360]
        test.drop(test.columns[todrop], axis=1, inplace=True)
        flag=False
        continue
    '''
    todrop = important_features_list[:10]
    test.drop(test.columns[todrop], axis=1, inplace=True)
    numbers.append(sum/30)

print(numbers)