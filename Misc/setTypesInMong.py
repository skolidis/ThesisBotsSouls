import pickle

import pymongo

conn = pymongo.MongoClient('mongodb://localhost')
db = conn['tweets']

users=db.ibm2.find({})
d = pickle.load(open("userLabels",'rb'))

for user in users:

    myquery = {"user": str(user["user"])}
    newvalues = {"$set": {"bot": d[user["user"]]}}
    print(d[user["user"]])
    db.ibm2.update_many(myquery, newvalues)
