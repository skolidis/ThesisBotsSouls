import pymongo
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.manifold import TSNE

conn = pymongo.MongoClient('mongodb://localhost:27017/')
db = conn.tweets
features = db.ibm2

fts=pd.DataFrame(list(features.find({},{'big5_openness':1,'big5_conscientiousness':1,'big5_extraversion':1,'big5_agreeableness':1,'big5_neuroticism':1})))
fts.drop('_id',axis=1,inplace=True)

types=['spambot','politicalbot','human','cyborg','selfdeclaredbots','otherbots','socialbot']

Z=TSNE(n_components=2).fit_transform(fts)

coloring=list(features.find({}))
ftsb=[]
i=0
for ty in types:
    ftsb = []
    i = 0
    print(ty)
    for c in coloring:
        if c['bot']==ty:
            ftsb.append(Z[i])
        i+=1
    ftsbx=[]
    ftsby=[]
    for f in ftsb:
        ftsbx.append(f[0])
        ftsby.append(f[1])
    plt.scatter(ftsbx,ftsby)
plt.legend(['Spambot','Politicalbot','Human','Cyborg','Self Declared Bots','Other bots','Socialbot'])
plt.show()