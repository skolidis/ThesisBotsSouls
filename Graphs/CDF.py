import numpy as np
import matplotlib.pyplot as plt
import pymongo
import scipy
import seaborn as sns
import pandas as pd
from sklearn.manifold import TSNE

conn = pymongo.MongoClient('mongodb://localhost:27017/')
db = conn.tweets
features = db.ebm

# Emotional Features
#big5=["O","C","E","A","N","ANXIETY","AVOIDANCE"]

# IBM Features
big5=['big5_openness','big5_conscientiousness','big5_extraversion','big5_agreeableness','big5_neuroticism']

names=['Openness','Conscientiousness','Extraversion','Agreeableness','Neuroticism',"Anxiety","Avoidance"]
fig = plt.figure()
placement=231
i=0
lim=5

ftsb=list(features.find({'botornot':"bot"}))
ftsh=list(features.find({'botornot':"human"}))

for typer in big5:
    if i>=5:
        lim=7
    valuesb=[]
    valuesh=[]
    for f in ftsb:
        valuesb.append(float(f[typer])*lim)
    for f in ftsh:
        valuesh.append(float(f[typer])*lim)
    valuesb_sorted = np.sort(valuesb)
    valuesh_sorted = np.sort(valuesh)
    pb = 1. * np.arange(len(valuesb)) / (len(valuesb) - 1)
    ph= 1. * np.arange(len(valuesh)) / (len(valuesh) - 1)

    ax2 = fig.add_subplot(placement)
    ax2.plot(valuesb_sorted, pb)
    ax2.plot(valuesh_sorted,ph, linestyle='dashed')
    ax2.set_xlabel(names[i])
    ax2.set_ylabel('$p$')
    plt.xlim([0, lim])
    i+=1
    placement+=1
plt.show()

