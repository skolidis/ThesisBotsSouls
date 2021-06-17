import pymongo
import matplotlib.pyplot as plt
import numpy as np

conn = pymongo.MongoClient('mongodb://localhost:27017/')
db = conn.tweets
features = db.ibm2

types=['spambot','politicalbot','human','cyborg','selfdeclaredbots','otherbots','socialbot']

# IBM Features
big5=['big5_openness','big5_conscientiousness','big5_extraversion','big5_agreeableness','big5_neuroticism']

# Emotional Features
#big5=["O","C","E","A","N","ANXIETY","AVOIDANCE"]

names=['Openness','Conscientiousness','Extraversion','Agreeableness','Neuroticism',"Anxiety","Avoidance"]
fig = plt.figure()
data=list(features.find({}))
placement=231
i=0
lim=5
for f in big5:
    if i>=5:
        lim=7
    ax2 = fig.add_subplot(placement)
    for t in types:
        ftsb=list(features.find({'bot':t}))
        valuesb=[]
        for v in ftsb:
            valuesb.append(float(v[f])*5)

        valuesb_sorted = np.sort(valuesb)

        pb = 1. * np.arange(len(valuesb)) / (len(valuesb) - 1)
        ax2.plot(valuesb_sorted, pb)
        ax2.set_xlabel(names[i])
        ax2.set_xlim([0, lim])

    i += 1
    placement += 1
plt.legend(['Spambot','Politicalbot','Human','Cyborg','Self Declared Bots','Other bots','Socialbot'],bbox_to_anchor=(1,0.1), loc="lower right",
                bbox_transform=fig.transFigure, ncol=3)
plt.show()