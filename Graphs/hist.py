import pymongo
import numpy
from matplotlib import pyplot as plt

conn = pymongo.MongoClient('mongodb://localhost:27017/')
db = conn.tweets
features = db.ibmusers

# Emotional Features
#big5=["O","C","E","A","N","ANXIETY","AVOIDANCE"]

# IBM Features
big5=['big5_openness','big5_conscientiousness','big5_extraversion','big5_agreeableness','big5_neuroticism']

names=['Openness','Conscientiousness','Extraversion','Agreeableness','Neuroticism',"Anxiety","Avoidance"]
fig = plt.figure()
placement=231
i=0
for typer in big5:
    hist1=[]
    for b in list(features.find({'botornot':"bot"}, {'_id':0,typer:1})):
        hist1.append(float(b[typer])*5)
    hist2=[]
    for h in list(features.find({'botornot':"human"}, {'_id':0,typer:1})):
        hist2.append(float(h[typer])*5)
    bins = numpy.linspace(0, 5, 100)
    ax2 = fig.add_subplot(placement)
    ax2.hist(hist2, bins, alpha=0.5,color ='#5D7199', label='human',edgecolor='black')
    ax2.hist(hist1, bins, alpha=0.5,color ='#BF6F4A', label='bot',edgecolor='black')
    ax2.set_xlabel(names[i])
    i+=1
    placement+=1
plt.legend(loc='upper right')
plt.show()
