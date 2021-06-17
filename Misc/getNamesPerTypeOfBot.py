import pickle
import pymongo
import pandas as pd

d = pickle.load(open('userLabels','rb'))
types=['spambot','politicalbot','human','cyborg','selfdeclaredbots','otherbots','socialbot']
limit=0
li=[]

# Get names per type of bot

for typer in types:
    for key in d:
        if d[key]==typer and limit<400:
            if limit>=200:
                li.append(key)
            limit+=1
    limit=0

pickle.dump(li, open("users","wb"))
