from pymongo import MongoClient
import pandas as pd

client = MongoClient('mongodb://localhost:27017/')
db = client.tweets
profiles=list(db.ibmusers.find())
botE=0
humanE=0
i=0
j=0

# Emotional Features
#big5=["O","C","E","A","N","ANXIETY","AVOIDANCE"]

# IBM Features
big5=['big5_openness','big5_conscientiousness','big5_extraversion','big5_agreeableness','big5_neuroticism']

f1=open("../averages.txt", "a")
for typer in big5:
    for profile in profiles:
        if profile["botornot"]=="bot":
            botE += float(profile["big5_neuroticism"])
            i+=1
        if profile["botornot"]=="human":
            humanE+=float(profile["big5_neuroticism"])
            j+=1
    f1.write("Average bot "+typer+" = "+str(botE/i)+"\n")
    f1.write("Average human "+typer+" = "+str(humanE/j)+"\n")
