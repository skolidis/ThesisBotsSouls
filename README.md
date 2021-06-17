# ThesisBotsSouls

The code in this repository is an assortment of scripts, used to complete my thesis through creating graphs or processing information through classification and machine learning. It is all divided in 3 sections:
1. Graphs
In graphs there is code for creating graphs. These include CDFs, Histograms, TSNEs and Averages for two different kinds of info, bot vs human and type of bots. This was done because the two kinds of info required a different approach to how they would be presented.
2. Machine Learning 
This includes the method used for classification, as well as parameter optimization through Gridsearch. It works universally based on different pymongo queries.
3. Misc
This includes an assortment of scripts such as the feature importance calculator, which calculates importance based on a descending number of features, a script that retrieved the names of different kinds of bots from a folder of datasets, the script used to save tweets through the twitter API and lastly, the method to set the type of bot in the mongo database after feature extraction
