import tweepy
from pymongo import MongoClient
import time

consumer_key="MF7VXR5ri7E7WoH14o2DRXGJZ"
consumer_secret="VKcgEpynmKhAWBrAOV3fHtFdducVcgBf7jiu3iiXhR9CwRjmTH"
access_token= "50138668-Z69bXdP0GYTMiTXSxrz2JM4gcl1sg30MTg4xEUKbF"
access_token_secret="qYsfZY2tXLeVDHVBBS7faU9XBaDyWYUH1EPLZh5rXnW2N"


auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)
api = tweepy.API(auth, wait_on_rate_limit=True)

client = MongoClient('mongodb://localhost:27017/')
db = client.tweets

i=0
f = open("Complete_dataset_users.txt", "r")
Lines = f.readlines()

for line in Lines:
    print(i)
    i+=1
    user = line.split()
    try:
        j = 0
        posts = []
        for status in tweepy.Cursor(api.user_timeline, user_id=user[0], tweet_mode="extended").items(1000):
            post = status._json
            post["botornot"] = user[1]
            db.tweetsnew.insert_one(post)
            j += 1
            if j == 1000:
                break
    except tweepy.RateLimitError:
        print("ha")
        time.sleep(15 * 60)
    except Exception as e:
        print(e)