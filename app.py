import snscrape.modules.twitter as sntwitter
import pandas as pd


def scraperKey(keyword,date_since,date_until):
    query = keyword+" lang:id until:"+str(date_until)+" since:"+str(date_since)
    #print(query)
    #print(datetime.now())
    #print("Sedang Mengumpulkan Data Twitter...")
    tweets = []
    for tweet in sntwitter.TwitterSearchScraper(query).get_items():
        tweets.append([tweet.date, tweet.user.username, tweet.content])
        
        

    df = pd.DataFrame(tweets, columns=['Date', 'User', 'Tweet'])