import snscrape.modules.twitter as sntwitter
import pandas as pd
import numpy as np
from datetime import datetime
import re
import string
import pickle
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

model = pickle.load(open('./old/modelSVC.pickle','rb'))
tfidf_vectorizer =  pickle.load(open('./old/tfidf_vectorizer.pickle','rb'))
APIurl = 'http://127.0.0.1:3000'

def cleaningText(text):
    text = re.sub(r'@[A-Za-z0-9]+', '', text) # remove mentions
    text = re.sub(r'#[A-Za-z0-9]+', '', text) # remove hashtag
    text = re.sub(r'RT[\s]', '', text) # remove RT
    text = re.sub(r"http\S+", '', text) # remove link
    text = re.sub(r'[0-9]+', '', text) # remove numbers
    text = text.encode("ascii", "ignore").decode() #remove emojis

    text = text.replace('\n', ' ') # replace new line into space
    text = text.translate(str.maketrans('', '', string.punctuation)) # remove all punctuations
    text = text.strip(' ') # remove characters space from both left and right text
    return text

def casefoldingText(text): # Converting all the characters in a text into lower case
    text = text.lower() 
    return text

def tokenizingText(text): # Tokenizing or splitting a string, text into a list of tokens
    text = word_tokenize(text) 
    return text

def filteringText(text): # Remove stopwors in a text
    listStopwords = set(stopwords.words('indonesian'))
    text = [w for w in text if not w in listStopwords]
    return text

def stemmingText(text): # Reducing a word to its word stem that affixes to suffixes and prefixes or to the roots of words
    #factory = StemmerFactory()
    from nltk.stem import PorterStemmer
    stemmer = PorterStemmer()
    text = [stemmer.stem(word) for word in text]
    return text

def toSentence(list_words): # Convert list of words into sentence
    sentence = ' '.join(word for word in list_words)
    return sentence

alay_dict = pd.read_csv('colloquial-indonesian-lexicon.csv', encoding='latin-1', header=None)
alay_dict = alay_dict.rename(columns={0: 'original', 1: 'replacement'})
alay_dict_map = dict(zip(alay_dict['original'], alay_dict['replacement']))

def normalize_alay(text):
 text = ' '.join([alay_dict_map[word] if word in alay_dict_map else word for word in text.split(' ')])
 text = re.sub(' +', ' ', text)
 return text

    ########################
    ###   DATA SCRAPING  ###
    ########################

def scraperKey(keyword,date_since,date_until):
    query = keyword+" lang:id until:"+str(date_until)+" since:"+str(date_since)
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    req.post('{}/progress/new'.format(APIurl),{'dateGet':now,'keyword':keyword,'dateSince':date_since,'dateUntil':date_until,'status':1,'source':'tw'} )
    #print(query)
    #print(datetime.now())
    #print("Sedang Mengumpulkan Data Twitter...")
    tweets = []
    while len(tweets) <= 20:
        for tweet in sntwitter.TwitterSearchScraper(query).get_items():
            tweets.append([datetime.now().date(), keyword, tweet.date, tweet.user.username, tweet.content, tweet.hashtags,tweet.mentionedUsers, tweet.likeCount, tweet.retweetCount, tweet.replyCount])
        
        
        
    df = pd.DataFrame(tweets, columns=['timestamp','keyword','date', 'user', 'tweet', 'hashtags', 'mentions', 'likeCount', 'retweetCount', 'replyCount'])

    ##############################
    ###   DATA PRE-PROCESSING  ###
    ##############################
