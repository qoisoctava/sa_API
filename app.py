import snscrape.modules.twitter as sntwitter
import pandas as pd
import numpy as np
from datetime import datetime
import re
import string
import pickle
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import requests
from flask import Flask, redirect, url_for, request, render_template
import urllib.request as request_api
import json
from flask_cors import CORS, cross_origin

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

def update(id,code):
    requests.put('{}/progress/twitter/update'.format(APIurl), data={'id':id,'status': code})
   
    ########################
    ###   DATA SCRAPING  ###
    ########################

def twitter(keyword,date_since,date_until,topic):
    query = keyword+" lang:id until:"+str(date_until)+" since:"+str(date_since)
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    up =  requests.post('{}/progress/twitter/new'.format(APIurl),{'dateGet':now,'keyword':keyword,'dateSince':date_since,'dateUntil':date_until,'status':1,'source':'tw'} )
    print(up)
    #print(datetime.now())
    print("Sedang Mengumpulkan Data Twitter...")
    tweets = []
    while len(tweets) <= 20:
        for tweet in sntwitter.TwitterSearchScraper(query).get_items():
            tweets.append([datetime.now().date(), keyword, tweet.date, tweet.user.username, tweet.content, tweet.hashtags,tweet.mentionedUsers, tweet.likeCount, tweet.retweetCount, tweet.replyCount])
        
        
       
    print("data sudah terkumpul")
    df = pd.DataFrame(tweets, columns=['timestamp','keyword','date', 'user', 'tweet', 'hashtags', 'mentions', 'likeCount', 'retweetCount', 'replyCount'])
    ##############################
    ###   DATA PRE-PROCESSING  ###
    ##############################

    id = requests.get('{}/progress/twitter/id/{}'.format(APIurl,now)).json()[0]['id']
    print("=====/ update /=====")
    print(id)


    if df.empty:
        update(id,404)
        print("Tidak ada data atau query keliru!")
    else:
        df = df.drop_duplicates(subset=['tweet'])
        dataset = df
        update(id,2)
        #print(dataset)
        print("Sedang Membersihkan Data Twitter...")
        #gabungin hashtags
        # dataset['Hashtags'] = '-'.join(dataset['Hashtags'])
        print(datetime.now())
        print('cleaning text')
        dataset['text_clean'] = dataset['tweet'].apply(cleaningText)
        print(datetime.now())
        print('casefolding text')
        dataset['text_clean'] = dataset['text_clean'].apply(casefoldingText)
        dataset['text_clean'] = dataset['text_clean'].apply(normalize_alay)
        
        print(datetime.now())
        print('tokenizing text')
        dataset['text_preprocessed'] = dataset['text_clean'].apply(tokenizingText)
        print(datetime.now())
        print("filtering text")
        dataset['text_preprocessed'] = dataset['text_preprocessed'].apply(filteringText)
        print(datetime.now())
        print('stemming text')
        dataset['text_preprocessed'] = dataset['text_preprocessed'].apply(stemmingText)
        dataset["popularityScore"] = (dataset["likeCount"] + dataset["retweetCount"] + dataset["replyCount"])/3
        print("Selesai Membersihkan Data Twitter :)")
        print(datetime.now())

            ###################################
            ###      SENTIMENT ANALYSIS     ###
            ###################################

        print("Sedang Menganalisis Sentimen Publik...")
        update(id,3)

        print(datetime.now())
        # Make text preprocessed (tokenized) to untokenized with toSentence Function
        X = dataset['text_preprocessed'].apply(toSentence)
        X = tfidf_vectorizer.transform(X.values)

        y_pred = model.predict(X)
        dataset['sentiment'] = y_pred

        polarity_decode = {0 : 'Negative', 1 : 'Neutral', 2 : 'Positive'}
        dataset['sentiment'] = dataset['sentiment'].map(polarity_decode)
        print("Selesai Menganalisis Sentimen Publik :)")

        print(datetime.now())

        dataset['date'] = pd.to_datetime(dataset['date']).dt.date
        dataset['date']=dataset['date'].astype(str)
            
        df = dataset[['timestamp', 'keyword', 'date', 'user', 'tweet', 'hashtags', 'mentions', 'likeCount', 'retweetCount', 'replyCount','popularityScore',  'sentiment']]
        
    data_dict = df.to_dict(orient='records')
    for i in data_dict:
        dicts ={
            'dateGet':i['timestamp'],
            'keyword':i['keyword'],
            'contentDate':i['date'],
            'username':i['user'],
            'tweet':i['tweet'],
            'hashtags':i['hashtags'],
            'mentions':i['mentions'],
            'likeCount':i['likeCount'],
            'retweetCount':i['retweetCount'],
            'replyCount':i['replyCount'],
            'popularityScore':i['popularityScore'],
            'sentiment':i['sentiment'],
            'topic':topic
            }
        # print(dicts)
        c = requests.post('{}/data/twitter'.format(APIurl),dicts )
        print(c)

    df_hashtags = df[['keyword','date','hashtags','sentiment']]
    df_hashtags = df_hashtags[df_hashtags['hashtags'].notna()].reset_index()
    df_mentions = df[['keyword','date','mentions','sentiment']]
    df_mentions = df_mentions[df_mentions['mentions'].notna()].reset_index()

    for i in range(df_hashtags.shape[0]):
        for j in range(len(df_hashtags['hashtags'][i])):
            contentDate = df_hashtags['date'][i]
            keyword = df_hashtags['keyword'][i]
            hashtag = df_hashtags['hashtags'][i][j]
            sentiment = df_hashtags['sentiment'][i]
            if hashtag != 0:
                requests.post('{}/data/twitter/content'.format(APIurl),{
                    'contentDate' : contentDate,
                    'topic' : topic,
                    'keyword' : keyword,
                    'content' : hashtag,
                    'type' : 'hashtag',
                    'sentiment' : sentiment
                } )
        
    for i in range(df_mentions.shape[0]):
        for j in range(len(df_mentions['mentions'][i])):
            contentDate = df_mentions['date'][i]
            keyword = df_mentions['keyword'][i]
            mention = str(df_mentions['mentions'][i][j]).replace('https://twitter.com/', '@')
            sentiment = df_mentions['sentiment'][i]
            if mention != 0:
                requests.post('{}/data/twitter/content'.format(APIurl),{
                    'contentDate' : contentDate,
                    'topic' : topic,
                    'keyword' : keyword,
                    'content' : mention,
                    'type' : 'mention',
                    'sentiment' : sentiment
                } )
                print(mention)
    update(id,4)              

app = Flask(__name__)
CORS(app)
#run_with_ngrok(app)
@app.route('/twitter/prediction/', methods=["POST","GET"])
def prediction_result():
    json.dumps(200)
    data = request.get_json(force=True)
    print(data)
    keyword = data.get('keyword')
    dateSince = data.get('dateSince')
    dateUntil = data.get('dateUntil')
    topic = data.get('topic')
    twitter(keyword,dateSince,dateUntil,topic)
    
    return json.dumps("diproses")



if __name__ == "__main__":
    #app.run(host='127.0.0.1', port='5000')
    app.run()
    #serve(app, host='127.0.0.1', port=5000)       