##############################
###     S C R A P E  R     ###
##############################

import snscrape.modules.twitter as sntwitter
import pandas as pd
from datetime import datetime
import datetime as dt
import re
import string
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

# from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
import pickle


from flask import Flask, redirect, url_for, request, render_template
import numpy as np
import urllib.request as request_api
import json

# from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
model = pickle.load(open('modelSVC.pickle','rb'))
tfidf_vectorizer =  pickle.load(open('tfidf_vectorizer.pickle','rb'))
#Some functions for preprocessing text

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

# hashtags
def find_tags(row_string):
    # use a list comprehension to find list items that start with #
    tags = [x for x in row_string if x.startswith('#')]
    return tags

# link
def find_link(row_string):
    # use a list comprehension to find list items that start with http
    link = [x for x in row_string if x.startswith('http')]
    return link

# mention
def find_mention(row_string):
    # use a list comprehension to find list items that start with @
    mention = [x for x in row_string if x.startswith('@')]
    return mention

alay_dict = pd.read_csv('colloquial-indonesian-lexicon.csv', encoding='latin-1', header=None)
alay_dict = alay_dict.rename(columns={0: 'original', 1: 'replacement'})
alay_dict_map = dict(zip(alay_dict['original'], alay_dict['replacement']))

def normalize_alay(text):
 text = ' '.join([alay_dict_map[word] if word in alay_dict_map else word for word in text.split(' ')])
 text = re.sub(' +', ' ', text)
 return text

def scraperKey(keyword,date_since,date_until):
    query = keyword+" lang:id until:"+str(date_until)+" since:"+str(date_since)
    #print(query)
    #print(datetime.now())
    #print("Sedang Mengumpulkan Data Twitter...")
    tweets = []
    for tweet in sntwitter.TwitterSearchScraper(query).get_items():
        tweets.append([tweet.date, tweet.user.username, tweet.content])
        
        

    df = pd.DataFrame(tweets, columns=['Date', 'User', 'Tweet'])
    #print("Selesai Mengumpulkan Data :)")
    #print(datetime.now())
    #print(df.shape)

    ##############################
    ###   DATA PRE-PROCESSING  ###
    ##############################



#Main Function

    if df.empty:
        return("Tidak ada data atau query keliru!")
    else:
        df = df.drop_duplicates(subset=['Tweet'])
        dataset = df
        #print(dataset)
        #print("Sedang Membersihkan Data Twitter...")
        #Extract Hashtags, Mentinoned Usernames,  and Link
        dataset['split'] = dataset['Tweet'].str.split(' ')
        dataset['Hashtags'] = dataset['split'].apply(lambda row : find_tags(row))
        dataset['Link'] = dataset['split'].apply(lambda row : find_link(row))
        dataset['Mention'] = dataset['split'].apply(lambda row : find_mention(row))
        # replace # as requested in OP, replace for new lines and \ as needed.
        dataset['Hashtags'] = dataset['Hashtags'].apply(lambda x : str(x).replace('#', '').replace('\\n', ',').replace('\\', '').replace("'", ""))
        dataset['Mention'] = dataset['Mention'].apply(lambda x : str(x).replace('\\n', ',').replace('\\', '').replace("'", ""))
        del dataset['split']
        #print(datetime.now())
        #print('cleaning text')
        dataset['text_clean'] = dataset['Tweet'].apply(cleaningText)
        #print(datetime.now())
        #print('casefolding text')
        dataset['text_clean'] = dataset['text_clean'].apply(casefoldingText)
        dataset['text_clean'] = dataset['text_clean'].apply(normalize_alay)
        dataset.drop(['Tweet'], axis = 1, inplace = True)
        #print(datetime.now())
        #print('tokenizing text')
        dataset['text_preprocessed'] = dataset['text_clean'].apply(tokenizingText)
        #print(datetime.now())
        #print("filtering text")
        dataset['text_preprocessed'] = dataset['text_preprocessed'].apply(filteringText)
        #print(datetime.now())
        #print('stemming text')
        dataset['text_preprocessed'] = dataset['text_preprocessed'].apply(stemmingText)
        #print("Selesai Membersihkan Data Twitter :)")
        #print(datetime.now())
        #print(dataset)
        cleanData = dataset

        ###################################
        ###      SENTIMENT ANALYSIS     ###
        ###################################

        #print("Sedang Menganalisis Sentimen Publik...")
        print(datetime.now())
        # Make text preprocessed (tokenized) to untokenized with toSentence Function
        X = cleanData['text_preprocessed'].apply(toSentence)
        X = tfidf_vectorizer.transform(X.values)

        y_pred = model.predict(X)
        cleanData['Result Prediction'] = y_pred

        polarity_decode = {0 : 'Negative', 1 : 'Neutral', 2 : 'Positive'}
        cleanData['Result Prediction'] = cleanData['Result Prediction'].map(polarity_decode)
        #print("Selesai Menganalisis Sentimen Publik :)")
        print(datetime.now())
        #print(cleanData[['text_clean','Result Prediction']])
        #print('OK')
        #print(cleanData)
        cleanData['Date'] = pd.to_datetime(cleanData['Date']).dt.date
        cleanData['Date']=cleanData['Date'].astype(str)
        
        #print(cleanData.dtypes)
        data_dict = cleanData.to_dict(orient='records')
        return(data_dict)




#from waitress import serve
#from flask_ngrok import run_with_ngrok

app = Flask(__name__)
#run_with_ngrok(app)
@app.route('/sa/prediction/', methods=["POST","GET"])
def prediction_result():
    #receive parameter sent by client
    jsonData = request.get_json(force=True)
    #print('OKJ',jsonData)
    data_arr = np.array(jsonData)
    #print(data_arr)
    #load the model
    keyword = data_arr[0]
    date_since = data_arr[1]
    date_until = data_arr[2]
    data = scraperKey(keyword,date_since,date_until)
    #json_out = data.dumps()
    #print(data)
    #json_data = jsonable_encoder(data)
    return json.dumps(data)



if __name__ == "__main__":
    #app.run(host='127.0.0.1', port='5000')
    app.run()
    #serve(app, host='127.0.0.1', port=5000)
    
