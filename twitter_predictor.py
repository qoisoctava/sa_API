import snscrape.modules.twitter as sntwitter
import pandas as pd
import numpy as np
from scipy import sparse
from datetime import datetime
import re
import string
import pickle
from nltk.tokenize import word_tokenize
import requests
from indoNLP.preprocessing import replace_word_elongation, emoji_to_words





#Model dan Vectorizer
model = pickle.load(open('./tfidf_svc_stat3.pickle','rb'))
tfidf_vectorizer =  pickle.load(open('./tfidf_vectorizer_stat3.pickle','rb'))
features = 5200

#URL untuk API
APIurl = 'http://127.0.0.1:3000'
# APIurl = 'http://be-sa.qoisoctava.com'

def cleaningText(text):
    text = re.sub(r'@[A-Za-z0-9]+', '', text) # remove mentions
    text = re.sub(r'#[A-Za-z0-9]+', '', text) # remove hashtag
    text = re.sub(r'RT[\s]', '', text) # remove RT
    text = re.sub(r"http\S+", '', text) # remove link
    text = re.sub(r'[0-9]+', '', text) # remove numbers
    # text = text.encode("ascii", "ignore").decode() #remove emojis

    text = text.replace('•', '') # replace new line into space
    text = text.replace('\n', ' ') # replace new line into space
    # text = [pipe(word) for word in text]
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

def replaceElongation(text): # Reducing a word to its word stem that affixes to suffixes and prefixes or to the roots of words
    text = [replace_word_elongation(word) for word in text]
    
    return text

def replaceEmojis(text): # Reducing a word to its word stem that affixes to suffixes and prefixes or to the roots of words
    text = ' '.join([emoji_to_words(word, lang='en').replace('!',' ').replace('_','') for word in text.split(' ')])
    text = re.sub(' +', ' ', text)
    text = text.encode("ascii", "ignore").decode()
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
    update = requests.put('{}/progress/twitter/update'.format(APIurl), data={'id':id,'status': code})
    # print(update)
    ########################
    ###   DATA SCRAPING  ###
    ########################


def twitter(keyword,date_since,date_until,topic):
    query = keyword+" lang:id until:"+str(date_until)+" since:"+str(date_since)
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    newProgress = {'dateGet':now,'keyword':keyword,'dateSince':date_since,'dateUntil':date_until,'status':1}
    print(newProgress)
    up =  requests.post('{}/progress/twitter/new'.format(APIurl), newProgress )
    print(up)
    #print(datetime.now())
    print("Sedang Mengumpulkan Data Twitter...")
    tweets = []
    while len(tweets) <= 20:
        for tweet in sntwitter.TwitterSearchScraper(query).get_items():
            tweets.append([datetime.now().date(), keyword, tweet.date, tweet.user.username, tweet.content, tweet.likeCount, tweet.retweetCount, tweet.replyCount])
        
        
       
    print("data sudah terkumpul")
    df = pd.DataFrame(tweets, columns=['timestamp','keyword','date', 'user', 'tweet', 'likeCount', 'retweetCount', 'replyCount'])
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
        dataset['text_clean'] = dataset['text_clean'].apply(replaceEmojis)
        dataset['text_clean'] = dataset['text_clean'].apply(casefoldingText)
        dataset['text_clean'] = dataset['text_clean'].apply(normalize_alay)
        
        print(datetime.now())
        print('tokenizing text')
        dataset['text_preprocessed'] = dataset['text_clean'].apply(tokenizingText)
        print(datetime.now())
        print("filtering text")
        dataset['text_preprocessed'] = dataset['text_preprocessed'].apply(replaceElongation)
        print(datetime.now())
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
        X_input_tfidf = tfidf_vectorizer.transform(X.values)

        e = X_input_tfidf.toarray()

        # #Mean
        # mean = []
        # for i in e:
        #     a = sum(i.tolist())/features
        #     mean.append(a)

        #Max
        maks = []
        for i in e:
            a = max(i.tolist())
            maks.append(a)

        #Min
        minn = []
        for i in e:
            ilist = i.tolist()
            nZ = [j for j in ilist if j != 0]
            if len(nZ) != 0:
                a = min(nZ)
            elif len(nZ) == 0:
                a = 0
            minn.append(a)

        #Sum
        summ = []
        for i in e:
            a = sum(i.tolist())
            summ.append(a)

        # #Std
        # stdev = []
        # for i in e:
        #     s = np.std(i.tolist())
        #     stdev.append(s)

        # #Count
        # count = []
        # for i in e:
        #     a = sum(map(lambda x : x != 0, i.tolist()))
        #     count.append(a)

        # #Mode
        # from statistics import mode
        # modes = []
        # for i in e:
        #     ilist = i.tolist()
        #     nZ = [j for j in ilist if j != 0]
        #     if len(nZ) != 0:
        #         a = mode(nZ)
        #     elif len(nZ) == 0:
        #         a = 0
        #     modes.append(a)

        # #Variance
        # vrc = []
        # for i in e:
        #     s = np.std(i.tolist())
        #     mean2 = sum(i.tolist()) / features
        #     res = sum((i - mean2) ** 2 for i in i.tolist()) / features
        #     vrc.append(res)

        # #Median
        # from statistics import median
        # meds = []
        # for i in e:
        #     ilist = i.tolist()
        #     nZ = [j for j in ilist if j != 0]
        #     if len(nZ) != 0:
        #         a = median(nZ)
        #     elif len(nZ) == 0:
        #         a = 0    
        #     meds.append(a)

        # extra_X = np.column_stack((e, np.array(mean),np.array(stdev), np.array(maks), np.array(minn), np.array(summ),np.array(meds)))
        extra_X = np.column_stack((e, np.array(maks), np.array(minn), np.array(summ)))

        X = sparse.csr_matrix(extra_X)

        y_pred = model.predict(X)
        proba = model.predict_proba(X)
        
        probs = []
        for i in proba:
            ilist = i.tolist()
            max_prob = 1
            if i[2] <= 0.0 and i[0] <= 0.40:
                max_prob = 4
            # elif i[2] <= 0.70 and i[0] <= 0.70:
            #     max_prob = 1
            else:
                max_prob = ilist.index(max(ilist))
            
            probs.append(max_prob)

        # dataset['sentiment'] = probs
        dataset['sentiment'] = y_pred
        # print(dataset['sentiment'].value_counts())
        dataset = dataset.drop(dataset[dataset['sentiment'] == 4].index)

        polarity_decode = {0 : 'Negative', 1 : 'Neutral', 2 : 'Positive'}
        dataset['sentiment'] = dataset['sentiment'].map(polarity_decode)
        # print(dataset['sentiment'].value_counts())
        print("Selesai Menganalisis Sentimen Publik :)")

        print(datetime.now())

        dataset['date'] = pd.to_datetime(dataset['date']).dt.date
        dataset['date']=dataset['date'].astype(str)
            
        df = dataset[['timestamp', 'keyword', 'date', 'user', 'tweet', 'likeCount', 'retweetCount', 'replyCount','popularityScore',  'sentiment']]
        
    data_dict = df.to_dict(orient='records')
    for i in data_dict:
        dicts ={
            'dateGet':i['timestamp'],
            'keyword':i['keyword'],
            'contentDate':i['date'],
            'username':i['user'],
            'tweet':i['tweet'],
            'likeCount':i['likeCount'],
            'retweetCount':i['retweetCount'],
            'replyCount':i['replyCount'],
            'popularityScore':i['popularityScore'],
            'sentiment':i['sentiment'],
            'topic':topic,
            'batch_id':id,
            }
        # print(dicts)
        requests.post('{}/data/twitter'.format(APIurl),dicts )
        

    update(id,4)              
