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
from googleapiclient.discovery import build
import json
from datetime import timezone



#Model dan Vectorizer
model = pickle.load(open('./tfidf_svc_stat3.pickle','rb'))
tfidf_vectorizer =  pickle.load(open('./tfidf_vectorizer_stat3.pickle','rb'))
# model = pickle.load(open('./tfidf_statistical_svc.pickle','rb'))
# tfidf_vectorizer =  pickle.load(open('./tfidf_vectorizer.pickle','rb'))
features = 5200

#URL untuk API Backend
APIurl = 'http://127.0.0.1:3000'
# APIurl = 'http://be-sa.qoisoctava.com'


# api key youtube
api_key = 'AIzaSyAOALHxbc03AvhCtaU0vYBaW6hAj8XTWYc'

def video_comments(video_id):
    # empty list for storing reply
	replies = []

	# creating youtube resource object
	youtube = build('youtube', 'v3', developerKey=api_key)
	
	video = youtube.videos().list(part='snippet', id=video_id).execute()

	title = ' '.join([item['snippet']['title'] for item in video['items']])
	videoDate = ' '.join([item['snippet']['publishedAt'] for item in video['items']])
	channel = ' '.join([item['snippet']['channelTitle'] for item in video['items']])
    
	# retrieve youtube video results
	video_response = youtube.commentThreads().list(part='snippet,replies', videoId=video_id).execute()

    
	# iterate video response
	while video_response:
		
		# extracting required info
		# from each result object
		for item in video_response['items']:
			
			# Extracting comments ()
			published = item['snippet']['topLevelComment']['snippet']['publishedAt']
			user = item['snippet']['topLevelComment']['snippet']['authorDisplayName']

			# Extracting comments
			comment = item['snippet']['topLevelComment']['snippet']['textDisplay']
			likeCount = item['snippet']['topLevelComment']['snippet']['likeCount']

			replies.append([videoDate,channel,title,published, user, comment, likeCount])
			
			# counting number of reply of comment
			replycount = item['snippet']['totalReplyCount']

			# if reply is there
			if replycount>0:
				# iterate through all reply
				for reply in item['replies']['comments']:
					
					# Extract reply
					published = reply['snippet']['publishedAt']
					user = reply['snippet']['authorDisplayName']
					repl = reply['snippet']['textDisplay']
					likeCount = reply['snippet']['likeCount']
					
					# Store reply is list
					#replies.append(reply)
					replies.append([videoDate,channel,title,published, user, repl, likeCount])

			# print comment with list of reply
			#print(comment, replies, end = '\n\n')

			# empty reply list
			#replies = []

		# Again repeat
		if 'nextPageToken' in video_response:
			video_response = youtube.commentThreads().list(
					part = 'snippet,replies',
					pageToken = video_response['nextPageToken'], 
					videoId = video_id
				).execute()
		else:
			break
	#endwhile
	return replies

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

# def filteringText(text): # Remove stopwors in a text
#     listStopwords = set(stopwords.words('indonesian'))
#     text = [w for w in text if not w in listStopwords]
#     return text

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

def dateformater(date):
    d = datetime.fromisoformat(date[:-1]).astimezone(timezone.utc)
    return d.strftime('%Y-%m-%d %H:%M:%S')

def update(id,code):
    update = requests.put('{}/progress/youtube/update'.format(APIurl), data={'id':id,'status': code})
    # print(update)
    ########################
    ###   DATA SCRAPING  ###
    ########################


def youtube(video_id,topic):
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    newProgress = {'get_date':now, 'channel_name':'loading...', 'title':'loading...', 'video_date':datetime.now(),  'status':1}
    #### First Requests
    # up =  requests.post('{}/progress/youtube/new'.format(APIurl), data=newProgress, headers=headers, allow_redirects=False)
    up =  requests.Session().post('{}/progress/youtube/new'.format(APIurl), newProgress,  timeout=30)
    # up =  httpx.post('{}/progress/youtube/new'.format(APIurl), data=newProgress, headers=headers)

    print(newProgress)
    print("-----")
    print(up)
    print(up.url)
    print(up.history)
    print(up.text)
    print("-----")
    #print(datetime.now())
    print("Sedang Mengumpulkan Data youtube...")
    comments = video_comments(video_id)
    df = pd.DataFrame(comments, columns=['video_date','channel_name','title','comment_date', 'commentator', 'content', 'like_count'])
    
       
    print("data sudah terkumpul")
    ##############################
    ###   DATA PRE-PROCESSING  ###
    ##############################

    id = json.loads(requests.post('{}/progress/youtube/id'.format(APIurl), {"get_date":now}).content.decode('utf-8'))[0]["id"]

    requests.put('{}/progress/youtube/detail'.format(APIurl), data={'id':id,'channel_name': df['channel_name'][0],'title': df['title'][0],'video_date': dateformater(df['video_date'][0])})   
    
    print("=====/ update /=====")
    print(id)


    if df.empty:
        update(id,404)
        print("Tidak ada data atau query keliru!")
    else:
        df = df.drop_duplicates(subset=['content'])
        dataset = df
        update(id,2)
        #print(dataset)
        print("Sedang Membersihkan Data youtube...")
        #gabungin hashtags
        # dataset['Hashtags'] = '-'.join(dataset['Hashtags'])
        print(datetime.now())
        print('cleaning text')
        dataset['text_clean'] = dataset['content'].apply(cleaningText)
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
        print("Selesai Membersihkan Data youtube :)")
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
            if i[0] >= 0.30 and i[1] >= 0.30 or i[0] >= 0.30 and i[2] >= 0.30:
                max_prob = 4
            # elif i[2] <= 0.80 and i[0] <= 0.80:
            #     max_prob = 1
            else:
                max_prob = ilist.index(max(ilist))
            
            probs.append(max_prob)

        dataset['sentiment'] = probs
        # dataset['sentiment'] = y_pred
        # print(dataset['sentiment'].value_counts())
        dataset = dataset.drop(dataset[dataset['sentiment'] == 4].index)

        polarity_decode = {0 : 'Negative', 1 : 'Neutral', 2 : 'Positive'}
        dataset['sentiment'] = dataset['sentiment'].map(polarity_decode)
        # print(dataset['sentiment'].value_counts())
        print("Selesai Menganalisis Sentimen Publik :)")
        dataset['get_date'] = now
        print(datetime.now())

            
        df = dataset[['get_date', 'video_date', 'title', 'channel_name', 'comment_date', 'commentator', 'content', 'like_count',  'sentiment']]
        
    data_dict = df.to_dict(orient='records')
    for i in data_dict:
        dicts ={
            'get_date':i['get_date'],
            'video_date':dateformater(i['video_date']),
            'title':i['title'],
            'channel_name':i['channel_name'],
            'comment_date':dateformater(i['comment_date']),
            'commentator':i['commentator'],
            'content':i['content'],
            'like_count':i['like_count'],
            'sentiment':i['sentiment'],
            'topic':topic,
            'batch_id':id,
            }
        # print(dicts)
        c = requests.post('{}/data/youtube'.format(APIurl),dicts )
        print(c)

    update(id,4)              
