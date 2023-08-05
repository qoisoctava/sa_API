import requests

APIurl = 'http://be-sa.qoisoctava.com'
  
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
    'sentiment':i['sentiment'],
    'batch_id':id
    }
    requests.post('{}/data/twitter'.format(APIurl),dicts )
                    
                    