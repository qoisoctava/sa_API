from flask import Flask, request
import json
from flask_cors import CORS
from twitter_predictor import twitter
from youtube_predictor import youtube

app = Flask(__name__)
CORS(app)

@app.route('/twitter/prediction/', methods=["POST"])
def twitter_prediction_result():
    json.dumps(200)
    data = request.get_json(force=True)
    print(data)
    keyword = data.get('keyword')
    dateSince = data.get('dateSince')
    dateUntil = data.get('dateUntil')
    topic = data.get('topic')
    twitter(keyword,dateSince,dateUntil,topic)
    return json.dumps(200)

@app.route('/youtube/prediction/', methods=["POST"])
def youtube_prediction_result():
    json.dumps(200)
    data = request.get_json(force=True)
    print(data)
    id = data.get('id')
    topic = data.get('topic')
    youtube(id,topic)
    
    return json.dumps("diproses")



if __name__ == "__main__":
    #app.run(host='127.0.0.1', port='5000')
    app.run()
    #serve(app, host='127.0.0.1', port=5000)       