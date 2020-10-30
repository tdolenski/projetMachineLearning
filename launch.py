import os
import pickle
from flask_bootstrap import Bootstrap
from flask import Flask, render_template, request
import json
import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfTransformer, CountVectorizer, TfidfVectorizer
from unidecode import unidecode
from nltk.stem import SnowballStemmer
from nltk.corpus import stopwords
from stop_words import get_stop_words
import nltk
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from werkzeug.utils import secure_filename
import sys

print('Coucou')
my_stop_word_list = get_stop_words('french')
final_stopwords_list = stopwords.words('french')
s_w = list(set(final_stopwords_list + my_stop_word_list))
s_w = [elem.lower() for elem in s_w]
fr = SnowballStemmer('french')
nltk.download('stopwords')


def trainModel(filename=None):
    if filename == None :
        df = pd.read_csv('corpus.csv')
    else:
        df = pd.read_csv('data/{}'.format(filename));
    len(df), len(df.drop_duplicates('review'))
    df['l_review'] = df['review'].apply(lambda x: len(x.split(' ')))
    if len(df['rating'].unique()) > 2:
        # good_reviews = len(df[(df['rating'] > 3) & (df['l_review'] > 5)])
        # bad_reviews = len(df[(df['rating'] < 3) & (df['l_review'] > 5)])
        df = df[df['l_review'] > 5]
        df['label'] = df['rating']
        # Je conserve autant de comments par label que possible (contraint equilibrer les classes 0 et 1)
        positif = df[df['label'] > 3].sample(391)
        # positif.index=list(range(0,len(positif)))
        negatif = df[df['label'] < 3]
        # negatif.index=list(range(len(positif),len(negatif)+len(positif)))
        Corpus = pd.concat([positif, negatif], ignore_index=True)[['review', 'label']]
        ### Attention c'est sale, si vous avez un plus gros corpus travaillez avec df.loc[liste_index,'nom_colonne']
        for ind in Corpus['label'].index:
            if Corpus.loc[ind, 'label'] > 3:
                Corpus.loc[ind, 'label'] = 1
            elif Corpus.loc[ind, 'label'] < 3:
                Corpus.loc[ind, 'label'] = 0
    else:
        # good_reviews = len(df[(df['rating'] == 0) & (df['l_review'] > 5)])
        # bad_reviews = len(df[(df['rating'] == 1) & (df['l_review'] > 5)])
        positif = df[df['rating'] == 0]
        negatif = df[df['rating'] == 1]
        df['label'] = df['rating']
        Corpus = pd.concat([positif, negatif], ignore_index=True)[['review', 'label']]
        ### Attention c'est sale, si vous avez un plus gros corpus travaillez avec df.loc[liste_index,'nom_colonne']
        for ind in Corpus['label'].index:
            if Corpus.loc[ind, 'label'] == 0:
                Corpus.loc[ind, 'label'] = 1
            elif Corpus.loc[ind, 'label'] == 1:
                Corpus.loc[ind, 'label'] = 0

    Corpus['review_net'] = Corpus['review'].apply(nettoyage)
    # Load
    vectorizer = TfidfVectorizer()
    vectorizer.fit(Corpus['review_net'])
    X = vectorizer.transform(Corpus['review_net'])

    # Save vectorizer.vocabulary_
    pickle.dump(vectorizer.vocabulary_, open("feature.pkl", "wb"))
    words_length = (len(vectorizer.get_feature_names()))
    y = Corpus['label']
    X.shape
    x_train, x_val, y_train, y_val = train_test_split(X, y, test_size=0.2)
    cls = LogisticRegression(max_iter=300).fit(x_train, y_train)

    accuracy = cls.score(x_val, y_val)

    # Save classifier
    pickle.dump(cls, open("cls.pkl", "wb"))

    return 'La précision est de : {0}% pour {1} mots'.format(int(accuracy * 100), words_length)

def nettoyage(string):
    l = []
    string = unidecode(string.lower())
    # Sans ponctuation pour le moment
    string = " ".join(re.findall("[a-zA-Z]+", string))

    for word in string.split():
        if word in s_w:
            print('do nothing');
        else:
            l.append(fr.stem(word))
    return ' '.join(l)

def getTypeMsg(message):
    phrase = message

    transformer = TfidfTransformer()
    loaded_vec = CountVectorizer(decode_error="replace",vocabulary=pickle.load(open("feature.pkl", "rb")))
    user = transformer.fit_transform(loaded_vec.fit_transform([nettoyage(phrase)]))

    cls=pickle.load(open("cls.pkl", "rb"))

    cls.predict(user),cls.predict_proba(user).max(),

    print(cls.predict(user))
    if cls.predict(user) >= 1:
        return 'avis OK'
    else:
        return 'avis de MERDE'

app = Flask(__name__)


@app.route("/")
def home():
    return render_template("index.html", title='Home')

@app.route("/pred")
def predicition():
    return render_template("prediction.html", title='prediction')

@app.route("/prediction", methods=['POST'])
def prediction():
    user_text = request.form.get('input_text')
    return getTypeMsg(user_text)

@app.route('/train')
def train():
    return render_template("entrainement.html")

@app.route("/entrainement", methods=['POST'])
def entrainement():
    return render_template("result.html", message=trainModel())

@app.route('/upload')
def upload():
    return render_template("upload.html")

@app.route("/uptrain", methods=['POST'])
def upload_train():
    if request.method =='POST':
        file = request.files['file[]']
        if file:
            filename = secure_filename(file.filename)
            file.save(os.path.join("data", filename))
    return render_template("result.html", message=trainModel(filename))

if __name__ == "__main__":
    Bootstrap(app)
    app.run(host='0.0.0.0',port=sys.argv[1],use_reloader=False)
