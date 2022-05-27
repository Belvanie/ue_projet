#from asyncio.windows_events import NULL
from pickle import TRUE
from tkinter import CASCADE
from wsgiref import headers
from django.db import models
from django.core.validators import MaxValueValidator,MinValueValidator
import requests
import json
import datetime
import dateutil.parser
import unicodedata
from django.utils import timezone
import re
from textblob import TextBlob
from textblob_fr import PatternTagger, PatternAnalyzer
import pandas as pd

import numpy as np
from bs4 import BeautifulSoup #pour l'analyse syntaxique des documents html et xml
import matplotlib.pyplot as plt #Pour la representation graphique des donnees(visualisation)
import seaborn as sns #Pour la visualisation des donnees
import nltk #Pour le traitement automatique du langage
from nltk.corpus import stopwords #Pour l'elimina tion des mots inutiles dans notre jeu de donnees
from nltk.stem import SnowballStemmer #Pour la radicalisation des mots(Rennvoyer chaque mots en son radical)
from nltk.tokenize import TweetTokenizer #pour la division des mots en tokenn

from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer # CountVectorizer: Pour converrtir une collection de documents en une matrices
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC 
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer, accuracy_score, f1_score
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import confusion_matrix, roc_auc_score, recall_score, precision_score





# Create your models here.

class Query():

    search_url = "https://api.twitter.com/2/tweets/search/recent" #Change to the endpoint you want to collect data from
    header = {'Authorization': 'Bearer AAAAAAAAAAAAAAAAAAAAANG8ZAEAAAAAYvdUvNEgVvab1x6B3ECBc15joko%3D4MdUWUdk8yY7sDeEsznxRucNU22VorBM4SRrNIE8zmsR2JmRur'}
    period = datetime.timedelta(hours=3)
    

    def __init__(self, keywors, start_time, end_time, max_results):
        self.keywords = keywors
        self.start_time = start_time
        self.end_time = end_time
        self.max_results = max_results
        self.start_list = []
        self.end_list = []

    def gen_time_period(self):
        period = self.period
        start_time = dateutil.parser.isoparse(self.start_time)
        end_time = dateutil.parser.isoparse(self.end_time)
        while (end_time - start_time) > period:
            self.start_list.append(start_time.isoformat())
            start_time = start_time + period
            self.end_list.append(start_time.isoformat())
        self.start_list.append(start_time.isoformat())
        self.end_list.append(end_time.isoformat())


    def create_url(self):
        #change params based on the endpoint you are using
        query_params = {'query': self.keywords,
                        'start_time': self.start_time,
                        'end_time': self.end_time,
                        'max_results': self.max_results,
                        'expansions': 'author_id,geo.place_id',
                        'tweet.fields': 'id,text,author_id,geo,created_at,lang,public_metrics',
                        'user.fields': 'username,created_at,public_metrics',
                        'place.fields': 'full_name,id,country,country_code,geo,name,place_type',
                        'next_token': {}}
        return  query_params

    def connect_to_endpoint(self, next_token = None):
        params = self.create_url()
        params['next_token'] = next_token   #params object received from create_url function
        response = requests.request("GET", self.search_url, headers =self.header, params= params)
        print("Endpoint Response Code: " + str(response.status_code))
        #if response.status_code != 200:
        #    raise Exception(response.status_code, response.text)
        print(response)
        return response

    @staticmethod
    def get_help():
        return("pour constituer une requete vous devez ...")



class Data(models.Model):
    
    created_at = models.fields.DateTimeField (null=True)
    created_by = models.fields.CharField(max_length= 50)
    subject = models.fields.CharField(max_length= 200)
    cardinal = models.fields.IntegerField(validators=[MinValueValidator(0)], null= True)
    #tweets = models.fields.CharField(max_length= 100000, null= True)
    

    def generate(self, response, subject, user):
        counter = 0
        tweet_list = []
        print(response)
        for tw in response['data']:
            tweet_list.append(tw['id'])
            tweet = Tweet()
            tweet.tweet_id = tw['id']
            tweet.created_at = tw['created_at'] 
            tweet.author_id = tw['author_id'] 
            tweet.retweet_count = tw['public_metrics']['retweet_count'] 
            tweet.reply_count = tw['public_metrics']['reply_count'] 
            tweet.like_count = tw['public_metrics']['like_count']
            tweet.quote_count = tw['public_metrics']['quote_count'] 
            tweet.text = tw['text']
            tweet.polarity = tweet.model_pred()
            tweet.data = self

            tweet.save()
            counter += 1

        self.created_at = timezone.now()
        self.created_by = user
        self.subject = subject
        self.cardinal = counter
        #self.tweets = ""
        #for id in tweet_list:
        #    self.tweets = self.tweets + str(id) + " "
        
    #def __str__(self):
    #    return f"""collectées le {self.created_at.strftime("%A %d. %B %Y %I:%M%p")} par {self.created_by}""" 
    
    def gen_time_period(self):
        pass
    def create_url(self):
        pass
    def connect_to_endpoint(self, next_token=None):
        pass


def step_func(z):
        return 1 if (z>0) else 0
def findLink(x):
    return len(re.findall(r'\bhttp[a-z]*', x))
def findMail(x):
    return len(re.findall(r"\@", x))
def findHashtag(x):
    return len(re.findall(r"\#", x))
def find_exclamation(x):
    return len(re.findall(r"\!", x))
def find_interogation(x):
    return len(re.findall(r"\?", x))

class Tweet(models.Model):
    pcp=(9802.1, 405.61, 4878.55, 19600.85)
    data = models.ForeignKey(Data, on_delete=models.CASCADE, null= True)
    tweet_id  = models.fields.CharField(max_length=30, primary_key=True)
    created_at = models.fields.DateTimeField()
    author_id = models.fields.CharField(max_length= 20, blank= True)
    like_count = models.fields.IntegerField(validators=[MinValueValidator(0)], default=0)
    quote_count = models.fields.IntegerField(validators=[MinValueValidator(0)], default=0)
    reply_count = models.fields.IntegerField(validators=[MinValueValidator(0)], default=0)
    retweet_count = models.fields.IntegerField(validators=[MinValueValidator(0)], default=0)
    text = models.fields.CharField(max_length= 280)
    polarity = models.fields.IntegerField(null=True)

    
    def __str__(self):
        return f"{self.text}"

#Fonction qui permet de recuperer le texte(ensemble de tweet), les traite(cree trois colones 'tag', 'hashtag' eet 'ponc'. compte le nombre dee tag, hashtag et dmonc de chaque tweet et l'insere). En bref, cette fonction faire l'encodage
    def gen_meta(self):
        df = [findMail(self.text)+findLink(self.text), findHashtag(self.text), find_interogation(self.text)+find_exclamation(self.text)]
        return df

    def model_pred(self):
        df=self.gen_meta()
        y= step_func(df[0]*self.pcp[0]+df[1]*self.pcp[1]+df[2]*self.pcp[2]-self.pcp[3])
        return y
    

class SMOModel:
    """Container object for the model used for sequential minimal optimization."""
    
    def __init__(self, X, y, C, kernel, alphas, b, errors):
        self.X = X               # training data vector
        self.y = y               # class label vector
        self.C = C               # regularization parameter
        self.kernel = kernel     # kernel function
        self.alphas = alphas     # lagrange multiplier vector
        self.b = b               # scalar bias term
        self.errors = errors     # error cache
        self._obj = []           # record of objective function value
        self.m = len(self.X)     # store size of training set

    def linear_kernel(x, y, b=1):
        """Returns the linear combination of arrays `x` and `y` with
        the optional bias term `b` (set to 1 by default)."""
        
        return x @ y.T + b # Note the @ operator for matrix multiplication

    def gaussian_kernel(x, y, sigma=1):
        """Returns the gaussian similarity of arrays `x` and `y` with
        kernel width parameter `sigma` (set to 1 by default)."""
        
        if np.ndim(x) == 1 and np.ndim(y) == 1:
            result = np.exp(- (np.linalg.norm(x - y, 2)) ** 2 / (2 * sigma ** 2))
        elif (np.ndim(x) > 1 and np.ndim(y) == 1) or (np.ndim(x) == 1 and np.ndim(y) > 1):
            result = np.exp(- (np.linalg.norm(x - y, 2, axis=1) ** 2) / (2 * sigma ** 2))
        elif np.ndim(x) > 1 and np.ndim(y) > 1:
            result = np.exp(- (np.linalg.norm(x[:, np.newaxis] - y[np.newaxis, :], 2, axis=2) ** 2) / (2 * sigma ** 2))
        return result

    # Objective function to optimize

    def objective_function(alphas, target, kernel, X_train):
        """Returns the SVM objective function based in the input model defined by:
        `alphas`: vector of Lagrange multipliers
        `target`: vector of class labels (-1 or 1) for training data
        `kernel`: kernel function
        `X_train`: training data for model."""
        
        return np.sum(alphas) - 0.5 * np.sum((target[:, None] * target[None, :]) * kernel(X_train, X_train) * (alphas[:, None] * alphas[None, :]))


    # Decision function

    def decision_function(alphas, target, kernel, X_train, x_test, b):
        """Applies the SVM decision function to the input feature vectors in `x_test`."""
        
        result = (alphas * target) @ kernel(X_train, x_test) - b
        return result


"""    def pretreat(self):

        text = self.text
        text = text.lower()
        text = text.replace('\n', ' ').replace('\r', '')
        text = ' '.join(text.split())
        text = re.sub(r"[A-Za-z\.]*[0-9]+[A-Za-z%°\.]*", "", text)
        text = re.sub(r"(\s\-\s|-$)", "", text)
        text = re.sub(r"[,\!\?\%\(\)\/\"]", "", text)
        text = re.sub(r"\&\S*\s", "", text)
        text = re.sub(r"\&", "", text)
        text = re.sub(r"\+", "", text)
        text = re.sub(r"\#", "", text)
        text = re.sub(r"\$", "", text)
        text = re.sub(r"\£", "", text)
        text = re.sub(r"\%", "", text)
        text = re.sub(r"\:", "", text)
        text = re.sub(r"\@", "", text)
        text = re.sub(r"\-", "", text)

        return text

    def treat(self):
        return TextBlob(self.pretreat(), pos_tagger=PatternTagger(), analyzer=PatternAnalyzer()).sentiment[0]
"""