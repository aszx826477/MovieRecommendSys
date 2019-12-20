import sys
import random
import os
import csv
import codecs
import pymysql
from django.shortcuts import render, redirect
from users.models import Resulttable, Insertposter
from .forms import RegisterForm

random.seed(0)


def register(request):
    '''Register page and form submit'''
    # request method must be POST
    if request.method == 'POST':
        form = RegisterForm(request.POST)

        # verify the uploaded data
        if form.is_valid():
            form.save()
            return redirect('/') # register sucessfully and return index
    else:
        # request not POST, show register form
        form = RegisterForm()

    # return register.html containing a form
    return render(request, 'users/register.html', context={'form': form})

def index(request):
    '''Return index'''
    return render(request, 'users/..//index.html')


def check(request):
    '''Return index'''
    return render(request, 'users/..//index.html')

def get_key(dict, value):
    return [k for k, v in dict.items() if v == value]

def get_imdbId(movieId):
    import pandas as pd
    links = pd.read_csv('users/rating/ml-latest-small/links.csv')
    imdbId = links[links['movieId'] == movieId]['imdbId'].values[0]
    return str(imdbId)

def showmessage(request):
    '''Show the user's scored movies'''
    userId = int(request.GET['userId']) + 1000
    user_movie = []
    data = Resulttable.objects.filter(userId=userId)
    
    try:
        conn = get_conn()
        cur = conn.cursor()
        # Insertposter.objects.filter(userId=USERID).delete()
        for row in data:
            cur.execute('select * from moviegenre where imdbId = %s', get_imdbId(row.movieId))
            rr = cur.fetchall()
            for imdbId, title, poster in rr:
                user_movie.append([title, row.rating])
    
    finally:
        conn.close()

    return render(request, 'users/message.html', locals())

def ranking(data, mode='dfm', df='item'):
    from users.rating.utils import generate_rating_matrix, generate_scores_matrix, recalling, \
                  naive_ranking, ranking, generate_data, reduce_dimension, dfm_ranking
    import numpy as np
    import pickle
    import pandas as pd

    user = []
    user = np.zeros(8728)

    # read the mapping from movieId to index
    with open('ml-latest-small/movies-to-index.pickle', 'rb') as f:
        ref = pickle.load(f)
    

    for row in data:
        user[get_key(ref, row.movieId)] = row.rating
    
    rating = np.load('ml-latest-small/rating_matrix.npy')
    scores = np.load('ml-latest-small/scores_matrix.npy')

    if df == 'item':
        candidate = recalling(rating, user=user, k=3, scheme='item')
    elif df =='user':
        candidate = recalling(rating, user=user, k=3, scheme='user')
    elif df == 'both':
        candidate = recalling(rating, user=user, k=3, scheme='both')
    else:
        candidate = []

    if mode == 'dfm':
        result = dfm_ranking(rating, scores, user=user, candidate=candidate)
    elif mode == 'rank':
        result = ranking(rating, scores, user=user, candidate=candidate)
    elif mode == 'naive':
        result = naive_ranking(scores, user=user, candidate=candidate)
    else:
        result = []


    links = pd.read_csv('ml-latest-small/links.csv')
    result_pre_process = []
    for i in result:
        result_pre_process += links[links['movieId'] == ref[i]]['imdbId'].values.tolist()
    
    return result_pre_process

def recommend(request):

    matrix = []
    userId = int(request.GET['userId2']) + 1000


    read_mysql_to_csv('users/static/users_resulttable.csv', userId)  
    
    data = Resulttable.objects.filter(userId=userId)
    workspace = os.getcwd()
    os.chdir(workspace + '/users/rating')
    matrix = ranking(data, mode='dfm', df='user')
    os.chdir(workspace)
    
    try:
        conn = get_conn()
        cur = conn.cursor()
        Insertposter.objects.filter(userId=userId).delete()
        for i in matrix:
            cur.execute('select * from moviegenre where imdbId = %s', i)
            rr = cur.fetchall()
            for imdbId, title, poster in rr:

                if(Insertposter.objects.filter(title=title)):
                    continue
                else:
                    Insertposter.objects.create(userId=userId, title=title, poster=poster)

    finally:
        results = Insertposter.objects.filter(userId=userId)
        conn.close()

    return render(request, 'users/movieRecommend.html', locals())




def insert(request):
    userId = int(request.GET["userId"]) + 1000
    rating = float(request.GET["rating"])
    movieId = int(request.GET["movieId"])


    Resulttable.objects.create(userId=userId, rating=rating, movieId=movieId)

    return render(request, 'index.html', {'userId': userId, 'rating': rating, 'movieId': movieId})

def get_conn():
    '''Get MySQL connection, need to set the SQL user and password here'''
    conn = pymysql.connect(host='127.0.0.1', port=3306, user='******', passwd='******', db='rmsys', charset='utf8')
    return conn

def query_all(cur, sql, args):
    '''Query to fetchall'''
    cur.execute(sql, args)
    return cur.fetchall()


def read_mysql_to_csv(filename, userId):
    '''Read mysql to users_resulttable.csv file'''
    with codecs.open(filename=filename, mode='w', encoding='utf-8') as f:
        write = csv.writer(f, dialect='excel')
        conn = get_conn()
        cur = conn.cursor()
        cur.execute('select * from users_resulttable')
        print(userId)
        sql = ('select * from users_resulttable WHERE userId = %s')
        results = query_all(cur=cur, sql=sql, args=userId)
        for result in results:
            write.writerow(result[1:])
        conn.close()

