# Movie Recommendation System

If you want to run the code, you need to make sure that your environment statisfies the following requirement:

* `Python 3`
* `Django`
* `pymsql`, `numpy`, `pandas`, `sklearn`, `tensorflow`
* `MySQL`

Our project is tested on `Ubuntu 18.04 x64`. If your system environment is different with us, it may cause some problems.

## Download

You need to download our pre-processed dataset `ml-latest-small` from [360Yun](https://yunpan.360.cn/surl_yu7vYs4Nkyb) ( ml-latest-small.zip ) which contains our pre-processed and itermediate data structure. Put the folder `ml-latest-small` in `Movierecommend\users\rating\`.

We write three kinds of ranking algorithms:

* naive_ranking
* ranking
* dfm_ranking

Look up the funtion `ranking(data, mode, df)` in `view.py`. You can send the parameter into the funtion to test different methods.

```
mode = 'naive' # naive ranking model
     = 'rank'  # ranking model
     = 'dfm'   #  dfm model

df   = 'user'  # user-based collaborative filter
     = 'item'  # item-based collaborative filter 
     = 'both'  # intersection of two collaborative filters
```

To run the second model, it need to pre train and the trained model can be downloaded at [360Yun](https://yunpan.360.cn/surl_yu7vuqHhAf3) ( save.zip ). If you want to test it, please decompress it to `users\rating`.


## Configure database

Create a database and then configure the SQL connection settings in `django_auth_example\settings.py`. Modify the `get_conn()` function in `views.py`. 

At the root of the project, run 

    python manage.py migrate 

to migrate the database tables. Moreover, you have to create a new table moviegenre and import the `ml-latest-small\process\moviegenre.csv` into it.

## Run server
Finally, after doing all of these, you can run the server by

    python manage.py runserver

