from django.db import models
from django.contrib.auth.models import AbstractUser

class User(AbstractUser):
    ''' User -> nickname'''
    nickname = models.CharField(max_length=50, blank=True)

    def __str__(self):
        return self.nickname

    class Meta(AbstractUser.Meta):
        pass


class Resulttable(models.Model):
    ''' Table that saves the users' scored films results '''
    userId = models.IntegerField(null=True)     # Field name made lowercase.
    movieId = models.IntegerField()             # Field name made lowercase.
    rating = models.DecimalField(max_digits=3, decimal_places=1, blank=True, null=True)


    def __str__(self):
        return self.userId+':'+self.rating


class Insertposter(models.Model):
    ''' Poster of films '''
    userId = models.IntegerField(null=True)
    title = models.CharField(max_length=200, blank=True, null = False)
    poster = models.CharField(max_length=500, blank=True, null=True)

    def __str__(self):
        return self.userId + ':' + self.poster

