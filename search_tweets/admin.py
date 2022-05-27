from typing import List
from django.contrib import admin

# Register your models here.
from search_tweets.models import Data, Tweet
admin.site.register(Data)
admin.site.register(Tweet)
