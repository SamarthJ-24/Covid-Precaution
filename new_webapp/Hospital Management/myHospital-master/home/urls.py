from django.contrib import admin
from django.urls import path, include

from . import views
from .views import *


urlpatterns = [
    path('', home),
    path('home', views.home),
    path('sd', views.sd),
    path('mask_feed', views.mask_feed, name='mask_feed'),
    path('sd_monitoring', views.social_distancing, name='sd_monitoring')
]
