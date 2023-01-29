from django.urls import path, include
from recognition import views

urlpatterns = [
    path('', views.index, name='index'),
    path('mask_feed', views.mask_feed, name='mask_feed'),
]
