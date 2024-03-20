from django.urls import path
from . import views

urlpatterns = [
    path('', views.home, name="home"),
    path('pitches', views.pitches, name="pitches"),
    path('projects', views.projects, name="projects")
]