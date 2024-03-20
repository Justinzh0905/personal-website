from django.shortcuts import render
from django.http import HttpResponse, HttpResponseRedirect
# Create your views here.

def home(request):
    context = {
        'title': 'Home',
    }
    return render(request, 'home.html', context)

def projects(request):
    context = {
        'title': 'Projects',
    }

    return render(request, 'projects.html', context)


def pitches(request):
    context = {
        'title': 'Pitches',
    }
    return render(request, 'pitches.html', context)

