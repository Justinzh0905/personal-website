from django.shortcuts import render
from django.http import HttpResponse, HttpResponseRedirect
from main.models import Stock, Resource
import yfinance as yf
import datetime
from . import chart
# Create your views here.

def home(request):
    context = {
        'title': 'Home',
        'isbn': 9781665954884,
    }
    return render(request, 'home.html', context)

def projects(request):
    context = {
        'title': 'Projects',
    }

    return render(request, 'projects.html', context)


def pitches(request):
    def stock_data(stock_model):
        stock = yf.Ticker(stock_model.ticker)

        price = round(stock.history(period='1wk').iloc[-1]['Close'],2)
        change = round(100*(price - stock_model.pprice)/stock_model.pprice, 2)
        difference = datetime.date.today() - stock_model.date
        difference_in_years = (difference.days)/365.2425 
        annualized = round(((price/stock_model.pprice)**(1/difference_in_years)-1)*100,2)
        pe = round(stock.info['trailingPE'],2)
        beta = round(stock.info['beta'],2)
        marketcap = format(round(stock.info['marketCap'] / 1000000000, 2), ',') 

        partners = [{'name': partner.name, 'linkedin': partner.linkedin} for partner in stock_model.partners.all()]
        resources = [{'type': resource.type, 'url': resource.url} for resource in Resource.objects.filter(stock=stock_model)]

        return {'name': stock_model.name, 'ticker': stock_model.ticker, 'description': stock_model.description, 'partners': partners, 'resources': resources, 'date': stock_model.date, 'price': price, 'marketcap': marketcap, 'pe': pe, 'beta': beta, 'pprice': stock_model.pprice, 'change': change, 'annualized': annualized}
    
    stocks = [stock_data(stock) for stock in Stock.objects.all().order_by('-date')]
    context = {
        'title': 'Pitches',
        'stocks': stocks
    }

    return render(request, 'pitches.html', context)