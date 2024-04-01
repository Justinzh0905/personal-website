from django.db import models
import datetime
import yfinance as yf
# Create your models here.

class Partner(models.Model):
    name = models.CharField(max_length=50)

    linkedin = models.URLField()

    def __str__(self):
        return self.name
    

class Stock(models.Model):
    name = models.CharField(max_length=100)

    ticker = models.CharField(max_length=4)

    description = models.TextField()

    date = models.DateField(default=datetime.date.today, blank=True)

    pprice = models.FloatField(default=None, blank=True)

    partners = models.ManyToManyField(Partner, blank=True)

    def save(self, *args, **kwargs):
        self.pprice = round(yf.Ticker(self.ticker).history(start=self.date.strftime('%Y-%m-%d')).iloc[0]['Close'], 2)
        super(Stock, self).save(*args, **kwargs)

    def __str__(self):
        return self.name

class Resource(models.Model):
    type = models.CharField(max_length=100)
    url = models.URLField()
    stock = models.ForeignKey(
        Stock,
        on_delete=models.CASCADE
    )

    def __str__(self):
        return self.type




