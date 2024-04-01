from django.contrib import admin

from main.models import Stock, Partner, Resource
# Register your models here.

@admin.register(Stock)
class StockAdmin(admin.ModelAdmin):
    list_display = ('name', 'ticker')


@admin.register(Partner)
class PartnerAdmin(admin.ModelAdmin):
    list_display = ('name', 'linkedin')
    


@admin.register(Resource)
class ResourceAdmin(admin.ModelAdmin):
    list_display = ('type', 'url')
    