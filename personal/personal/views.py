from . import views
from django.conf.urls import url, include
from django.conf import settings
from django.contrib import admin
from django.shortcuts import render
from django.template import Template
from django.urls import path
from django.views.generic import TemplateView




class HomePage(TemplateView):
    template_name = 'personal/index.html'


class TestPage(TemplateView):
    template_name = 'test.html'

class ThanksPage(TemplateView):
    template_name = 'thanks.html'


############################
def other(request):
    template_name = 'other.html'
    print ( 'Other' )
    return render(request,'other.html')

def date_generator_form(request):
    template_name = 'date_generator_form.html'
    print ( 'Other' )
    return render(request,'date_generator_form.html')
