"""server_django URL Configuration

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/2.1/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.conf.urls import url
from django.urls import path
from django.contrib import admin
from . import views

urlpatterns = [

    path('auth/login', views.auth_login, name='login'),
    path('auth/logout', views.auth_logout, name='logout'),
    path('admin/', views.admin, name='admin'),
    path('reg/', views.reg, name='registration'),
    # path('home/', views.Qviews, name='home'),
    path('help/', views.help, name='help'),
    path('about/', views.about, name='about'),
    path('input/', views.Qviews, name='input'),
    path('backup/', views.BackupViews, name='backup'),
    path('restore/', views.RestoreViews, name='restore'),
    path('image/<arg>', views.GraphViews, name='image'),
    path('image/', views.GraphPrintViews, name='imageprint'),
    path('', views.index, name='index'),
]
