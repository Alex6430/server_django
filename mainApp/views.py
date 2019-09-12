from django.contrib import auth
from django.contrib.auth.forms import UserCreationForm
from django.core.mail import send_mail
from django.shortcuts import render, redirect
from django.shortcuts import render_to_response
from django.contrib import admin

# Create your views here.
from django.template.context_processors import csrf
from django.views.decorators.csrf import csrf_protect

from mainApp.forms import MyUserForm
from mainApp.forms import QForm
from mainApp.forms import NuForm
from mainApp.models import *
from mainApp.main import printImage
from server_django import settings


def index(request):
    return render(request, 'mainApp/homePage.html')


def help(request):
    return render(request, 'mainApp/help.html')


def about(request):
    send_mail(
       'Тема почты',
        'Само сообщение',
        settings.EMAIL_HOST_USER,
        ['averina10091996@gmail.com'],
        fail_silently=False,
    )
    return render(request, 'mainApp/about.html')


def contact(request):
    return render(request, 'mainApp/contact.html')


def admin(request):
    return render(request, 'admin')


def auth_login(request):
    args = {}
    args.update(csrf(request))
    if request.POST:
        username = request.POST.get('username', '')
        password = request.POST.get('password', '')
        user = auth.authenticate(username=username, password=password)
        if user is not None:
            auth.login(request, user)
            return redirect("/")
        else:
            args['login_error'] = "пользователь не найден"
            return render_to_response("mainApp/auth.html", args)
    else:
        return render_to_response("mainApp/auth.html", args)


def auth_logout(request):
    auth.logout(request)
    return redirect("/")


def reg(request):
    args = {}
    args.update(csrf(request))
    args['form1'] = MyUserForm()
    if request.POST:
        newuser_form = MyUserForm(request.POST)
        if newuser_form.is_valid():
            newuser_form.save()
            newuser = auth.authenticate(username=newuser_form.cleaned_data['username'],
                                        password=newuser_form.cleaned_data['password1'])
            auth.login(request, newuser)
            return redirect("/")
        else:
            args['form1'] = newuser_form
    return render_to_response('mainApp/registration.html', args)


@csrf_protect
def Qviews(request):
    args = {}
    if request.POST:
        valueq = request.POST.get('value_q')
        valuenu = request.POST.get('value_nu')
        valuer = request.POST.get('value_r')
        valuek = request.POST.get('id_k')
        valuem = request.POST.get('id_m')
        document = Q(value_q=valueq)
        document.save()
        document = Nu(value_nu=valuenu)
        document.save()
        document = R(value_r=valuer)
        document.save()
        document = Matrixsize(id_k=valuek, id_m=valuem)
        document.save()
        id_grapf = printImage()
        return redirect('image', id_grapf)
    else:
        return render(request, 'mainApp/input.html')


def GraphViews(request, arg):
    args = {}
    args['path'] = Graph.objects.get(id_graph=arg).images_graph
    return render(request, 'mainApp/image.html', {'args': args})


def GraphPrintViews(request):
    args = {}
    args['pathnew'] = Graph.objects.all()
    return render(request, 'mainApp/image.html', {'args': args})


def BackupViews(request):
    backup()
    return render(request, 'mainApp/wrapper.html')


def RestoreViews(request):
    restore()
    return render(request, 'mainApp/wrapper.html')
