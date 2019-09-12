from django import forms
from django.contrib.auth.forms import UserCreationForm
from django.contrib.auth.models import User
from django.forms import ModelForm
from mainApp.models import *



class MyUserForm(UserCreationForm):
    class Meta:
        model = User
        fields = ('first_name', 'username', 'email', 'password1', 'password2')
        help_texts = {
            'username': None,
            'password1': None,
            'password2': None,
        }

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.fields['first_name'].label = 'Имя'
        self.fields['username'].label = 'Login'
        self.fields['email'].label = 'эл.почта'
        self.fields['password1'].label = 'Пароль'
        self.fields['password2'].label = 'Повторите пароль'


class QForm(ModelForm):
    class Meta:
        model = Q
        fields = ('value_q',)
        exclude = ["id_q"]

class NuForm(ModelForm):
    class Meta:
        model = Nu
        fields = ('value_nu',)
        exclude = ["id_nu"]

class RForm(ModelForm):
    class Meta:
        model = R
        fields = ('value_r',)
        exclude = ["id_r"]

class MatrixForm(ModelForm):
    class Meta:
        model = Matrixsize
        fields = ('id_k', 'id_m',)
        exclude = ["id_km"]
