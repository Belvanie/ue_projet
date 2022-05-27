# listings/forms.py
from email.policy import default
from django import forms
from django.core.validators import MaxValueValidator,MinValueValidator
import datetime
import dateutil.parser
import unicodedata
from django.core.exceptions import ValidationError
from django.utils.translation import gettext_lazy as _
from django.utils import timezone



class ContactUsForm(forms.Form):
   name = forms.CharField(required = False)
   email = forms.EmailField()
   message = forms.CharField(max_length=1000)


def start_time_validator(value):
   week = datetime.timedelta(days=6, hours=23)
   if value < (timezone.now()-week):
      raise ValidationError(_('%(value)s is not a valid value of start_time'), params={'value': value},)

def end_time_validator(value):
   if value > timezone.now():
      raise ValidationError(_('%(value)s is not a valid value of end_time'), params={'value': value},)


class QueryForm(forms.Form):

   
   keywords = forms.CharField(required = True, max_length= 350)
   start_time = forms.DateTimeField (required = True, validators=[start_time_validator], error_messages={'required': 'start_time est obligatoire', 'invalid': 'start_time n\'est pas valide'})
   end_time = forms.DateTimeField(required = True, validators=[end_time_validator], error_messages={'required': 'end_time est obligatoire', 'invalid': 'end_time n\'est pas valide'})
   max_results = forms.IntegerField(min_value=10)
    