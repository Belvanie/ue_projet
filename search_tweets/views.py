from django.shortcuts import render, redirect, get_object_or_404
from django.http import HttpResponse
from search_tweets.models import Tweet, Data, Query
from search_tweets.forms import ContactUsForm, QueryForm
import datetime
import dateutil.parser
from django.contrib import messages
import json
# Create your views here.

def config(request):
    if request.method == 'POST':
        form = QueryForm(request.POST)
        
        if form.is_valid():
            keywords = form.cleaned_data['keywords'],
            start_time = form.cleaned_data['start_time'].isoformat(),
            end_time = form.cleaned_data['end_time'].isoformat(),
            max_results = form.cleaned_data['max_results']

            query = Query(keywords,
                      start_time,
                      end_time,
                      10                   
                        )
            response = query.connect_to_endpoint()
            if response.status_code == 200:
                query.max_results = max_results
                response = query.connect_to_endpoint()    
                response = response.json()
                data = Data()
                data.save()
                data.generate(response, keywords, "wils")
                data.save()
                #tweets_id= data.tweets.split()
                #print(tweets_id)
                tweets = Tweet.objects.filter(data=data.pk)
                messages.success(request, "données collectées  avec succes")
                return render(request,
                            'search_tweets/validation.html',
                            {'tweets': tweets,}
                            )
            else:
                messages.error(request, "echec de la requete bien lire la notice avant d'entrer la requete")
    else:
        form = QueryForm()
    return render(request,
            'search_tweets/config.html',
            {'form': form})


"""def validation(request, tweets_id):
    
    return render(request,
                'search_tweets/validation.html',
                {'tweets': tweets})
"""
def contact(request):
    
    if request.method == 'POST':
        # créer une instance de notre formulaire et le remplir avec les données POST
        form = ContactUsForm(request.POST)
        if form.is_valid():
            send_mail(
            subject=f'Message from {form.cleaned_data["name"] or "anonyme"} via AST Contact Us form',
            message=form.cleaned_data['message'],
            from_email=form.cleaned_data['email'],
            recipient_list=['admin@AST.xyz'],
                    )
            return redirect('email-sent')
    else:
        # ceci doit être une requête GET, donc créer un formulaire vide
        form = ContactUsForm()

    return render(request,
          'search_tweets/contact.html',
          {'form': form})  # passe ce formulaire au gabarit