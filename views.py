from django.http import HttpResponse

def index(request):
    return HttpResponse("Facial Recognition System Setup - Ready to Start!")
from django.shortcuts import render
from .models import CapturedFace

def gallery(request):
    faces = CapturedFace.objects.all().order_by('-timestamp')
    return render(request, 'gallery.html', {'faces': faces})
