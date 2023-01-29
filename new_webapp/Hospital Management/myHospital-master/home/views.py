from django.shortcuts import render
from django.contrib import messages
from django.http.response import StreamingHttpResponse
from .camera import MaskDetect
from .Sdmonitoring import SocialDistancing

count = 0


def home(request):
    messages.add_message(request, messages.INFO, 'Welcome to The Hospital Portal.')
    return render(request, 'home/base.html')


def gen(camera):
    while True:
        frame = camera.get_frame()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')


def mask_feed(request):
    return StreamingHttpResponse(gen(MaskDetect()),
                                 content_type='multipart/x-mixed-replace; boundary=frame')


def sd(request):
    return render(request, 'home/sd.html')


def generate(Sdmonitoring):
    while True:
        frame, num = Sdmonitoring.startsd()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')


def social_distancing(request):
    return StreamingHttpResponse(generate(SocialDistancing()),
                                 content_type='multipart/x-mixed-replace; boundary=frame')


def index(request):
    context = {'foo': 'bar'}
    return render(request, 'index.html', context)