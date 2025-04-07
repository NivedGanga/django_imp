from django.shortcuts import render
from django.views.decorators.csrf import csrf_exempt
from rest_framework.decorators import api_view
from .tasks import enhance_face_resolution ,get_enhanced_image_url, set_enhanced_image_url
from rest_framework.response import Response

# Create your views here.
@csrf_exempt
@api_view(['POST'])
def enhance_image(request):
    url = request.data.get('url')
    print(f"URL: {url}")
    exists, ehancedUrl = get_enhanced_image_url(url)
    if exists:
        print(f"Enhanced image URL: {ehancedUrl}")
        return Response({"image": ehancedUrl})
    else:
        newUrl = enhance_face_resolution(url)
        success, message = set_enhanced_image_url(url, newUrl.get('image'))
        if not success:
            return Response({"error": message}, status=400)
        else:
            return Response( newUrl)
