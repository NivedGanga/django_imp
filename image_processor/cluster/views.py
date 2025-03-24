from rest_framework.decorators import api_view, parser_classes
from rest_framework.parsers import MultiPartParser
from rest_framework.response import Response
from django.views.decorators.csrf import csrf_exempt
from vector_extract.vectorize import get_number_of_faces
from vector_extract import vectorize
from asgiref.sync import sync_to_async

# Convert synchronous functions to asynchronous
get_number_of_faces_async = sync_to_async(get_number_of_faces)
get_face_embeddings_async = sync_to_async(vectorize.get_face_embeddings)

@csrf_exempt
@api_view(['POST'])
@parser_classes([MultiPartParser])
async def process_image(request):
    print("hellosdfhg")
    print(f"Request method: {request.method}")
    print(f"Request FILES: {request.FILES}")
    for filename, file in request.FILES.items():
        print(filename)
    print(f"Request POST: {request.POST}")
    
    if "image" not in request.FILES or "event_id" not in request.POST:
        return Response({"error": "Image and event_id are required"}, status=400)
    
    image = request.FILES["image"]
    event_id = request.POST["event_id"]
    print(f"Image: {image}")
    try:
        # Use await with the async version of the function
        num_faces = await get_number_of_faces_async(image)
        print(num_faces)
        if num_faces is None:
            return Response({"error": "Error processing image"}, status=500)
        elif num_faces > 1 or num_faces == 0:
            return Response({"error": "Multiple faces detected" if num_faces > 1 else "No faces detected"}, status=400)
        else:
            # Use await with the async version of the function
            v = await get_face_embeddings_async(image)
            print(v)
            return Response({"num_faces": num_faces, "event_id": event_id})
    except Exception as e:
        return Response({"error": f"An error occurred: {str(e)}"}, status=500)