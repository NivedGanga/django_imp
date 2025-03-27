import dlib
import time
import requests
import numpy as np
import os
from PIL import Image
from io import BytesIO
import tempfile

async def get_face_embeddings(image_url):
    start = time.time()
    # Load models
    predictor_path = './vector_extract/shape_predictor_5_face_landmarks.dat'
    face_rec_model_path = './vector_extract/dlib_face_recognition_resnet_model_v1.dat'

    detector = dlib.get_frontal_face_detector()
    dlib.DLIB_USE_CUDA = False  # Ensure CPU mode

    sp = dlib.shape_predictor(predictor_path)
    facerec = dlib.face_recognition_model_v1(face_rec_model_path)

    descriptors = []  # To store vector embeddings
    images = []  # To store numpy array of image and face landmarks

    # Fetch the image from the URL
    response = requests.get(image_url)
    if response.status_code != 200:
        raise ValueError("Error fetching image from URL")
    
    img = Image.open(BytesIO(response.content))
    img = img.convert('RGB')  # Ensure the image is in RGB format
    
    # Save the image to a temporary file
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as temp_file:
        img.save(temp_file.name)
        temp_img_file = temp_file.name
    
    # Load image in dlib format
    img = dlib.load_rgb_image(temp_img_file)

    # Detect faces
    dets = detector(img, 1)
    print("Number of faces detected:", len(dets))

    # Extract face embeddings
    for d in dets:
        shape = sp(img, d)
        face_descriptor = facerec.compute_face_descriptor(img, shape)
        descriptors.append(face_descriptor)
        images.append((img, shape))

    # Cleanup
    os.remove(temp_img_file)
    
    print("Execution time:", time.time() - start)
    return descriptors

def get_face_embeddings_sync(image_file):
    start = time.time()
    # Load models
    predictor_path = './vector_extract/shape_predictor_5_face_landmarks.dat'
    face_rec_model_path = './vector_extract/dlib_face_recognition_resnet_model_v1.dat'

    detector = dlib.get_frontal_face_detector()
    dlib.DLIB_USE_CUDA = False  # Ensure CPU mode

    sp = dlib.shape_predictor(predictor_path)
    facerec = dlib.face_recognition_model_v1(face_rec_model_path)

    descriptors = []  # To store vector embeddings
    images = []  # To store numpy array of image and face landmarks

    # Fetch the image from the URL
    img = Image.open(image_file)
    img = img.convert('RGB') 
    
    # Save the image to a temporary file
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as temp_file:
        img.save(temp_file.name)
        temp_img_file = temp_file.name
    
    # Load image in dlib format
    img = dlib.load_rgb_image(temp_img_file)

    # Detect faces
    dets = detector(img, 1)
    print("Number of faces detected:", len(dets))

    # Extract face embeddings
    for d in dets:
        shape = sp(img, d)
        face_descriptor = facerec.compute_face_descriptor(img, shape)
        descriptors.append(face_descriptor)
        images.append((img, shape))

    # Cleanup
    os.remove(temp_img_file)
    
    print("Execution time:", time.time() - start)
    return descriptors[0]

def get_number_of_faces(image_file):
    import dlib
    import time
    import tempfile
    from PIL import Image
    
    start = time.time()
    # Load models
    predictor_path = './vector_extract/shape_predictor_5_face_landmarks.dat'

    detector = dlib.get_frontal_face_detector()
    dlib.DLIB_USE_CUDA = False  # Ensure CPU mode

    sp = dlib.shape_predictor(predictor_path)

    # Save the uploaded file to a temporary file
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as temp_file:
        for chunk in image_file.chunks():
            temp_file.write(chunk)
        temp_img_file = temp_file.name
    
    # Load image in dlib format
    img = dlib.load_rgb_image(temp_img_file)

    # Detect faces
    dets = detector(img, 1)
    print("Number of faces detected:", len(dets))
    
    # Clean up the temporary file
    import os
    os.unlink(temp_img_file)
    
    return len(dets)
