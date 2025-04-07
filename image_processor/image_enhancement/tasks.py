import os
import requests
from gradio_client import Client
from PIL import Image
from pathlib import Path
from cloudinary.uploader import upload
from io import BytesIO
from django.core.exceptions import ObjectDoesNotExist
from vector_extract.models import FileStore

def download_image(image_url, save_path="input_image.jpg"):
    """Downloads an image from a URL and saves it locally."""
    response = requests.get(image_url, stream=True)
    if response.status_code == 200:
        with open(save_path, "wb") as file:
            for chunk in response.iter_content(1024):
                file.write(chunk)
        return save_path
    else:
        raise Exception("Failed to download image.")




def enhance_face_resolution(image_input, version="v1.2", scale=5):
    try:
        # Download the image
        local_image_path = download_image(image_input)
        
        # Process the image
        client = Client("https://xintao-gfpgan.hf.space/")
        result = client.predict(
            local_image_path,
            version,
            scale,
            api_name="/predict"
        )
        
        upscaled_image_path = result[1]
        
        # Open the enhanced image and prepare for upload
        with Image.open(upscaled_image_path) as img:
            img_bytes = BytesIO()
            img.save(img_bytes, format='JPEG')
            img_bytes.seek(0)
            
            # Upload to Cloudinary
            response = upload(
                img_bytes,
                folder="enhanced_image",
            )
            output_url = response.get("secure_url")
        
        # Clean up files
        try:
            if os.path.exists(local_image_path):
                os.remove(local_image_path)
            if os.path.exists(upscaled_image_path):
                os.remove(upscaled_image_path)
        except OSError as e:
            print(f"Warning: Could not delete temporary files: {e}")
        
        print(f"Upscaled image saved as: {output_url}")
        return {"image": output_url}
        
    except Exception as e:
        # Clean up files even if an error occurs
        try:
            if 'local_image_path' in locals() and os.path.exists(local_image_path):
                os.remove(local_image_path)
            if 'upscaled_image_path' in locals() and os.path.exists(upscaled_image_path):
                os.remove(upscaled_image_path)
        except OSError as cleanup_error:
            print(f"Warning: Could not delete temporary files during error cleanup: {cleanup_error}")
        
        return {"error": "Error while processing the image", "details": str(e)}


# Example usage 
# enhanced_image = enhance_face_resolution("https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcRlfbrTnS9nPPqzQGYnDC-pMEOEdG38CpHWZuu-_exB8dCOBGaw")
# img = Image.open(enhanced_image)
# img.show()

# pip install "gradio_client<0.1



def get_enhanced_image_url(url):
    """
    Get the enhancedImageUrl for a specific file ID
    Returns: (exists: bool, result: str/None)
    """
    try:
        file_entry = FileStore.objects.get(url=url)
        if file_entry.enhancedImageUrl:
            return True, file_entry.enhancedImageUrl
        return False, None
    except ObjectDoesNotExist:
        return False, None
    except Exception as e:
        raise Exception(f"Error getting enhanced image URL: {str(e)}")

def set_enhanced_image_url(url, enhanced_url):
    """
    Set the enhancedImageUrl for a specific file ID
    Returns: (success: bool, message: str)
    """
    try:
        file_entry = FileStore.objects.get(url=url)
        file_entry.enhancedImageUrl = enhanced_url
        file_entry.save()
        return True, "Enhanced image URL updated successfully"
    except ObjectDoesNotExist:
        return False, "url not found"
    except Exception as e:
        return False, f"Error setting enhanced image URL: {str(e)}"

def get_file_details(file_id):
    """
    Get all details for a specific file ID
    Returns: FileStore object or None if not found
    """
    try:
        return FileStore.objects.get(field=file_id)
    except ObjectDoesNotExist:
        return None