import os
import requests
from gradio_client import Client
from PIL import Image
from pathlib import Path


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



def enhance_face_resolution(image_input, output_path="upscaled_image.jpg", version="v1.2", scale=5):
    """
    Enhances the facial resolution of an image using GFPGAN via a Gradio API.
    
    Parameters:
        image_input (str): URL or local file path of the input image.
        output_path (str): Path to save the upscaled image.
        version (str): Version of GFPGAN to use (default: "v1.2").
        scale (int): Rescaling factor (default: 5).
    
    Returns:
        str: Path to the upscaled image.
    """
    if image_input.startswith("http"):
    
        local_image_path = download_image(image_input)

    elif Path(image_input).is_file():

        local_image_path = image_input

    else:

        raise ValueError("Invalid image input. Provide a valid url or file path.")
    
    client = Client("https://xintao-gfpgan.hf.space/")
    
    result = client.predict(
        local_image_path,  # Pass file path instead of bytes
        version,  # GFPGAN version
        scale,  # Rescaling factor
        api_name="/predict"
    )
    
    upscaled_image_path = result[1]  # Extract output image path
    
    # Save the upscaled image to the specified output path
    img = Image.open(upscaled_image_path)
    img.save(output_path)
    
    print(f"Upscaled image saved as: {output_path}")
    return output_path



# Example usage 
# enhanced_image = enhance_face_resolution("https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcRlfbrTnS9nPPqzQGYnDC-pMEOEdG38CpHWZuu-_exB8dCOBGaw")
# img = Image.open(enhanced_image)
# img.show()

# pip install "gradio_client<0.10"