# !pip install flask flask-ngrok pyngrok torch diffusers
!pip install torchvision
import os
import torch
import base64
from io import BytesIO
from diffusers import StableDiffusionPipeline
from flask import Flask, jsonify, request, send_from_directory
from pyngrok import ngrok
from PIL import Image

# Load the Stable Diffusion pipeline
model_id = "CompVis/stable-diffusion-v1-4"

# Set Ngrok auth token
ngrok.set_auth_token('2sCWllt2VAwYxCE3h3mfB0zYcpd_2kSEqi8P7LPrsFk1U77hm')

# Initialize the pipeline
pipeline = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
pipeline = pipeline.to("cuda")  # Move pipeline to GPU for faster generation

# Create Flask app
app = Flask(__name__)

# Directory to save generated images
OUTPUT_DIR = "generated_images"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Function to generate an image based on a text prompt
def generate_image(prompt, width=512, height=512):
    try:
        # Set the size of the generated image
        pipeline.scheduler.config.image_size = (width, height)

        # Generate the image
        image = pipeline(prompt, height=height, width=width).images[0]

        # Create a file name and save the image
        output_path = os.path.join(OUTPUT_DIR, f"{prompt.replace(' ', '_')}.png")
        image.save(output_path)
        print(f"Image saved at {output_path}")

        # Convert the image to Base64 string
        with BytesIO() as img_byte_arr:
            image.save(img_byte_arr, format='PNG')
            img_byte_arr.seek(0)
            base64_image = base64.b64encode(img_byte_arr.read()).decode('utf-8')

        return base64_image  # Return the Base64 string of the image
    except Exception as e:
        print(f"Error generating image: {str(e)}")
        return None

# API route to generate image
@app.route("/generate_image", methods=["POST"])
def generate_image_api():
    # Ensure Content-Type is application/json
    if request.content_type != 'application/json':
        return jsonify({"error": "Invalid Content-Type, expected 'application/json'"}), 415

    try:
        # Get prompt, width, and height from the request body, with default values
        data = request.get_json()
        prompt = data.get("prompt", "A white dog")  # Default to a white dog if no prompt is provided
        width = int(data.get("width", 512))  # Default to 512 if no width is provided
        height = int(data.get("height", 512))  # Default to 512 if no height is provided
        print(f"Received prompt: {prompt}")
        print(f"Received width: {width}")
        print(f"Received height: {height}")

        # Generate the image and get the Base64 string
        base64_image = generate_image(prompt, width=width, height=height)
        if not base64_image:
            return jsonify({"error": "Failed to generate image"}), 500

        # Return the Base64 string of the generated image
        return jsonify({"image_base64": base64_image}), 200

    except Exception as e:
        print(f"Error in /generate_image endpoint: {str(e)}")
        return jsonify({"error": "An error occurred during image generation."}), 500

# Run the Flask app with Ngrok tunnel
if __name__ == "__main__":
    # Open an HTTP tunnel on port 5000
    public_url = ngrok.connect(5000)
    print(f" * Ngrok tunnel \"{public_url}\" -> \"http://127.0.0.1:5000\"")

    # Run the Flask app
    app.run(debug=True, use_reloader=False)  # Disable reloader as ngrok interferes with it

