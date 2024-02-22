from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS  
from pymongo import MongoClient
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input
import numpy as np
import pickle
from dotenv import load_dotenv
import os

app = Flask(__name__)
CORS(app)

load_dotenv()  # This loads the environment variables from .env

mongo_uri = os.getenv('MONGO_URI')
database_name = os.getenv('DATABASE_NAME')
collection_name = os.getenv('COLLECTION_NAME')
model_path = os.getenv('MODEL_PATH')

client = MongoClient(mongo_uri)
database = client[database_name]
collection = database[collection_name]

# Load the pre-trained ResNet50 model
model = load_model(model_path)

IMAGE_DATA_PATH = 'data'

# Example endpoint to get a list of image filenames for a specific label
@app.route('/get_images/<label>')
def get_images(label):
    image_folder_path = f'{IMAGE_DATA_PATH}/{label}'
    print(f"Image folder path: {image_folder_path}")
    try:
        # Return a JSON response with a list of image filenames
        image_filenames = os.listdir(image_folder_path)
        print(f"Image filenames: {image_filenames}")
        return jsonify(image_filenames)
    except FileNotFoundError:
        return jsonify({"error": "Label not found"}), 404

@app.route('/get_image/<label>/<filename>')
def get_image(label, filename):
    image_path = os.path.join(IMAGE_DATA_PATH, label, filename)
    return send_from_directory(IMAGE_DATA_PATH, os.path.join(label, filename))


@app.route('/get_plant_info', methods=['POST'])
def get_plant_info():
    # Get the image file from the frontend
    image_file = request.files.get('image')

    # Process the image file and get the plant label
    plant_label = process_image(image_file)

    # Query MongoDB for plant information
    plant_info = collection.find_one({'label': plant_label})


    if plant_info:
        plant_info['_id'] = str(plant_info['_id'])
        print(jsonify(plant_info))
        return jsonify(plant_info)
    else:
        return jsonify({'error': 'Plant not found'}), 404

def process_image(image_file):

    # Load class indices
    with open('class_indices.pkl', 'rb') as f:
        class_indices = pickle.load(f)

    # Save the uploaded image temporarily
    temp_path = "temp_image.jpg"
    image_file.save(temp_path)

    # Load and preprocess the image for model prediction
    img = image.load_img(temp_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)

    # Use the loaded model to predict the plant label
    predictions = model.predict(img_array)
    predicted_class = np.argmax(predictions)

    # Get the class labels from the loaded class indices
    class_labels = list(class_indices.keys())
    
    # Map the predicted class index to the corresponding label
    plant_label = class_labels[predicted_class]

    # Remove the temporary image file
    os.remove(temp_path)

    return plant_label

if __name__ == '__main__':
    app.run(debug=True)