from flask import Flask, request, jsonify
from flask_cors import CORS  # Import CORS from flask_cors
from pymongo import MongoClient
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input
import numpy as np
import pickle
import os

app = Flask(__name__)
CORS(app)

# Replace these values with your MongoDB connection details
mongo_uri = "mongodb+srv://nicolasfelipedelgado:ERk7TbdktbAeHq6F@cluster0.qs1kmuw.mongodb.net/?retryWrites=true&w=majority"
database_name = "plants"
collection_name = "plants_descriptions"

# Replace with the path to your trained model
model_path = "plant_type_classifier_resnet_improved.h5"

# Connect to MongoDB
client = MongoClient(mongo_uri)
database = client[database_name]
collection = database[collection_name]

# Load the pre-trained ResNet50 model
model = load_model(model_path)

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