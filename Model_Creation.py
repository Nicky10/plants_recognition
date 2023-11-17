#!/usr/bin/env python
# coding: utf-8

# In[1]:


import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau  # Import ReduceLROnPlateau
import os
import random
from shutil import copyfile

dataset_root = 'data'
train_split = 0.8

# Create directories for train and test data
train_data_dir = 'train'
test_data_dir = 'test'
os.makedirs(train_data_dir, exist_ok=True)
os.makedirs(test_data_dir, exist_ok=True)


# In[2]:


for plant_category in os.listdir(dataset_root):
    if os.path.isdir(os.path.join(dataset_root, plant_category)):
        images = os.listdir(os.path.join(dataset_root, plant_category))
        num_images = len(images)
        num_train = int(train_split * num_images)

        random.shuffle(images)

        for i, image in enumerate(images):
            src = os.path.join(dataset_root, plant_category, image)
            if i < num_train:
                dst = os.path.join(train_data_dir, plant_category, image)
            else:
                dst = os.path.join(test_data_dir, plant_category, image)
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            copyfile(src, dst)


img_width, img_height = 224, 224  # ResNet50 input size
batch_size = 32

train_dir = 'train'
test_dir = 'test'

# Data augmentation and preprocessing
train_datagen = ImageDataGenerator(
    rescale=1.0 / 255,
    rotation_range=30,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

test_datagen = ImageDataGenerator(rescale=1.0 / 255)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='categorical'
)

test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='categorical'
)

# Load pre-trained ResNet50 model without top (fully connected) layers
base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(img_width, img_height, 3))

# Add custom top layers for your classification task
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(512, activation='relu')(x)
x = Dropout(0.5)(x)
predictions = Dense(len(train_generator.class_indices), activation='softmax')(x)

# Create the final model
model = Model(inputs=base_model.input, outputs=predictions)

# Freeze the layers of the base model for transfer learning
for layer in base_model.layers:
    layer.trainable = False

# Compile the model with a lower learning rate
model.compile(optimizer=Adam(learning_rate=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])

# Define callbacks, including early stopping and learning rate reduction
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=1e-6)

# Train the model with more epochs and callbacks
epochs = 50
history = model.fit(
    train_generator,
    epochs=epochs,
    validation_data=test_generator,
    callbacks=[early_stopping, reduce_lr]
)

# Evaluate the model
test_loss, test_acc = model.evaluate(test_generator)
print("Test accuracy:", test_acc)

# Save the model
model.save('plant_type_classifier_resnet_improved.h5')