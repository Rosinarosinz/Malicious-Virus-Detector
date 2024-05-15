# Import necessary libraries
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.callbacks import ModelCheckpoint
import streamlit as st

# Load the dataset
train_data_dir = "C:\\Users\\KASIM\\Desktop\\Mal\\malevis_train_val_224x224\\train"
val_data_dir = "C:\\Users\\KASIM\\Desktop\\Mal\\malevis_train_val_224x224\\val"
classes = sorted(os.listdir(train_data_dir))
num_classes = len(classes)

# Define CNN model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(512, activation='relu'),
    Dense(num_classes, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Image data generator
train_datagen = ImageDataGenerator(rescale=1./255)
val_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
        train_data_dir,
        target_size=(128, 128),
        batch_size=32,
        class_mode='categorical')

val_generator = val_datagen.flow_from_directory(
        val_data_dir,
        target_size=(128, 128),
        batch_size=32,
        class_mode='categorical')

# Train the model with fewer epochs
checkpoint = ModelCheckpoint('best_model.keras', monitor='val_accuracy', mode='max', save_best_only=True, verbose=1)
history = model.fit(train_generator, validation_data=val_generator, epochs=3, callbacks=[checkpoint])

# Function to predict malware type
def predict_malware(image_path):
    img = tf.keras.preprocessing.image.load_img(image_path, target_size=(128, 128))
    img_array = tf.keras.preprocessing.image.img_to_array(img) / 255.0
    img_array = tf.expand_dims(img_array, 0)  # Create batch axis
    prediction = model.predict(img_array)
    predicted_class = classes[np.argmax(prediction)]
    return predicted_class

# Streamlit web application
st.title('Malware Detection in Images')
st.markdown("<h1 style='text-align: center; color: white;'>🛡️ Malware Detection in Images 🛡️</h1>", unsafe_allow_html=True)

uploaded_file = st.file_uploader("Choose an image...", type="jpg")

if uploaded_file is not None:
    # Display the uploaded image
    st.image(uploaded_file, caption='Uploaded Image.', use_column_width=True)
    st.write("")
    
    # Inform the user about the current process
    st.write("<h2 style='text-align: center; color: white;'>Current Process: Detecting the malware in the uploaded image... 🔄</h2>", unsafe_allow_html=True)
    st.write("")
    st.write("")
    
    # Perform malware detection
    class_prediction = predict_malware(uploaded_file)
    
    # Display the result
    st.write("<h2 style='text-align: center; color: white;'>The type of Malware that is present in the uploaded image is: <span style='color: #FF5733;'>{}</span></h2>".format(class_prediction), unsafe_allow_html=True)

    # Add a button to allow the user to re-upload another image
    if st.button("Upload Another Image"):
        uploaded_file = None
