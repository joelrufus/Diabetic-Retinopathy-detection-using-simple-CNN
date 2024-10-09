#Code to train the model 

import pandas as pd
import numpy as np
import cv2
import os
from sklearn.utils import class_weight
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.data import Dataset
import tensorflow as tf

df = pd.read_csv(r'') #enter the path of your csv file

print(df.head())

def load_images_from_folder(folder, df):
    images = []
    labels = []
    class_folders = {
        0: 'No_DR',
        1: 'Mild',
        2: 'Moderate',
        3: 'Severe',
        4: 'Proliferate_DR'
    }
    for index, row in df.iterrows():
        diagnosis = row['diagnosis']
        class_folder = class_folders[diagnosis]
        img_path = os.path.join(folder, class_folder, row['id_code'] + '.png')
        if os.path.isfile(img_path):
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)  
            if img is not None:
                images.append(img)
                labels.append(diagnosis)
            else:
                print(f"Failed to load image: {img_path}")
        else:
            print(f"Image file does not exist: {img_path}")
    return images, labels

folder = r'' #enter the path of the dataset folder containing  classes
images, labels = load_images_from_folder(folder, df)

print(f"Number of images loaded: {len(images)}")
print(f"Number of labels loaded: {len(labels)}")

images = np.array(images)
labels = np.array(labels).astype(int)  

print("Sample labels:", labels[:10])

processed_images = np.array([cv2.resize(img, (224, 224)) for img in images])
processed_images = processed_images.reshape(processed_images.shape[0], 224, 224, 1)  


processed_images = processed_images / 255.0

print("Processed images shape:", processed_images.shape)

unique_classes = np.unique(labels)
class_weights = class_weight.compute_class_weight('balanced', classes=unique_classes, y=labels)
class_weights = {int(c): weight for c, weight in zip(unique_classes, class_weights)}

print("Class Weights:", class_weights)


model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 1)),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(5, activation='softmax')  
])


model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])


dataset = Dataset.from_tensor_slices((processed_images, labels))


dataset = dataset.shuffle(buffer_size=1024).repeat().batch(32)


history = model.fit(dataset, epochs=10, steps_per_epoch=len(processed_images) // 32, class_weight=class_weights, validation_data=(processed_images, labels))

model.save(r'') #enter the path in which u want to save the model for testing and save it in .keras format

print("Model architecture and weights saved successfully in .keras format")
