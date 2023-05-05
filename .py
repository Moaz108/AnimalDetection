import os
import random
import shutil
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras import regularizers
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization

from tensorflow.keras import layers,models
import seaborn as sns  # For statistical visualizations
import plotly.graph_objs as go  # For interactive visualizations
import matplotlib.pyplot as plt  # For creating static plots
    
    
    
    
    
    
    # Define the data directory
data_dir = 'C:\\Users\\sonic\\Videos\\New folder\\Dataset'
val_dir = 'C:\\Users\\sonic\\Videos\\New folder\\val2\\'
# Define the train and test directories
train_dir = 'C:\\Users\\sonic\\Videos\\New folder\\train2\\'
test_dir = 'C:\\Users\\sonic\\Videos\\New folder\\test2\\'


class_names = sorted(os.listdir(data_dir))
num_classes = len(class_names)
print("Class Names: \n", class_names)
print("Number of Classes:", num_classes)

class_sizes = []
for name in class_names:
    class_size = len(os.listdir(data_dir + "/" + name))
    class_sizes.append(class_size)

print("Class Distribution:\n", class_sizes)

data = go.Pie(labels=class_names, values=class_sizes)

# Define the layout
layout = go.Layout(title={"text": "Class Distribution", "x": 0.5})

# Create the figure
fig = go.Figure(data=data, layout=layout)

# Display the figure
fig.show()

# Plot a bar graph of the number of images in each class

# Set the size of the figure
plt.figure(figsize=(10,5))

# Plot a bar chart using the class names as the x-axis and class sizes as the y-axis
sns.barplot(x=class_names, y=class_sizes)

# Add a grid to the plot
plt.grid()

# Add a horizontal line to show the mean number of images across all classes
plt.axhline(np.mean(class_sizes), color='black', linestyle=':', label="Average number of images per class")

# Add a legend to the plot
plt.legend()

# Show the plot
plt.show()





# Create the train, validation, and test directories if they don't exist
os.makedirs(train_dir, exist_ok=True)
os.makedirs(val_dir, exist_ok=True)
os.makedirs(test_dir, exist_ok=True)

# Get a list of all the animal type directories
animal_dirs = [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))]

# Split the data into training, validation, and testing sets
train_data = []
val_data = []
test_data = []
for animal_dir in animal_dirs:
    animal_dir_path = os.path.join(data_dir, animal_dir)
    image_files = [os.path.join(animal_dir_path, f) for f in os.listdir(animal_dir_path) if f.endswith('.jpeg')]
    random.shuffle(image_files)
    train_files, test_files = train_test_split(image_files, test_size=0.2)
    train_files, val_files = train_test_split(train_files, test_size=0.2)
    train_data.extend([(f, animal_dir) for f in train_files])
    val_data.extend([(f, animal_dir) for f in val_files])
    test_data.extend([(f, animal_dir) for f in test_files])

# Copy the training data to the train directory
for file, animal in train_data:
    animal_dir_path = os.path.join(train_dir, animal)
    os.makedirs(animal_dir_path, exist_ok=True)
    shutil.copy(file, animal_dir_path)

# Copy the validation data to the val directory
for file, animal in val_data:
    animal_dir_path = os.path.join(val_dir, animal)
    os.makedirs(animal_dir_path, exist_ok=True)
    shutil.copy(file, animal_dir_path)

# Copy the testing data to the test directory
for file, animal in test_data:
    animal_dir_path = os.path.join(test_dir, animal)
    os.makedirs(animal_dir_path, exist_ok=True)
    shutil.copy(file, animal_dir_path)



#mlhaaash lazma

# train_datagen = ImageDataGenerator(rescale=1./255, 
# horizontal_flip=True, 
# vertical_flip=True, 
# rotation_range=20, 
# validation_split=0.2)

# val_datagen = ImageDataGenerator(data_generator.flow_from_directory(
# sampled_data_path, 
# target_size=(256,256), 
# class_mode='binary', 
# batch_size=32, 
# shuffle=True, 
# subset='validation')




data_generator = ImageDataGenerator(
rescale=1./255, 
horizontal_flip=True, 
vertical_flip=True, 
rotation_range=20, 
validation_split=0.2)

test_generator = ImageDataGenerator(
rescale=1./255, 
horizontal_flip=True, 
vertical_flip=True, 
rotation_range=20)

# Load the data
#convert , labels
train_dataGen = data_generator.flow_from_directory(train_dir,
                                               target_size=(80,80), 
                                                        class_mode='categorical', 
                                                        batch_size=32, 
                                                        color_mode='rgb',
                                                        seed=42,

                                                        shuffle=True )
                                                        #subset='training')

val_dataGen = data_generator.flow_from_directory(val_dir,
                                          target_size=(80,80), 
                                                   class_mode='categorical', 
                                                   batch_size=32,
                                                   color_mode='rgb',
                                                   seed=42,

                                                   shuffle=True )
                                                   #subset='validation')

test_dataGen = test_generator.flow_from_directory(test_dir,
                                            target_size=(80,80), 
                                                     class_mode='categorical', 
                                                     batch_size=32, 
                                                     color_mode='rgb',
                                                     seed=42,

                                                     shuffle=True)
                                                     #subset='testing')










# # Load the data
# train_data = train_datagen.flow_from_directory(train_dir,
#                                                target_size=(64, 64),
                                              
#                                                class_mode='categorical')

# test_data = test_datagen.flow_from_directory(test_dir,
#                                              target_size=(64, 64),
                                             
#                                              class_mode='categorical')
# val_data = test_datagen.flow_from_directory(val_dir,
#                                              target_size=(64, 64),
                                             
#                                              class_mode='categorical')

# Build the model


# Define the model architecture
model = tf.keras.models.Sequential([
tf.keras.layers.Conv2D(64, (4, 4), activation='relu', input_shape=(80, 80, 3), padding='valid'),
tf.keras.layers.MaxPooling2D(3, 3),
tf.keras.layers.Conv2D(128, (4, 4), activation='relu', padding='valid'),
tf.keras.layers.MaxPooling2D(3, 3),
tf.keras.layers.Conv2D(256, (4, 4), activation='relu', padding='valid'),

tf.keras.layers.MaxPooling2D(3, 3),
tf.keras.layers.Flatten(),
tf.keras.layers.Dense(120, activation='relu', kernel_regularizer=regularizers.l1(0.01)),
#tf.keras.layers.Dropout(0.5),
tf.keras.layers.Dense(120, activation='relu'),



tf.keras.layers.Dense(10, activation='softmax')
])


# Compile the model
model.compile(loss='categorical_crossentropy',
              optimizer='RMSprop',
              metrics=['accuracy'])

# Train the model
history=model.fit(train_dataGen,batch_size= 165, epochs=350, validation_data=val_dataGen)








test_loss,test_acc=model.evaluate(test_dataGen)
print("Test Loss:", test_loss)
print("Test Accuracy: {:.2f}%".format(test_acc* 100))
    




new_image = image.load_img('C:\\Users\\sonic\\Videos\\2.jpeg', target_size=(80, 80))

# Convert the image to a numpy array
new_image = image.img_to_array(new_image)

# Reshape the array to match the input shape of the model
new_image = np.expand_dims(new_image, axis=0)

# Scale the pixel values to the range [0, 1]
new_image /= 255.0

# Use the model to make a prediction
prediction = model.predict(new_image)

# Get the predicted class label
predicted_class = np.argmax(prediction)

# Print the predicted class label
class_labels = list(train_dataGen.class_indices.keys())
predicted_label = class_labels[predicted_class]
print(predicted_label)





accuracy = history.history['accuracy']
val_accuracy = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(len(accuracy))

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

ax1.plot(epochs, accuracy, 'b', label='Training accuracy')
ax1.plot(epochs, val_accuracy, 'r', label='Validation accuracy')
ax1.set_title('Training and validation accuracy')
ax1.set_xlabel('Epochs')
ax1.set_ylabel('Accuracy')
ax1.legend()

ax2.plot(epochs, loss, 'b', label='Training loss')
ax2.plot(epochs, val_loss, 'r', label='Validation loss')
ax2.set_title('Training and validation loss')
ax2.set_xlabel('Epochs')
ax2.set_ylabel('Loss')
ax2.legend()

fig.suptitle('Training and validation metrics', fontsize=16)
plt.show()
