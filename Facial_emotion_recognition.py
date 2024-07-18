!pip install scikit-Plot
!pip install keras-applications
!pip install seaborn
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
import scikitplot
import numpy as np
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import optimizers
from tensorflow.keras.layers import BatchNormalization

import seaborn as sns
df = pd.read_csv('/content/drive/MyDrive/fer2013.csv')
print(df.shape)
print(df.head())
print(df.emotion.unique())

emotion_label_text = {0:'anger', 1:'disgust', 2:'fear', 3:'happiness', 4: 'sadness', 5: 'surprise', 6: 'neutral'}
print(df.emotion.value_counts())


emotion_frequency = df['emotion'].value_counts()

# Sort the frequency by emotion label index
emotion_frequency = emotion_frequency.sort_index()

# Plot the bar graph
plt.figure(figsize=(10, 6))
plt.bar(emotion_frequency.index, emotion_frequency.values, color='skyblue')
plt.title('Frequency of Each Emotion')
plt.xlabel('Emotion Label')
plt.ylabel('Frequency')
plt.xticks(emotion_frequency.index, [emotion_label_text[label] for label in emotion_frequency.index])
plt.grid(axis='y')
plt.show()

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Assuming df is your DataFrame containing pixel values for emotions
df = pd.read_csv('/content/drive/MyDrive/fer2013.csv')

# Mapping of emotion labels to text representations
emotion_label_text = {0: 'anger', 1: 'disgust', 2: 'fear', 3: 'happiness', 4: 'sadness', 5: 'surprise', 6: 'neutral'}

# Filter the DataFrame for happiness, sadness, and neutral emotions
happiness_images = df[df.emotion == 3].pixels.values[:5]
sadness_images = df[df.emotion == 4].pixels.values[:5]
neutral_images = df[df.emotion == 6].pixels.values[:5]

# Function to plot images
def plot_images(images, title):
    fig, axes = plt.subplots(1, len(images), figsize=(15, 3))
    for ax, image in zip(axes, images):
        pixels = np.array(image.split(' ')).astype('float32').reshape(48, 48)
        ax.imshow(pixels, cmap='gray')
        ax.axis('off')
    fig.suptitle(title, fontsize=16)
    plt.show()

# Plot images for happiness, sadness, and neutral emotions
plot_images(happiness_images, 'Happiness')
plot_images(sadness_images, 'Sadness')
plot_images(neutral_images, 'Neutral')



desired_labels = [3,4,6] #Skipping emotion label 1 as there are very little number of data points. Skipping all labels other than happiness/neutral/sadness(83% accuracy), Considering all except 1 yeilded in 63/64%.
df = df[df.emotion.isin(desired_labels)]
df.shape

#Getting the data ready for processing
img_array = df.pixels.apply(lambda x: np.array(x.split(' ')).reshape(48, 48, 1).astype('float32'))
img_array = np.stack(img_array, axis=0)

le = LabelEncoder()
img_labels = le.fit_transform(df.emotion)
img_labels = to_categorical(img_labels)
img_labels.shape

le_name_mapping = dict(zip(le.classes_, le.transform(le.classes_)))
print(le_name_mapping)

X_train, X_valid, y_train, y_valid = train_test_split(img_array, img_labels,
                                                    shuffle=True, stratify=img_labels,
                                                    test_size=0.2, random_state=42)
X_train.shape, X_valid.shape, y_train.shape, y_valid.shape

img_width = X_train.shape[1]
img_height = X_train.shape[2]
img_depth = X_train.shape[3]
num_classes = y_train.shape[1]

# Normalizing results, as neural networks are very sensitive to unnormalized data.
X_train = X_train / 255.
X_valid = X_valid / 255.


def build_net(optim):
    net = Sequential(name='Deep-CNN')
    net.add(Conv2D(filters=64, kernel_size=(5,5), input_shape=(img_width, img_height, img_depth), activation='elu', padding='same', kernel_initializer='he_normal', name='conv2d_a'))
    net.add(BatchNormalization(name='batchnorm_a'))
    net.add(Conv2D(filters=64, kernel_size=(5,5), activation='elu', padding='same', kernel_initializer='he_normal', name='conv2d_b'))
    net.add(BatchNormalization(name='batchnorm_b'))
    net.add(MaxPooling2D(pool_size=(2,2), name='maxpool_a'))
    net.add(Dropout(0.4, name='dropout_a'))
    net.add(Conv2D(filters=128, kernel_size=(3,3), activation='elu', padding='same', kernel_initializer='he_normal', name='conv2d_c'))
    net.add(BatchNormalization(name='batchnorm_c'))
    net.add(Conv2D(filters=128, kernel_size=(3,3), activation='elu', padding='same', kernel_initializer='he_normal', name='conv2d_d'))
    net.add(BatchNormalization(name='batchnorm_d'))
    net.add(MaxPooling2D(pool_size=(2,2), name='maxpool_b'))
    net.add(Dropout(0.4, name='dropout_b'))
    net.add(Conv2D(filters=256, kernel_size=(3,3), activation='elu', padding='same', kernel_initializer='he_normal', name='conv2d_e'))
    net.add(BatchNormalization(name='batchnorm_e'))
    net.add(Conv2D(filters=256, kernel_size=(3,3), activation='elu', padding='same', kernel_initializer='he_normal', name='conv2d_f'))
    net.add(BatchNormalization(name='batchnorm_f'))
    net.add(MaxPooling2D(pool_size=(2,2), name='maxpool_c'))
    net.add(Dropout(0.5, name='dropout_c'))
    net.add(Flatten(name='flatten'))
    net.add(Dense(128, activation='elu', kernel_initializer='he_normal', name='dense_a'))
    net.add(BatchNormalization(name='batchnorm_g'))
    net.add(Dropout(0.6, name='dropout_d'))
    net.add(Dense(num_classes, activation='softmax', name='out_layer'))
    net.compile(loss='categorical_crossentropy', optimizer=optim, metrics=['accuracy'])
    net.summary()
    return net




early_stopping = EarlyStopping(monitor='val_accuracy', min_delta=0.00005, patience=11, verbose=1, restore_best_weights=True,)

lr_scheduler = ReduceLROnPlateau(monitor='val_accuracy', factor=0.5, patience=7, min_lr=1e-7, verbose=1,)

callbacks = [early_stopping,lr_scheduler,]

# As the data in hand is less as compared to the task so ImageDataGenerator is good to go.
train_datagen = ImageDataGenerator(
    rotation_range=15,
    width_shift_range=0.15,
    height_shift_range=0.15,
    shear_range=0.15,
    zoom_range=0.15,
    horizontal_flip=True,
)
train_datagen.fit(X_train)

batch_size = 32 #batch size of 32 performs the best.
epochs = 100
optims = [
    optimizers.Nadam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-07, name='Nadam'),
    optimizers.Adam(0.001),
]

model = build_net(optims[1])
model.build((None, img_width, img_height, img_depth))
history = model.fit_generator(
    train_datagen.flow(X_train, y_train, batch_size=batch_size),
    validation_data=(X_valid, y_valid),
    steps_per_epoch=len(X_train) / batch_size,
    epochs=epochs,
    callbacks=callbacks,
    use_multiprocessing=True
)

model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)

model.save('model.keras')

sns.set()
fig = plt.figure(figsize=(12, 4))

ax1 = plt.subplot(1, 2, 1)
sns.lineplot(x=history.epoch, y=history.history['accuracy'], label='train', ax=ax1)
sns.lineplot(x=history.epoch, y=history.history['val_accuracy'], label='valid', ax=ax1)
plt.title('Accuracy')

ax2 = plt.subplot(1, 2, 2)
sns.lineplot(x=history.epoch, y=history.history['loss'], label='train', ax=ax2)
sns.lineplot(x=history.epoch, y=history.history['val_loss'], label='valid', ax=ax2)
plt.title('Loss')

plt.tight_layout()
plt.savefig('epoch_history_dcnn.png')
plt.show()

yhat_valid = np.argmax(model.predict(X_valid), axis=1)
scikitplot.metrics.plot_confusion_matrix(np.argmax(y_valid, axis=1), yhat_valid, figsize=(7, 7))
plt.savefig("confusion_matrix_dcnn.png")

print(f'total wrong validation predictions: {np.sum(np.argmax(y_valid, axis=1) != yhat_valid)}\n\n')
print(classification_report(np.argmax(y_valid, axis=1), yhat_valid))

import numpy as np
from tensorflow.keras.layers import Input, GlobalAveragePooling2D, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.optimizers import Adam

# Preprocess the input data to have three channels
X_train_rgb = np.repeat(X_train, 3, axis=-1)
X_valid_rgb = np.repeat(X_valid, 3, axis=-1)

# Load the pre-trained ResNet50 model without the top classification layer
base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(48, 48, 3))

# Freeze the convolutional layers in the base model
for layer in base_model.layers:
    layer.trainable = False

# Add custom classification layers on top of the base model
x = GlobalAveragePooling2D()(base_model.output)
x = Dense(128, activation='relu')(x)
predictions = Dense(3, activation='softmax')(x)

# Create the final model
model = Model(inputs=base_model.input, outputs=predictions)

# Compile the model
model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(
    X_train_rgb, y_train,
    validation_data=(X_valid_rgb, y_valid),
    epochs=30,
    batch_size=32
)


import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications import VGG16
# Convert grayscale images to RGB
X_train_rgb = np.repeat(X_train, 3, -1)
X_valid_rgb = np.repeat(X_valid, 3, -1)



# Load pre-trained VGG16 model without top layers
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(img_width, img_height, 3))

# Freeze the layers in the base model
for layer in base_model.layers:
    layer.trainable = False

# Define custom top layers
top_model = Sequential()
top_model.add(Flatten(input_shape=base_model.output_shape[1:]))
top_model.add(Dense(256, activation='relu'))
top_model.add(Dropout(0.5))
top_model.add(Dense(num_classes, activation='softmax'))  # Assuming num_classes for emotions

# Combine base model and top layers
combined_model = Sequential()
combined_model.add(base_model)
combined_model.add(top_model)

# Compile the model
combined_model.compile(optimizer=Adam(lr=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
history = combined_model.fit(X_train_rgb, y_train, epochs=20, batch_size=32, validation_data=(X_valid_rgb, y_valid))

# Evaluate the model
loss, accuracy = combined_model.evaluate(X_valid_rgb, y_valid)
print("Validation Accuracy: {:.2f}%".format(accuracy * 100))


