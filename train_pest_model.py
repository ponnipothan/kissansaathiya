import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2

# dataset path
data_dir = "dataset"

# preprocessing
datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2,

    # augmentation
    rotation_range=25,
    zoom_range=0.25,
    horizontal_flip=True
)

train = datagen.flow_from_directory(
    data_dir,
    target_size=(224, 224),
    batch_size=16,
    class_mode='categorical',
    subset='training'
)

val = datagen.flow_from_directory(
    data_dir,
    target_size=(224, 224),
    batch_size=16,
    class_mode='categorical',
    subset='validation'
)

print("Classes:", train.class_indices)

# 🔥 PRETRAINED MODEL (KEY CHANGE)
base_model = MobileNetV2(
    input_shape=(224, 224, 3),
    include_top=False,
    weights='imagenet'
)

# freeze base layers
base_model.trainable = False

# custom layers
model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.3),   # helps reduce overfitting
    layers.Dense(train.num_classes, activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

print("Training started...\n")

model.fit(
    train,
    validation_data=val,
    epochs=10  # 🔥 increased
)

# save model
model.save("plant_disease_model.h5")

print("\n✅ Model saved as plant_disease_model.h5")