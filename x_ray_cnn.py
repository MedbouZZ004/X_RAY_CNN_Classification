# -------------------- 📌 INSTALL & IMPORT LIBRARIES --------------------
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout, BatchNormalization
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

print("TensorFlow:", tf.__version__)

# -------------------- 📌 DATA PATHS --------------------
# Use the script directory so paths work on local Windows/VS environments
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
train_path = os.path.join(BASE_DIR, 'train')
test_path = os.path.join(BASE_DIR, 'test')

print(f"Using train_path: {train_path}")
print(f"Using test_path: {test_path}")

# Fail fast with a clear message if directories are missing
if not os.path.isdir(train_path):
    raise FileNotFoundError(f"Train directory not found: {train_path}")
if not os.path.isdir(test_path):
    raise FileNotFoundError(f"Test directory not found: {test_path}")

# -------------------- 📌 DATA AUGMENTATION --------------------
# Using preprocess_input from MobileNetV2 for proper scaling
train_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    rotation_range=20,
    # Modification: Increased shear range for better perspective variation
    shear_range=0.15,
    zoom_range=0.1,
    horizontal_flip=True
)

test_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)

# -------------------- 📌 CREATE GENERATORS --------------------
IMG_SIZE = (224, 224)
BATCH_SIZE = 32

train_ds = train_datagen.flow_from_directory(
    train_path,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='binary'  # Correct for sigmoid output
)

test_ds = test_datagen.flow_from_directory(
    test_path,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='binary',
    shuffle=False  # Essential for evaluation metrics
)

# Extract class names from the generator
CLASS_NAMES = list(train_ds.class_indices.keys())
print(f"Detected Class Names: {CLASS_NAMES}")

# -------------------- 📌 IMPORT BASE MODEL --------------------
base_model = MobileNetV2(include_top=False, weights='imagenet', input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3))

# Freeze convolutional base at first (for stability)
base_model.trainable = False

# -------------------- 📌 CUSTOM CLASSIFIER --------------------
x = base_model.output
x = GlobalAveragePooling2D()(x)
# Modification: Added an extra Dense layer and increased neurons for better feature learning
x = Dense(512, activation='relu')(x)
x = BatchNormalization()(x)
x = Dropout(0.4)(x)  # Slightly increased dropout
x = Dense(256, activation='relu')(x)
x = BatchNormalization()(x)
x = Dropout(0.3)(x)
predictions = Dense(1, activation='sigmoid')(x)

model = Model(inputs=base_model.input, outputs=predictions)

# -------------------- 📌 COMPILE & CALLBACKS --------------------
# Using a slightly lower LR for stability during warm-up
WARMUP_LR = 1e-4
FINE_TUNE_LR = 1e-5

model.compile(optimizer=tf.keras.optimizers.Adam(WARMUP_LR),
              loss='binary_crossentropy',
              metrics=['accuracy'])

model.summary()

# ModelCheckpoint saves the best model based on validation accuracy
checkpoint = ModelCheckpoint("best_covid_model.h5",
                             monitor="val_accuracy",
                             save_best_only=True,
                             mode="max",
                             verbose=1)

# -------------------- 📌 STAGE 1: WARM-UP TRAINING --------------------
print("\n" + "=" * 50)
print("STAGE 1: WARM-UP (Training Classifier Layers)")
print("=" * 50)

# Modification: Increased epochs to 15 for better classifier convergence
history = model.fit(train_ds,
                    epochs=15,
                    validation_data=test_ds,
                    callbacks=[checkpoint])

# -------------------- 📌 STAGE 2: FINE-TUNING --------------------
print("\n" + "=" * 50)
print("STAGE 2: FINE-TUNING (Unfreezing deeper layers)")
print("=" * 50)

# Unfreeze deeper layers for fine-tuning
base_model.trainable = True

# Modification: Unfreeze the last ~50 layers (MobileNetV2 has 155 layers total)
# The last 40 layers is a good starting point, keeping the initial layers frozen
for layer in base_model.layers[:-50]:
    layer.trainable = False

# Recompile the model with a much lower LR for fine-tuning
model.compile(optimizer=tf.keras.optimizers.Adam(FINE_TUNE_LR),
              loss='binary_crossentropy',
              metrics=['accuracy'])

model.summary()

# Modification: Added Learning Rate Scheduler for fine-tuning stability
# Reduces LR if validation accuracy doesn't improve for 3 epochs (patience=3)
lr_scheduler = ReduceLROnPlateau(monitor='val_accuracy',
                                 factor=0.2,
                                 patience=3,
                                 min_lr=1e-6,
                                 verbose=1)

# Modification: Increased fine-tuning epochs to 10
history_ft = model.fit(train_ds,
                       epochs=10,
                       validation_data=test_ds,
                       callbacks=[checkpoint, lr_scheduler])

# -------------------- 📌 EVALUATION --------------------
print("\n" + "=" * 50)
print("EVALUATION")
print("=" * 50)

# Load best model (saved from either stage 1 or stage 2)
best_model = tf.keras.models.load_model("best_covid_model.h5")

# Predict test set
# Steps must be explicitly calculated as test_ds.steps is only relevant if shuffle=True
y_pred_steps = int(np.ceil(test_ds.samples / BATCH_SIZE))
y_pred_proba = best_model.predict(test_ds, steps=y_pred_steps)
y_pred = (y_pred_proba > 0.5).astype(int).reshape(-1)

# True labels
y_true = test_ds.classes[:len(y_pred)]  # Truncate true labels to match predictions length if batching was partial

# 📌 Confusion matrix
cm = confusion_matrix(y_true, y_pred)

plt.figure(figsize=(7, 7))  # Increased figure size
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=CLASS_NAMES,  # Using dynamic class names
            yticklabels=CLASS_NAMES)
plt.ylabel("True Label")
plt.xlabel("Predicted Label")
plt.title("Confusion Matrix")
plt.show()

# 📌 Classification Metrics
print("\n🧾 Classification Report:\n")
print(classification_report(y_true, y_pred, target_names=CLASS_NAMES))

