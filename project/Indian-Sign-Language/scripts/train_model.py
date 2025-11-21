import os
import numpy as np
import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
from keras.applications import MobileNetV2
from keras.layers import Dense, Dropout, GlobalAveragePooling2D
from keras.models import Model
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, EarlyStopping
from sklearn.preprocessing import LabelEncoder
import joblib

# ---------------- GPU Configuration ----------------
# gpus = tf.config.experimental.list_physical_devices('GPU')
# if gpus:
#     try:
#         for g in gpus:
#             tf.config.experimental.set_memory_growth(g, True)
#         print("✅ GPU detected and enabled:", gpus)
#     except Exception as e:
#         print("⚠️ Could not set memory growth:", e)
# else:
#     print("⚠️ No GPU detected. Training will run on CPU.")

# ---------------- Paths ----------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(BASE_DIR, "dataset_images")
OUTPUT_DIR = os.path.join(BASE_DIR, "dataset")
MODEL_PATH = os.path.join(OUTPUT_DIR, "sign_language_mobilenet.h5")
SAVED_MODEL_DIR = os.path.join(OUTPUT_DIR, "sign_language_mobilenet_saved")
ENCODER_PATH = os.path.join(OUTPUT_DIR, "label_encoder_mobilenet.pkl")
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ---------------- Parameters ----------------
IMG_SIZE = 128
BATCH_SIZE = 32
EPOCHS = 20
LR = 1e-4

# ---------------- Verify Dataset ----------------
if not os.path.exists(DATASET_DIR):
    raise FileNotFoundError(f"❌ Dataset folder not found: {DATASET_DIR}")

classes = sorted([d for d in os.listdir(DATASET_DIR)
                  if os.path.isdir(os.path.join(DATASET_DIR, d))])
print(f"✅ Found {len(classes)} classes: {classes}")

# ---------------- Load Dataset ----------------
datagen = ImageDataGenerator(
    rescale=1.0 / 255.0,
    validation_split=0.2,
    rotation_range=15,
    zoom_range=0.2,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    horizontal_flip=True
)

train_gen = datagen.flow_from_directory(
    DATASET_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode="sparse",
    subset="training"
)

val_gen = datagen.flow_from_directory(
    DATASET_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode="sparse",
    subset="validation"
)

# ---------------- Label Encoder ----------------
label_encoder = LabelEncoder()
label_encoder.fit(list(train_gen.class_indices.keys()))
joblib.dump(label_encoder, ENCODER_PATH)
print(f"✅ Saved label encoder to {ENCODER_PATH}")

# ---------------- Build Model ----------------
base_model = MobileNetV2(weights="imagenet", include_top=False,
                         input_shape=(IMG_SIZE, IMG_SIZE, 3))
base_model.trainable = False  # freeze base model

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(256, activation="relu")(x)
x = Dropout(0.4)(x)
outputs = Dense(train_gen.num_classes, activation="softmax")(x)

model = Model(inputs=base_model.input, outputs=outputs)
model.compile(optimizer=Adam(learning_rate=LR),
              loss="sparse_categorical_crossentropy", metrics=["accuracy"])

model.summary()

# ---------------- Callbacks ----------------
checkpoint_cb = ModelCheckpoint(
    MODEL_PATH, monitor="val_accuracy", save_best_only=True, verbose=1)
early_stop_cb = EarlyStopping(monitor="val_loss", patience=5,
                              restore_best_weights=True, verbose=1)

# ---------------- Train (Transfer Learning) ----------------
history = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=EPOCHS,
    callbacks=[checkpoint_cb, early_stop_cb]
)

# ---------------- Fine-Tuning ----------------
print("\n🔧 Fine-tuning the top 40 layers...")
base_model.trainable = True
for layer in base_model.layers[:-40]:
    layer.trainable = False

model.compile(optimizer=Adam(learning_rate=1e-5),
              loss="sparse_categorical_crossentropy", metrics=["accuracy"])

history_fine = model.fit(
    train_gen, validation_data=val_gen, epochs=5, callbacks=[checkpoint_cb])

# ---------------- Save Model ----------------
try:
    model.save(SAVED_MODEL_DIR)
    print(f"✅ Saved TensorFlow SavedModel to {SAVED_MODEL_DIR}")
except Exception as e:
    print(f"⚠️ SavedModel save failed: {e}")

try:
    model.save(MODEL_PATH)
    print(f"✅ Saved H5 model to {MODEL_PATH}")
except Exception as e:
    print(f"⚠️ Could not save H5 model: {e}")

print("🎯 Training complete!")
