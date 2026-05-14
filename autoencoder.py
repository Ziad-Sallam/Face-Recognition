
import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.optimizers import Adam

# =========================================================
# HYPERPARAMETERS
# =========================================================

DATASET_PATH = "data"

IMAGE_HEIGHT = 112
IMAGE_WIDTH = 92

# Autoencoder architecture
ENCODER_LAYER_1 = 1024
ENCODER_LAYER_2 = 512
LATENT_DIM = 128

# Training
LEARNING_RATE = 0.001
BATCH_SIZE = 16
EPOCHS = 50
TEST_SIZE = 0.2

# Output files
OUTPUT_FEATURES_CSV = "encoded_faces.csv"
SAVE_MODELS = True

# =========================================================
# LOAD DATASET
# =========================================================

images = []
labels = []

for subject in range(1, 41):

    subject_path = os.path.join(DATASET_PATH, f"s{subject}")

    for image_name in os.listdir(subject_path):

        image_path = os.path.join(subject_path, image_name)

        # Read grayscale image
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

        # Resize if needed
        img = cv2.resize(img, (IMAGE_WIDTH, IMAGE_HEIGHT))

        # Normalize
        img = img / 255.0

        images.append(img)
        labels.append(subject)

images = np.array(images)
labels = np.array(labels)

print("Dataset loaded successfully")
print("Images shape:", images.shape)
print("Labels shape:", labels.shape)

# =========================================================
# PREPROCESSING
# =========================================================

# Flatten images
images_flat = images.reshape(images.shape[0], IMAGE_HEIGHT * IMAGE_WIDTH)

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    images_flat,
    labels,
    test_size=TEST_SIZE,
    random_state=42,
    stratify=labels
)

print("Train set:", X_train.shape)
print("Test set:", X_test.shape)

# =========================================================
# BUILD AUTOENCODER
# =========================================================

INPUT_DIM = IMAGE_HEIGHT * IMAGE_WIDTH

# Encoder
input_img = Input(shape=(INPUT_DIM,))

encoded = Dense(ENCODER_LAYER_1, activation='relu')(input_img)
encoded = Dense(ENCODER_LAYER_2, activation='relu')(encoded)
latent = Dense(LATENT_DIM, activation='relu')(encoded)

# Decoder
decoded = Dense(ENCODER_LAYER_2, activation='relu')(latent)
decoded = Dense(ENCODER_LAYER_1, activation='relu')(decoded)
decoded = Dense(INPUT_DIM, activation='sigmoid')(decoded)

# Full autoencoder
autoencoder = Model(input_img, decoded)

# Encoder only
encoder = Model(input_img, latent)

# Compile
optimizer = Adam(learning_rate=LEARNING_RATE)

autoencoder.compile(
    optimizer=optimizer,
    loss='mse'
)

# =========================================================
# MODEL SUMMARY
# =========================================================

print("\nAutoencoder Summary")
autoencoder.summary()

# =========================================================
# TRAIN MODEL
# =========================================================

history = autoencoder.fit(
    X_train,
    X_train,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    shuffle=True,
    validation_data=(X_test, X_test)
)

# =========================================================
# EXTRACT FEATURES
# =========================================================

all_features = encoder.predict(images_flat)

print("Encoded feature shape:", all_features.shape)

# =========================================================
# CREATE NEW DATASET
# =========================================================

# Create dataframe
feature_columns = []

for i in range(LATENT_DIM):
    feature_columns.append(f"feature_{i+1}")

features_df = pd.DataFrame(all_features, columns=feature_columns)

# Add labels
features_df['label'] = labels

# Save dataset
features_df.to_csv(OUTPUT_FEATURES_CSV, index=False)

print(f"\nNew encoded dataset saved as: {OUTPUT_FEATURES_CSV}")
print(features_df.head())

# =========================================================
# CREATE OUTPUT DIRECTORIES
# =========================================================

os.makedirs("results", exist_ok=True)
os.makedirs("results/reconstructed", exist_ok=True)
os.makedirs("results/graphs", exist_ok=True)

# =========================================================
# VISUALIZE RECONSTRUCTION
# =========================================================

decoded_images = autoencoder.predict(X_test)

n = 5

plt.figure(figsize=(12, 5))

for i in range(n):

    # Original image
    ax = plt.subplot(2, n, i + 1)
    plt.imshow(X_test[i].reshape(IMAGE_HEIGHT, IMAGE_WIDTH), cmap='gray')
    plt.title("Original")
    plt.axis('off')

    # Reconstructed image
    ax = plt.subplot(2, n, i + 1 + n)
    plt.imshow(decoded_images[i].reshape(IMAGE_HEIGHT, IMAGE_WIDTH), cmap='gray')
    plt.title("Reconstructed")
    plt.axis('off')

plt.tight_layout()

# Save reconstruction comparison
plt.savefig("results/reconstructed/reconstruction_results.png")

plt.show()

# =========================================================
# SAVE MODELS
# =========================================================

if SAVE_MODELS:

    autoencoder.save("face_autoencoder.h5")
    encoder.save("face_encoder.h5")

    print("\nModels saved successfully")

# =========================================================
# SAVE RECONSTRUCTED IMAGES
# =========================================================

for i in range(min(10, len(decoded_images))):

    reconstructed = decoded_images[i].reshape(IMAGE_HEIGHT, IMAGE_WIDTH)
    reconstructed = (reconstructed * 255).astype(np.uint8)

    original = X_test[i].reshape(IMAGE_HEIGHT, IMAGE_WIDTH)
    original = (original * 255).astype(np.uint8)

    cv2.imwrite(
        f"results/reconstructed/original_{i+1}.png",
        original
    )

    cv2.imwrite(
        f"results/reconstructed/reconstructed_{i+1}.png",
        reconstructed
    )

print("Reconstructed images saved successfully")

# =========================================================
# TRAINING LOSS PLOT
# =========================================================

plt.figure(figsize=(8, 5))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Autoencoder Training Loss')
plt.legend()
plt.grid(True)

# Save training graph
plt.savefig("results/graphs/training_loss.png")

plt.show()

print("Training graph saved successfully")
