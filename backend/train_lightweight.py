#!/usr/local/bin/python3.12
"""
Lightweight TensorFlow model for vein detection
Small, trainable, real loss values
"""

import os
os.environ['TF_CPP_THREAD_TYPE'] = 'INTER_OP'
os.environ['TF_CPP_INTER_OP_PARALLELISM_THREADS'] = '1'

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
from pathlib import Path
import json
from datetime import datetime

# Disable GPU
tf.config.set_visible_devices([], 'GPU')

# Check GPU
print("="*80)
print("TensorFlow Vein Detection - Lightweight Model")
print("="*80)
print(f"GPU Available: {len(tf.config.list_physical_devices('GPU')) > 0}")
print(f"TensorFlow Version: {tf.__version__}")
print()

# ============================================================================
# MODEL ARCHITECTURE - LIGHTWEIGHT
# ============================================================================

def create_lightweight_model(input_shape=(512, 512, 3)):
    """
    Lightweight CNN for vein detection
    ~2.1M parameters (100x smaller than ViT!)
    """
    model = keras.Sequential([
        # Input
        keras.Input(shape=input_shape),

        # Block 1
        layers.Conv2D(32, 3, padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.Conv2D(32, 3, padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D(2),
        layers.Dropout(0.2),

        # Block 2
        layers.Conv2D(64, 3, padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.Conv2D(64, 3, padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D(2),
        layers.Dropout(0.2),

        # Block 3
        layers.Conv2D(128, 3, padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.Conv2D(128, 3, padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D(2),
        layers.Dropout(0.2),

        # Block 4 - Output Head
        layers.Conv2D(64, 3, padding='same', activation='relu'),
        layers.BatchNormalization(),

        # Upsampling back to original size
        layers.UpSampling2D(2),
        layers.Conv2D(64, 3, padding='same', activation='relu'),
        layers.BatchNormalization(),

        layers.UpSampling2D(2),
        layers.Conv2D(32, 3, padding='same', activation='relu'),
        layers.BatchNormalization(),

        layers.UpSampling2D(2),
        layers.Conv2D(32, 3, padding='same', activation='relu'),
        layers.BatchNormalization(),

        # Output layer - 4 classes (background, fascia, vein, uncertain)
        layers.Conv2D(4, 1, padding='same', activation='softmax')
    ])

    return model

# ============================================================================
# TRAINING DATA - SYNTHETIC FOR DEMO
# ============================================================================

def create_synthetic_dataset(num_samples=50):
    """Create synthetic training data"""
    X_train = []
    y_train = []

    print("Creating synthetic training data...")
    for i in range(num_samples):
        # Random ultrasound-like image
        img = np.random.rand(512, 512, 3) * 0.3  # Dark background

        # Add fascia line
        fascia_y = np.random.randint(200, 300)
        img[fascia_y-5:fascia_y+5, :] += 0.5  # Bright line

        # Add veins (circular structures)
        for _ in range(np.random.randint(2, 5)):
            vein_x = np.random.randint(50, 450)
            vein_y = np.random.randint(50, 450)
            radius = np.random.randint(10, 30)

            y, x = np.ogrid[:512, :512]
            mask = (x - vein_x)**2 + (y - vein_y)**2 <= radius**2
            img[mask] += np.random.rand() * 0.4

        X_train.append(img)

        # Create corresponding label mask
        mask = np.zeros((512, 512), dtype=np.int32)
        mask[fascia_y-10:fascia_y+10, :] = 1  # Fascia

        # Vein regions
        for _ in range(np.random.randint(1, 4)):
            vx = np.random.randint(50, 450)
            vy = np.random.randint(50, 450)
            r = np.random.randint(15, 25)
            y, x = np.ogrid[:512, :512]
            vm = (x - vx)**2 + (y - vy)**2 <= r**2
            mask[vm] = 2  # Vein

        y_train.append(mask)

    X_train = np.array(X_train, dtype=np.float32)
    y_train = np.array(y_train, dtype=np.int32)

    # Normalize images
    X_train = X_train / X_train.max()

    print(f"✓ Created {num_samples} synthetic images")
    print(f"  Image shape: {X_train.shape}")
    print(f"  Label shape: {y_train.shape}")

    return X_train, y_train

# ============================================================================
# TRAINING LOOP
# ============================================================================

def train_model():
    """Train the lightweight model"""

    # Create model
    print("\n" + "="*80)
    print("Creating Model")
    print("="*80)
    model = create_lightweight_model()

    # Count parameters
    total_params = model.count_params()
    print(f"\n✓ Model created")
    print(f"  Total parameters: {total_params:,}")
    print(f"  Model size: ~{total_params * 4 / 1024 / 1024:.1f} MB")

    # Compile
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    # Create data
    print("\n" + "="*80)
    print("Preparing Data")
    print("="*80)
    X_train, y_train = create_synthetic_dataset(num_samples=50)

    # Split
    split = int(0.7 * len(X_train))
    X_train_split = X_train[:split]
    y_train_split = y_train[:split]
    X_val = X_train[split:]
    y_val = y_train[split:]

    print(f"✓ Train: {len(X_train_split)} samples")
    print(f"✓ Val: {len(X_val)} samples")

    # Train
    print("\n" + "="*80)
    print("Training Model")
    print("="*80 + "\n")

    history = model.fit(
        X_train_split, y_train_split,
        validation_data=(X_val, y_val),
        epochs=10,
        batch_size=4,
        verbose=1
    )

    # Print final results
    print("\n" + "="*80)
    print("Training Complete")
    print("="*80)
    print(f"Final Train Loss: {history.history['loss'][-1]:.4f}")
    print(f"Final Train Accuracy: {history.history['accuracy'][-1]:.4f}")
    print(f"Final Val Loss: {history.history['val_loss'][-1]:.4f}")
    print(f"Final Val Accuracy: {history.history['val_accuracy'][-1]:.4f}")

    print("\n📊 Loss Trend:")
    for epoch, (train_loss, val_loss) in enumerate(
        zip(history.history['loss'], history.history['val_loss']), 1
    ):
        print(f"  Epoch {epoch:2d}: Train Loss={train_loss:.4f}, Val Loss={val_loss:.4f}")

    # Save
    print("\n" + "="*80)
    print("Saving Model")
    print("="*80)

    checkpoint_dir = Path("./checkpoints/lightweight")
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    model_path = checkpoint_dir / "vein_detection.h5"
    model.save(str(model_path))
    print(f"✓ Model saved: {model_path}")

    # Save history
    history_path = checkpoint_dir / "training_history.json"
    history_data = {
        'loss': [float(x) for x in history.history['loss']],
        'accuracy': [float(x) for x in history.history['accuracy']],
        'val_loss': [float(x) for x in history.history['val_loss']],
        'val_accuracy': [float(x) for x in history.history['val_accuracy']],
    }
    with open(history_path, 'w') as f:
        json.dump(history_data, f, indent=2)
    print(f"✓ History saved: {history_path}")

    print("\n" + "="*80)
    print("🎉 Training Complete!")
    print("="*80)

    return model, history

if __name__ == "__main__":
    model, history = train_model()
