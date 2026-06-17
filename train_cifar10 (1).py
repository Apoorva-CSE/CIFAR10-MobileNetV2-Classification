"""
CIFAR-10 Image Classification with MobileNetV2 Transfer Learning
================================================================
Run: python train_cifar10.py
"""

import tensorflow as tf
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import os

# ─────────────────────────────────────────────
# 1. Hyperparameters
# ─────────────────────────────────────────────
BATCH_SIZE   = 64
EPOCHS       = 10        # increase for better accuracy
INPUT_SHAPE  = (32, 32, 3)
RESIZE_SHAPE = (96, 96, 3)   # MobileNetV2 prefers ≥96 px
LR           = 1e-3
SAVE_PATH    = 'cifar10_mobilenetv2.keras'

class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']

# ─────────────────────────────────────────────
# 2. Load & Prepare Data
# ─────────────────────────────────────────────
print("Loading CIFAR-10 dataset …")
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
print(f"  Train: {x_train.shape}  |  Test: {x_test.shape}")

# ─────────────────────────────────────────────
# 3. Data-Augmentation Pipeline (runs on GPU during training)
# ─────────────────────────────────────────────
data_augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomFlip("horizontal"),
    tf.keras.layers.RandomRotation(0.1),
    tf.keras.layers.RandomZoom(0.1),
    tf.keras.layers.RandomContrast(0.1),
], name="data_augmentation")

# ─────────────────────────────────────────────
# 4. Load Pre-trained Base Model (ImageNet weights, no top)
# ─────────────────────────────────────────────
print("Loading MobileNetV2 backbone …")
base_model = tf.keras.applications.MobileNetV2(
    input_shape=RESIZE_SHAPE,
    include_top=False,
    weights='imagenet'
)
base_model.trainable = False   # freeze backbone → only train head

# ─────────────────────────────────────────────
# 5. Build the Classification Model
# ─────────────────────────────────────────────
inputs  = tf.keras.Input(shape=INPUT_SHAPE)
x       = data_augmentation(inputs)
x       = tf.keras.layers.Resizing(96, 96)(x)
x       = tf.keras.applications.mobilenet_v2.preprocess_input(x)  # scale to [-1,1]
x       = base_model(x, training=False)
x       = tf.keras.layers.GlobalAveragePooling2D()(x)
x       = tf.keras.layers.BatchNormalization()(x)
x       = tf.keras.layers.Dropout(0.3)(x)
outputs = tf.keras.layers.Dense(10, activation='softmax')(x)

model = tf.keras.Model(inputs, outputs)
model.summary()

# ─────────────────────────────────────────────
# 6. Compile
# ─────────────────────────────────────────────
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=LR),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# ─────────────────────────────────────────────
# 7. Callbacks
# ─────────────────────────────────────────────
callbacks = [
    tf.keras.callbacks.EarlyStopping(patience=3, restore_best_weights=True,
                                     monitor='val_accuracy'),
    tf.keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=2,
                                         monitor='val_loss', verbose=1),
    tf.keras.callbacks.ModelCheckpoint(SAVE_PATH, save_best_only=True,
                                       monitor='val_accuracy', verbose=1),
]

# ─────────────────────────────────────────────
# 8. Phase 1 – Train Head Only
# ─────────────────────────────────────────────
print("\n=== Phase 1: Training classification head (backbone frozen) ===")
history = model.fit(
    x_train, y_train,
    epochs=EPOCHS,
    validation_data=(x_test, y_test),
    batch_size=BATCH_SIZE,
    callbacks=callbacks,
)

# ─────────────────────────────────────────────
# 9. Phase 2 – Fine-tune Top Layers of Backbone
# ─────────────────────────────────────────────
print("\n=== Phase 2: Fine-tuning top 30 layers of backbone ===")
base_model.trainable = True
for layer in base_model.layers[:-30]:
    layer.trainable = False

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=LR / 10),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

history_ft = model.fit(
    x_train, y_train,
    epochs=5,
    validation_data=(x_test, y_test),
    batch_size=BATCH_SIZE,
    callbacks=callbacks,
)

# ─────────────────────────────────────────────
# 10. Evaluate
# ─────────────────────────────────────────────
print("\n=== Final Evaluation on Test Set ===")
test_loss, test_acc = model.evaluate(x_test, y_test, verbose=2)
print(f"Test Accuracy : {test_acc*100:.2f}%")
print(f"Test Loss     : {test_loss:.4f}")

# ─────────────────────────────────────────────
# 11. Save Model
# ─────────────────────────────────────────────
model.save(SAVE_PATH)
print(f"Model saved → {SAVE_PATH}")

# ─────────────────────────────────────────────
# 12. Plot Training Curves
# ─────────────────────────────────────────────
def merge_history(h1, h2):
    merged = {}
    for k in h1.history:
        merged[k] = h1.history[k] + h2.history[k]
    return merged

all_hist = merge_history(history, history_ft)
epochs_range = range(len(all_hist['accuracy']))

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Accuracy
axes[0].plot(epochs_range, all_hist['accuracy'],    label='Train Accuracy', linewidth=2)
axes[0].plot(epochs_range, all_hist['val_accuracy'],label='Val Accuracy',   linewidth=2)
axes[0].axvline(len(history.history['accuracy'])-1, color='gray',
                linestyle='--', label='Fine-tune start')
axes[0].set_title('Accuracy over Epochs', fontsize=14)
axes[0].set_xlabel('Epoch'); axes[0].set_ylabel('Accuracy')
axes[0].legend(); axes[0].grid(True, alpha=0.3)

# Loss
axes[1].plot(epochs_range, all_hist['loss'],    label='Train Loss', linewidth=2)
axes[1].plot(epochs_range, all_hist['val_loss'],label='Val Loss',   linewidth=2)
axes[1].axvline(len(history.history['loss'])-1, color='gray',
                linestyle='--', label='Fine-tune start')
axes[1].set_title('Loss over Epochs', fontsize=14)
axes[1].set_xlabel('Epoch'); axes[1].set_ylabel('Loss')
axes[1].legend(); axes[1].grid(True, alpha=0.3)

plt.suptitle('CIFAR-10 MobileNetV2 Transfer Learning', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig('training_curves.png', dpi=150, bbox_inches='tight')
print("Training curves saved → training_curves.png")
plt.show()
