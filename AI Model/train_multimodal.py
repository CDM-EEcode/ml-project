import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import ModelCheckpoint
from multimodal_eeg_fnirs import MultimodalEEGNetWorkload
import os

# === Load Data ===
X_eeg = np.load('X_eeg.npy')          # [samples, 32, 500, 1]
X_fnirs = np.load('X_fnirs.npy')      # [samples, 36, 11]
y = np.load('y.npy')                  # [samples] (0 = easy, 1 = medium, 2 = hard)

# === One-hot encoding ===
y_cat = to_categorical(y, num_classes=3)

# === Train/test split ===
X_eeg_train, X_eeg_test, X_fnirs_train, X_fnirs_test, y_train, y_test = train_test_split(
    X_eeg, X_fnirs, y_cat, test_size=0.2, random_state=42, stratify=y)

# === Create Model ===
model = MultimodalEEGNetWorkload(eeg_shape=(32, 500, 1), fnirs_shape=(36, 11), nb_classes=3)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

# === Checkpoint Callback ===
os.makedirs("checkpoints", exist_ok=True)
checkpoint = ModelCheckpoint("checkpoints/best_model.h5", monitor='val_accuracy',
                             save_best_only=True, mode='max', verbose=1)

# === Train ===
history = model.fit(
    [X_eeg_train, X_fnirs_train], y_train,
    validation_data=([X_eeg_test, X_fnirs_test], y_test),
    epochs=100, batch_size=32, callbacks=[checkpoint]
)

# === Save Final Model ===
model.save("MultimodalEEGNetWorkload.h5")

# === Plot Training History ===
plt.figure()
plt.plot(history.history['accuracy'], label='Train Acc')
plt.plot(history.history['val_accuracy'], label='Val Acc')
plt.title('Accuracy over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.savefig('training_accuracy.png')
plt.close()

plt.figure()
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.title('Loss over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.savefig('training_loss.png')
plt.close()

# === Evaluate ===
y_pred = model.predict([X_eeg_test, X_fnirs_test])
y_true = np.argmax(y_test, axis=1)
y_pred_class = np.argmax(y_pred, axis=1)
print("\nClassification Report:")
print(classification_report(y_true, y_pred_class, target_names=["easy", "medium", "hard"]))
