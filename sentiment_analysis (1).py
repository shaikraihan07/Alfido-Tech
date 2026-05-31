import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, GlobalAveragePooling1D, Dense
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt

# --------------------------------------------------
# Load IMDb Dataset
# --------------------------------------------------

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.imdb.load_data(num_words=10000)

print("Training samples:", len(x_train))
print("Testing samples:", len(x_test))

# --------------------------------------------------
# Padding
# --------------------------------------------------

x_train = pad_sequences(x_train, maxlen=250)
x_test = pad_sequences(x_test, maxlen=250)

print("Review length after padding:", len(x_train[0]))

# --------------------------------------------------
# Build Deep Learning Model
# --------------------------------------------------

model = Sequential([
    Embedding(input_dim=10000, output_dim=16, input_length=250),

    GlobalAveragePooling1D(),

    Dense(16, activation='relu'),

    Dense(1, activation='sigmoid')
])

# --------------------------------------------------
# Compile Model
# --------------------------------------------------

model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# --------------------------------------------------
# Model Summary
# --------------------------------------------------

model.summary()

# --------------------------------------------------
# Train Model
# --------------------------------------------------

history = model.fit(
    x_train,
    y_train,
    epochs=10,
    validation_split=0.2
)

# --------------------------------------------------
# Evaluate Model
# --------------------------------------------------

loss, accuracy = model.evaluate(x_test, y_test)

print("\nTest Accuracy:", accuracy)

# --------------------------------------------------
# Predictions
# --------------------------------------------------

y_pred = model.predict(x_test)

y_pred = (y_pred > 0.5).astype(int)

# --------------------------------------------------
# Confusion Matrix
# --------------------------------------------------

print("\nConfusion Matrix:\n")

print(confusion_matrix(y_test, y_pred))

# --------------------------------------------------
# Classification Report
# --------------------------------------------------

print("\nClassification Report:\n")

print(classification_report(y_test, y_pred))

# --------------------------------------------------
# Accuracy Graph
# --------------------------------------------------

plt.figure(figsize=(8,5))

plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])

plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')

plt.legend(['Train', 'Validation'])

plt.show()

# --------------------------------------------------
# Loss Graph
# --------------------------------------------------

plt.figure(figsize=(8,5))

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])

plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')

plt.legend(['Train', 'Validation'])

plt.show()