import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import tensorflow as tf
from keras import layers
from keras.preprocessing import image
from sklearn.metrics import confusion_matrix
import pandas as pd

DATASET_DIR = 'datasets'
CLASS_NAMES = ["COVID-19", "NORMAL", "Viral Pneumonia"]

train_data = defect_tree = tf.keras.preprocessing.image_dataset_from_directory(
    DATASET_DIR,
    labels="inferred",
    label_mode="int",
    class_names=None,
    color_mode="rgb",
    batch_size=32,
    image_size=(244, 244),
    shuffle=True,
    seed=123,
    validation_split=0.2,
    subset="training",
    interpolation="bilinear",
    follow_links=False,
)
test_data = tf.keras.preprocessing.image_dataset_from_directory(
    DATASET_DIR,
    labels="inferred",
    label_mode="int",
    class_names=None,
    color_mode="rgb",
    batch_size=32,
    image_size=(244, 244),
    shuffle=True,
    seed=123,
    validation_split=0.2,
    subset="validation",
    interpolation="bilinear",
    follow_links=False,
)


def showImage(path, disease):
    image_path = path
    new_img = image.load_img(image_path, target_size=(244, 244))
    img = image.img_to_array(new_img)
    np.expand_dims(img, axis=0)
    print(disease)
    plt.imshow(new_img)
    plt.show()


showImage("datasets/Viral Pneumonia/Viral Pneumonia (31).png", "Viral pneumonia")
showImage("datasets/NORMAL/NORMAL (34).png", "NORMAL")
showImage("datasets/COVID/COVID (100).png", "COVID-19")

model = tf.keras.models.Sequential([
    layers.BatchNormalization(),
    layers.Conv2D(32, 3, activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(64, 3, activation='relu'),
    layers.MaxPooling2D(),
    layers.Dropout(0.3),
    layers.Conv2D(128, 3, activation='relu'),
    layers.MaxPooling2D(),
    layers.Dropout(0.2),
    layers.Flatten(),
    layers.Dense(256, activation='relu'),
    layers.Dropout(0.15),
    layers.Dense(3, activation='softmax')
])

early = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
history = model.fit(train_data, validation_data=test_data, batch_size=32, epochs=20, callbacks=[early])
model.evaluate(test_data)

sns.set()

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(loss) + 1)

plt.plot(epochs, acc, color='green', label='Training Accuracy')
plt.plot(epochs, val_acc, color='blue', label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend()
plt.figure()
plt.show()

plt.plot(epochs, loss, color='pink', label='Training Loss')
plt.plot(epochs, val_loss, color='red', label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()

plt.title('Training Data - Accuracy and Loss')
plt.plot(history.history['val_loss'], label='training loss')
plt.plot(history.history['val_accuracy'], label='training accuracy')
plt.legend()
plt.show()

plt.title('Test Data - Accuracy and Loss')
plt.plot(history.history['loss'], label='test loss')
plt.plot(history.history['accuracy'], label='test accuracy')
plt.legend()
plt.show()

y_pred = []
y_true = []

for image_batch, label_batch in test_data:
    y_true.append(label_batch)
    preds = model.predict(image_batch)
    y_pred.append(np.argmax(preds, axis=- 1))

correct_labels = tf.concat([item for item in y_true], axis=0)
predicted_labels = tf.concat([item for item in y_pred], axis=0)

matrix = confusion_matrix(correct_labels, predicted_labels)
conf_matrix = pd.DataFrame(matrix, index=CLASS_NAMES, columns=CLASS_NAMES)
plt.figure(figsize=(15, 15))
sns.heatmap(conf_matrix, annot=True, annot_kws={"size": 15}, fmt='d')


def predict(path):
    new_img = image.load_img(path, target_size=(244, 244))
    img = image.img_to_array(new_img)
    img = np.expand_dims(img, axis=0)
    prediction = model.predict(img)
    prediction = np.argmax(prediction, axis=1)
    print(CLASS_NAMES[prediction[0]])
    plt.imshow(new_img)
    plt.show()


predict("datasets/Viral Pneumonia/Viral Pneumonia (318).png")
predict("datasets/NORMAL/NORMAL (2).png")
predict("datasets/COVID/COVID (67).png")
