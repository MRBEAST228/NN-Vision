from  PIL import Image
import tensorflow as tf
import cv2
import os
from matplotlib import pyplot as plt
from NN_training import DATASET_PATH

num_classes = len(os.listdir(DATASET_PATH))
class_mode = 'binary' if num_classes == 2 else 'categorical'

def predict_image(image_path):
    if not os.path.exists(image_path):
        print(f'file not found on this path: {image_path}')
        return
    try:
        img = Image.open(image_path)
        img.verify()
        img = Image.open(image_path)
    except (OSError, IOError):
        print(f'file damaged')
        return
    model = tf.keras.models.load_model('best_cnn.keras')
    img = cv2.imread(image_path)

    if img is None:
        print('image unreadable')
        return

    img = cv2.resize(img, (128, 128))
    img = img/255
    img = tf.expand_dims(img, axis=0)

    prediction = model.predict(img)
    class_names = os.listdir(DATASET_PATH)
    if class_mode == 'binary':
        predicted_class = class_names[int(bool(prediction[0] > 0.5))]
    else:
        predicted_class = class_names[tf.argmax(prediction, axis=-1).numpy()[0]]
    print(f'predicted class {predicted_class}')
    img = Image.open(image_path)
    plt.imshow(img)
    plt.title(f'predicted class {predicted_class}')
    plt.axis('off')
    plt.show()

predict_image('dataset/haski/haski7.png')