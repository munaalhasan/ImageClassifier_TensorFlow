# Import TensorFlow
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import keras

import tensorflow as tf
import tensorflow_datasets as tfds
import tensorflow_hub as hub
import tf_keras

# Make all other necessary imports.
import matplotlib.pyplot as plt
import json
import numpy as np

from PIL import Image
import argparse
import logging
logger = tf.get_logger()
logger.setLevel(logging.ERROR)

batch_size = 64
image_size = 224

def process_image(image):
    image = tf.cast(image, tf.float32)
    image = tf.image.resize(image, (image_size, image_size)).numpy()
    image /= 255
    
    return image


def predict(image_path, model, top_k=5):
    image = Image.open(image_path)
    test_image = np.asarray(image)
    processed_test_image = process_image(test_image)
    expanded_test_image = np.expand_dims(processed_test_image, axis=0)
    
    pred_image = model.predict(expanded_test_image)
    values, indices = tf.math.top_k(pred_image, k=top_k)
    probs = values.numpy()[0]
    classes = indices.numpy()[0]
    
    # preapere the result for presenting
    probs = list(probs)
    classes = list(map(str, classes))
    
    return probs, classes


class_names = [ ]

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('arg1')
    parser.add_argument('arg2')
    parser.add_argument('--top_k')
    parser.add_argument('--category_names') 
    
    
    args = parser.parse_args()
    
    image_path = args.arg1
    
    model = tf_keras.models.load_model(args.arg2 ,custom_objects={'KerasLayer':hub.KerasLayer})
    

    if args.top_k is None and args.category_names is None:
        probs, classes = predict(image_path, model)
        print("The probabilities and classes of the images: ")
        

    elif args.top_k is not None:
        top_k = int(args.top_k)
        probs, classes = predict(image_path, model, top_k)
        print("The top {} probabilities and classes of the images: ".format(top_k))
       

    elif args.category_names is not None:
        with open(args.category_names, 'r') as f:
            class_names = json.load(f)
        probs, classes = predict(image_path, model)
        print("The probabilities and classes of the images: ")
        classes = [class_names[class_] for class_ in  classes]
            
            
    for prob, class_ in zip(probs, classes):
        print('\u2022 "{}":  {:.3%}'.format(class_, prob))
        
    
    print('\nThe flower label is: "{}"'.format(classes[0]))

#$ python predict.py test_images/cautleya_spicata.jpg keras_model.h5
#$ python predict.py test_images/cautleya_spicata.jpg keras_model.h5 --top_k 5
#$ python predict.py test_images/cautleya_spicata.jpg keras_model.h5 --category_names label_map.json
#$ python predict.py test_images/cautleya_spicata.jpg keras_model.h5 --top_k 8 --category_names label_map.json
#$ python predict.py ./test_images/hard-leaved_pocket_orchid.jpg keras_model.h5
#$ python predict.py ./test_images/hard-leaved_pocket_orchid.jpg keras_model.h5 --category_names label_map.json

    