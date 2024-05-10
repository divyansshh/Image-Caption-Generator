from django.shortcuts import render
from django.http import HttpResponse
import os
import numpy as np
from keras.models import load_model
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
import pickle
from PIL import Image
from keras.applications.vgg19 import VGG19, preprocess_input, decode_predictions
import matplotlib.pyplot as plt
from keras.models import Model
from keras.preprocessing.image import load_img,img_to_array

def preprocess_image(image_path,image_name):
    # load VGG19 Model
    model = VGG19(weights='imagenet')

    # restructuring model
    model = Model(inputs = model.inputs, outputs = model.layers[-2].output)

    features = {}
    # dir = 'E:/Project/Project/mlapp/static/Images'

    # Loading image from file
    # img_path = dir + '/' + img
    image = load_img(image_path, target_size=(224,224))
    # image to ndarray
    image = img_to_array(image)
    # reshaping for  model
    image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
    # preprocessing input by vgg
    image = preprocess_input(image)
    # extracting features
    feature = model.predict(image, verbose = 0)

    # saving features to dictionary
    features[image_name.split(".")[0]] = feature
    return features

# Create your views here.
def home(request):
    return render(request, 'home.html')

def upload_file(request):
    if request.method == 'POST':
        uploaded_file = request.FILES.get('file')  # Assuming 'file' is the name of the input field in your HTML form
        if uploaded_file:
            # Process the uploaded file (save it, perform analysis, etc.)
            # For demonstration purposes, let's just store the file name

            image_name = uploaded_file.name

            save_dir = 'E:/Project/Project/mlapp/static/Images'  # Replace with your actual static directory
            global image_path
            image_path = os.path.join(save_dir, image_name)

            # Save the uploaded file to a directory
            with open(os.path.join(save_dir, image_name), 'wb+') as destination:
                for chunk in uploaded_file.chunks():
                    destination.write(chunk)

            context = {'image_name': image_name, 'image_path': image_path}

            # Add the rest of your caption generation logic
            caption_data = generate_caption(image_name)

            return render(request, 'result.html', {'caption_data': caption_data, 'context': context})

    return render(request, 'home.html')

def delete_file(request):
    os.remove(image_path)
    return render(request, 'home.html')

ALLOWED_IMAGE_EXTENSIONS = set(['jpeg','jpg','png','webp'])

def allowed_video_file(filename):
    if (filename.rsplit('.',1)[1].lower() in ALLOWED_IMAGE_EXTENSIONS):
        return True
    else: 
        return False
    
# with open('E:/Project/Project/models/features.pkl', 'rb') as f:
#     features = pickle.load(f)
    
with open('E:/Project/Project/models/mapping.pkl', 'rb') as m:
    mapping = pickle.load(m)

# load the model
model = load_model('E:/Project/Project/models/best_model2.h5')

# saving all captions in a list
all_captions = []
for key in mapping.keys():
    for val in mapping[key]:
        all_captions.append(val)

max_len = max([len(cap.split())for cap in all_captions])

tokenizer = Tokenizer()
tokenizer.fit_on_texts(all_captions)
vocab_size = len(tokenizer.word_index) +1


def idx_to_word(integer, tokenizer):
    for word, index in tokenizer.word_index.items():
        if index == integer:
            return word
    return None

def predict_caption(model, image, tokenizer, max_length):
    # add start tag for generation process
    in_text = 'starttag'
    # iterate over the max length of sequence
    for i in range(max_length):
        # encode input sequence
        sequence = tokenizer.texts_to_sequences([in_text])[0]
        # pad the sequence
        sequence = pad_sequences([sequence], max_length)
        # predict next word
        yhat = model.predict([image, sequence], verbose=0)
        # get index with high probability
        yhat = np.argmax(yhat)
        # convert index to word
        word = idx_to_word(yhat, tokenizer)
        # stop if word not found
        if word is None:
            break
        # append word as input for generating next word
        in_text += " " + word
        # stop if we reach end tag
        if word == 'endtag':
            break
      
    return in_text


def generate_caption(image_name):
    # load the image
    image_id = image_name.split('.')[0]
    img_path = os.path.join('E:/Project/Project/mlapp/static', "Images", image_name)
    # image = Image.open(img_path)
    # predict the caption
    features = preprocess_image(img_path,image_name)
    y_pred = predict_caption(model, features[image_id], tokenizer, max_len)
    y_pred = y_pred.split('starttag')[1].split('endtag')[0]
    print('Predicted:\n')
    print(y_pred)
    return y_pred