
# Code by WisdomML
## wisdomml.in

from __future__ import division, print_function
# coding=utf-8
import sys
import os
import glob
import numpy as np
import cv2
# Keras

from tensorflow.keras.models import load_model
from PIL import Image
from tensorflow.keras.preprocessing import image
from tensorflow.keras.utils import img_to_array

# Flask utils
from flask import Flask, request, render_template
from werkzeug.utils import secure_filename

os.environ["CUDA_VISIBLE_DEVICES"]="-1"
# Define a flask app
app = Flask(__name__)

# Model saved with Keras model.save()
MODEL_PATH = 'bestmodel18122022.h5'

#Load your trained model
model = load_model(MODEL_PATH)
#model._make_predict_function()          # Necessary to make everything ready to run on the GPU ahead of time
print('Model loaded. Start serving...')



def model_predict(img_path, model):
    img = image.load_img(img_path, target_size=(224,224)) #target_size must agree with what the trained model expects!!

    # # Preprocessing the image
    # #img = image.img_to_array(img)
    input_arr=img_to_array(img)/255

#     img = cv2.imread(img_path)
# # new_img = crop_brain_contour(ex_img, True)
#     imge=Image.fromarray(img)
#     img=imge.resize((240,240))
#     img=np.array(img)


    input_arr=np.expand_dims(input_arr,axis=0)

    #img = img.astype('float32')/255
   
    preds = model.predict(input_arr)
    print(preds)
    #pred = np.argmax(preds,axis = 1)
    preds= 1 if(preds>0.59) else 0
    return preds
   
   
    #pred = np.argmax(preds,axis = 1)
    


@app.route('/', methods=['GET'])
def index():
    # Main page
    return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        # Get the file from post request
        f = request.files['file']

        # Save the file to ./uploads
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
            basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)

        # Make prediction
        pred = model_predict(file_path, model)
        # accu=pred*100
        # pred = 1 if pred>0.6 else 0
         
        # print(pred,accu)
        os.remove(file_path)#removes file from the server after prediction has been returned

        # Arrange the correct return according to the model. 
		# In this model 1 is Pneumonia and 0 is Normal.
        str0 = 'no  '
        str1 = 'yes '
        # str3 = 'pituitary'
        # str2 = 'No Tumour'
        if pred == 0:
            return str0
        elif(pred==1):
            return str1
        else:
            return "Confused error"
    return None

    #this section is used by gunicorn to serve the app on Heroku
if __name__ == '__main__':
        app.run(debug=True, host="0.0.0.0")
    #uncomment this section to serve the app locally with gevent at:  http://localhost:5000
    # Serve the app with gevent 
    #http_server = WSGIServer(('', 5000), app)
    #http_server.serve_forever()
