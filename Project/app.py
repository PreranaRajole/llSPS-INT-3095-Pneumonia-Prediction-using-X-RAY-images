import numpy as np
import os
from keras.models import load_model
from keras.preprocessing import image
import tensorflow as tf
global graph
graph = tf.get_default_graph()
from flask import Flask , request, render_template
from werkzeug.utils import secure_filename
from gevent.pywsgi import WSGIServer

app = Flask(__name__)
model = load_model("sPE60NoRescalePdetection.h5")

@app.route('/')
def index():
    return render_template('base.html')

@app.route('/predict',methods = ['GET','POST'])
def upload():
    if request.method == 'POST':
        f = request.files['image']
        print("current path")
        basepath = os.path.dirname(__file__)
        print("current path", basepath)
        filepath = os.path.join(basepath,'uploads',f.filename)
        print("upload folder is ", filepath)
        f.save(filepath)
        
        img = image.load_img(filepath,target_size = (64,64))
        x = image.img_to_array(img)
        x = np.expand_dims(x,axis =0)
        
        with graph.as_default():
            pred = model.predict_classes(x)
            
            #print("prediction",pred)
            
        index = ['NORMAL', 'PNEUMONIA']
        if pred[0][0]==0:
            text = "the patient is " + str(index[pred[0][0]])
        else:
            text = "the patient is having " + str(index[pred[0][0]])
        
    return text
if __name__ == '__main__':
    app.run(debug = False, threaded = False)

        
        
        
    
    
    

