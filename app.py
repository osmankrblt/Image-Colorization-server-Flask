import os,cv2
import io
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

import base64
from flask import Flask,request
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt
from skimage import  color


app = Flask(__name__,)

model = load_model("model/model.hdf5")



def convertColorfulOutput(x,output):
    output *= 128
  
    cur = np.zeros((400, 400, 3))
    cur[:,:,0] = x[0][:,:,0]
    cur[:,:,1:] = output[0]
    return cur

def preprocess_image_to_gray_lab(img):

    
    img = cv2.resize(img,(400,400))
    img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)

    img = color.rgb2lab(img/255)
    img = img[:,:,0]
    img = img.reshape(1,400, 400, 1)
    
    return img

def predictImage(image, model):
    
    colorful_image = model.predict(image)

    cur = convertColorfulOutput(image,colorful_image)

    img = color.lab2rgb(cur)
    
    img = img * 255
    img = img.astype("uint8")

  

    img = cv2.cvtColor(img,cv2.COLOR_RGB2BGR)

    return img


@app.route('/predict',methods=['POST',"GET"])
def predict():
    
    if request.method == "POST":

        encodedImg = request.form.get('file')
        
        imgData = base64.b64decode(encodedImg)

        img = np.frombuffer(imgData, dtype=np.uint8)
        img = cv2.imdecode(img, flags=1)


        
        image_shape = img.shape

        lab_img = preprocess_image_to_gray_lab(img)
        

        

        colorful =  predictImage((lab_img),model)
       
        colorful = cv2.resize(colorful,(image_shape[1],image_shape[0]))
    
        _, im_buf_arr = cv2.imencode(".jpg", colorful)

        colorful_bytes = im_buf_arr.tobytes()

        colorful_bytes = base64.b64encode(colorful_bytes)
       

        return colorful_bytes
 
  
    return 'Ok'

        
if __name__ == '__main__':
    
    app.run(debug=True,host='0.0.0.0',port=5000)
