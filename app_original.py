from flask import Flask, request, redirect, render_template
import torch
import torchvision
from arch import *
from utils import *
import cv2
import numpy as np
import io
import base64
from PIL import Image
from matplotlib import cm
from db.db import *
import datetime
import traceback

app = Flask(__name__)

device = 'cpu' # torch.device('cuda' if torch.cuda.is_available() else 'cpu')

ac_loaded = AutoencoderConv()
ac_loaded.load_state_dict(torch.load('ac_2.pth', map_location=torch.device(device)))
ac_loaded = ac_loaded.to(device)
ac_loaded.eval()

@app.route("/")
def index():
    return render_template('index.html')

@app.route('/process', methods = ['POST'])
def result():
    stringParameter = request.form['stringParameter']
    intParameter = request.form['intParameter']
    
    print('request.files', request.files)
    imgParameter = request.files['imgParameter']
    print('request.files[imgParameter]', imgParameter)
    
    try:
        pil_image = Image.open(imgParameter)
        print(pil_image.size)
    except Exception as e:
        print(e)
        print(traceback.format_exc())
        errorName = e
        stackTrace = traceback.format_exc()
        return render_template('error.html', errorName = errorName, stackTrace = stackTrace)
    
    imgOriginal = img2byte(pil_image)
    #imgOriginal.save("imgOriginal.jpg")
    
    numpy_array = np.array(pil_image, dtype=np.float32)    
    print(numpy_array)
    
    cv2.imwrite('form_img.jpg', numpy_array)
    numpy_array = numpy_array / 255
    
    numpy_array_reshaped = numpy_array.reshape(1, 1, numpy_array.shape[0], numpy_array.shape[1])
    print('numpy_array_reshaped.shape', numpy_array_reshaped.shape)
    
    tensor_val = torch.tensor(numpy_array_reshaped, dtype=torch.float32)
    print('tensor_val.size()', tensor_val.size())
    
    with torch.no_grad():
        output = ac_loaded(tensor_val)
        print('output.shape', output.shape)
        
    output_np = output.numpy()
    output_np = output_np.reshape(output_np.shape[-1], output_np.shape[-2])
    output_np = output_np * 255
    output_np = np.array(output_np, dtype=np.uint8)
    print('output_np.shape', output_np.shape)
    print('output_np', output_np)
    
    cv2.imwrite('output.jpg', output_np)
    
    test = Image.fromarray(output_np, mode="L")
    print('test.size', test.size)
    imgProcessed = img2byte(test)
    #imgProcessed.save("imgProcessed.jpg")
    
    db_manager = DbManager('db\\db.db')
    x = datetime.datetime.now()
    #print(x)
    date, time = str(x).split(' ')
    #print(date, time)
    db_manager.insert_history(date, time)
    
    return render_template('process.html', stringParameter = stringParameter, intParameter = intParameter, 
    imgOriginal = imgOriginal.decode('utf-8'), imgProcessed = imgProcessed.decode('utf-8'))

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0')