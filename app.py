from flask import Flask, request, redirect, render_template
import torch
import torchvision
from arch import *
from utils import *
import numpy as np
from PIL import Image
from matplotlib import cm
import sqlite3
import datetime
import traceback
from torchvision import transforms
import matplotlib.pyplot as plt

app = Flask(__name__)

device = 'cpu' # torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model_autoencoder = AutoencoderConv()
model_autoencoder.load_state_dict(torch.load(r'C:\Users\Imaev-AR\Desktop\Google диск\Обучение Аналитика данных\Итоговая аттестация\Innopolis_Final\autoencoder_model.pth'))
model_autoencoder = model_autoencoder.to(device)

model_classifier = TumorClassifier(num_classes=4)
model_classifier.load_state_dict(torch.load(r'C:\Users\Imaev-AR\Desktop\Google диск\Обучение Аналитика данных\Итоговая аттестация\Innopolis_Final\classifier_model.pth'))
model_classifier = model_classifier.to(device)

transform = transforms.Compose([transforms.Resize((224, 224)),torchvision.transforms.Grayscale(num_output_channels=1),transforms.ToTensor()])
transform2 = transforms.Compose([transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
transform3 = transforms.Compose([torchvision.transforms.Grayscale(num_output_channels=1)])

@app.route("/")
def index():
    return render_template('index.html')

@app.route('/process', methods = ['POST'])
def process():
    print('request.files', request.files)
    imgParameter = request.files['imgParameter']
    print('request.files[imgParameter]', imgParameter)
    
    try:
        input_image = Image.open(imgParameter)
        input_tensor = transform(input_image).reshape([1,1,224,224])
    except Exception as e:
        print(e)
        print(traceback.format_exc())
        errorName = e
        stackTrace = traceback.format_exc()
        return render_template('error.html', errorName = errorName, stackTrace = stackTrace)
    
    with torch.no_grad():
        clean_image = model_autoencoder(input_tensor)
        
    with torch.no_grad():
        output_classes = model_classifier(transform2(clean_image.repeat(1, 3, 1, 1)))

    classes = ['glioma', 'meningioma', 'notumor', 'pituitary']

    stringParameter = ''
    output_classes = output_classes.to('cpu').numpy()[0]
    output_classes[output_classes < 0] = 0
    for class_name, value in zip(classes, output_classes):
        if value > 0:
            print(f'{class_name}: {round(value/sum(output_classes) * 100,0)}%')
            stringParameter = stringParameter + f'{class_name}: {round(value/sum(output_classes) * 100,0)}% '
    
    imgOriginal = img2byte(input_image)
    
    print(clean_image.shape)
    clean_image =  clean_image[0][0]
    clean_image_numpy = clean_image.numpy().reshape(224,224)
    print(clean_image_numpy)
    
    clean_image_numpy = np.array(clean_image_numpy * 255, dtype=np.uint8)
    test = Image.fromarray(clean_image_numpy, mode="L")
    
    imgProcessed = img2byte(test)
    
    sqliteConnection = sqlite3.connect(r'C:\Users\Imaev-AR\Desktop\Google диск\Обучение Аналитика данных\Итоговая аттестация\Innopolis_Final\SQLite_database.db')
    cursor = sqliteConnection.cursor()
    sqlite_insert_blob_query = """INSERT INTO table_1
                                  (input_image, output_image, model_prediction, datetime) VALUES (?, ?, ?, ?)"""
                 
    now = datetime.datetime.now()

    data_tuple = (imgOriginal.decode('utf-8'), imgProcessed.decode('utf-8'), stringParameter, str(now))
    cursor.execute(sqlite_insert_blob_query, data_tuple)
    sqliteConnection.commit()
        
    sqliteConnection.close()
    
    return render_template('process.html', stringParameter = stringParameter, #, intParameter = intParameter, 
    imgOriginal = imgOriginal.decode('utf-8'), imgProcessed = imgProcessed.decode('utf-8'))

@app.route("/result")
def result():
    sqliteConnection = sqlite3.connect(r'C:\Users\Imaev-AR\Desktop\Google диск\Обучение Аналитика данных\Итоговая аттестация\Innopolis_Final\SQLite_database.db')
    cursor = sqliteConnection.cursor()

    cursor.execute('SELECT * FROM table_1')
    data = cursor.fetchall()

    to_render = []
    for row in data:
        to_render.append(row)
    sqliteConnection.commit()
    
    return render_template('result.html', data = to_render)

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0')