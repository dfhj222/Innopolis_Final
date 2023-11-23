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
import sqlite3
import datetime
import traceback
import matplotlib.pyplot as plt
from skimage import color
from skimage import io
from torchvision import transforms

device = 'cpu' # torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model_autoencoder = AutoencoderConv()
model_autoencoder.load_state_dict(torch.load(r'C:\Users\Imaev-AR\Desktop\Google диск\Обучение Аналитика данных\Итоговая аттестация\InnopolysFinal-main\ac_12.pth'))
model_autoencoder = model_autoencoder.to(device)

model_classifier = TumorClassifier(num_classes=4)
model_classifier.load_state_dict(torch.load(r'C:\Users\Imaev-AR\Desktop\Google диск\Обучение Аналитика данных\Итоговая аттестация\InnopolysFinal-main\best_model.pth'))
model_classifier = model_classifier.to(device)


# imgParameter = r'C:\Users\Imaev-AR\Desktop\Google диск\Обучение Аналитика данных\Итоговая аттестация\дз10. Автокодировщик\after_clean\0.jpg'
imgParameter = r'C:\Users\Imaev-AR\Desktop\Google диск\Обучение Аналитика данных\Итоговая аттестация\дз10. Автокодировщик\224x224_noise\1\glioma_0.jpg'


# input_image = Image.open(imgParameter)

# convert_tensor = transforms.ToTensor()
# convert_tensor(img)

# plt.imshow(input_image, cmap='gray')
# plt.show()

# imgOriginal = img2byte(input_image)
#imgOriginal.save("imgOriginal.jpg")




transform = transforms.Compose([transforms.Resize((224, 224)),torchvision.transforms.Grayscale(num_output_channels=1),transforms.ToTensor()])
transform2 = transforms.Compose([transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
transform3 = transforms.Compose([torchvision.transforms.Grayscale(num_output_channels=1)])

input_image = Image.open(imgParameter)
input_tensor = transform(input_image).reshape([1,1,224,224])

input_numpy = np.array(input_image)
# plt.imshow(input_numpy, cmap='gray')
# plt.show()

with torch.no_grad():
    clean_image = model_autoencoder(input_tensor)
    
# clean_image_numpy = clean_image.numpy().reshape((224,224,1))
# print(clean_image_numpy)
# plt.imshow(clean_image_numpy, cmap='gray')
# plt.show()


clean_image = transform3(clean_image[0])
clean_image_numpy = clean_image.numpy().reshape(224,224)
print(clean_image_numpy)
plt.imshow(clean_image_numpy, cmap='gray')
plt.show()



    
# output_np = output[0].numpy()
# output_np = output_np.reshape(output_np.shape[-1], output_np.shape[-2])
# output_np = output_np * 255
# output_np = np.array(output_np, dtype=np.uint8)

# plt.imshow(output_np, cmap='gray')
# plt.show()

clean_image = transform2(clean_image.repeat(1, 3, 1, 1))
with torch.no_grad():
    output_classes = model_classifier(clean_image)

classes = ['glioma', 'meningioma', 'notumor', 'pituitary']

output_classes = output_classes.to('cpu').numpy()[0]
output_classes[output_classes < 0] = 0
for class_name, value in zip(classes, output_classes):
    if value > 0:
        print(f'{class_name}: {round(value/sum(output_classes) * 100,0)}%')

exit(1)

print('output_np.shape', output_np.shape)
print('output_np', output_np)

cv2.imwrite('output.jpg', output_np)

test = Image.fromarray(output_np, mode="L")
print('test.size', test.size)
imgProcessed = img2byte(test)
#imgProcessed.save("imgProcessed.jpg")

# connection = sqlite3.connect('my_database.db') # ('db\\db.db')
# x = datetime.datetime.now()
# #print(x)
# date, time = str(x).split(' ')
# #print(date, time)
# db_manager.insert_history(date, time)

# connection.close()
