import os
from flask import Flask, redirect, render_template, request
from PIL import Image
import torchvision.transforms.functional as TF
from CNN import CNN
import numpy as np
import torch
import pandas as pd


disease_info = pd.read_csv('MyApp/disease_info.csv', encoding='cp1252')
supplement_info = pd.read_csv('MyApp/supplement_info.csv', encoding='cp1252')

model = CNN(K=39)
model.load_state_dict(torch.load('MyApp/ProjectXOAdv.pt', map_location=torch.device('cpu')))
model.eval()

def prediction(image_path):
    image = Image.open(image_path).convert('RGB')  # Ensure image has 3 channels
    image = image.resize((224, 224))
    input_data = TF.to_tensor(image)
    input_data = input_data.unsqueeze(0)  # Add a batch dimension
    output = model(input_data)
    output = output.detach().numpy()
    index = np.argmax(output)
    return index


app = Flask(__name__)

@app.route('/')
def home_page():
    return render_template('home.html')

@app.route('/contact')
def contact():
    return render_template('contact-us.html')

@app.route('/index')
def ai_engine_page():
    return render_template('index.html')


@app.route('/submit', methods=['GET', 'POST'])
def submit():
    if request.method == 'POST':
        image = request.files['image']
        filename = image.filename
        from werkzeug.utils import secure_filename
        secure_name = secure_filename(filename)
        file_path = os.path.join('MyApp/static/uploads', secure_name)
        image.save(file_path)
        print(file_path)
        pred = prediction(file_path)
        title = disease_info['disease_name'][pred]
        description = disease_info['description'][pred]
        prevent = disease_info['Possible Steps'][pred]
        image_url = disease_info['image_url'][pred]
        supplement_name = supplement_info['supplement name'][pred]
        supplement_image_url = supplement_info['supplement image'][pred]
        supplement_buy_link = supplement_info['buy link'][pred]
        return render_template('submit.html', title=title, desc=description, prevent=prevent, 
                               image_url=image_url, pred=pred, sname=supplement_name, 
                               simage=supplement_image_url, buy_link=supplement_buy_link)


if __name__ == '__main__':
    app.run(debug=True)
