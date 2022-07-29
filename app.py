from flask import Flask, render_template, request
from keras.models import load_model
from keras.preprocessing import image
from matplotlib.pyplot import imshow
import numpy as np
import os
from werkzeug.utils import secure_filename

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = "./static/"

dic = {0:'A', 1:'B', 2:'C', 3:'D', 4:'E', 5:'F', 6:'G', 7:'H', 8:'I', 9:'J', 10:'K', 11:'L', 12:'M', 13:'N', 14:'O', 15:'P', 16:'Q', 17:'R', 18:'S', 19:'T', 20:'U', 21:'V', 22:'W', 23:'X', 24:'Y', 25:'Z'}

model = load_model('signs_resnet.h5', compile=False)

def predict_label(img_path):
    i = image.load_img(img_path, target_size = (64, 64))
    i = image.img_to_array(i)
    i = np.expand_dims(i, axis=0)
    i = i/255.0
    i = i.reshape(1, 64, 64, 3)
    p = model.predict(i)
    p = np.argmax(p)
    return dic[p]

def get_file_path_and_save(request):
    # Get the file from post request
    f = request.files['file']

    # Save the file to ./static
    basepath = os.path.dirname(__file__)
    file_path = os.path.join(
        basepath, 'static', secure_filename(f.filename))
    f.save(file_path)
    return file_path

#routes
@app.route('/', methods=['GET', 'POST'])
def index():
    request_type_str = request.method
    if request_type_str == 'GET':
        # Main page
        return render_template('index.html')
    elif request_type_str == 'POST':
        file_path = get_file_path_and_save(request)
        prediction = predict_label(file_path)
        return render_template('index.html', prediction = prediction)
        
if __name__ == "__main__":
    app.run()

