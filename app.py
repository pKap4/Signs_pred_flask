from flask import Flask, render_template, request
import tensorflow
from keras.models import load_model
from keras.utils import load_img, img_to_array
from matplotlib.pyplot import imshow
import numpy as np
import os
from werkzeug.utils import secure_filename

app = Flask(__name__)
UPLOAD_FOLDER = "/Users/Parth Kapadia/Desktop/Signs Tf FLASK/main/Signs_pred_flask/static/"

model = load_model('signs_resnet.h5', compile=False)
@app.route('/', methods = ['GET', 'POST'])
def base():
    dic = {0:'A', 1:'B', 2:'C', 3:'D', 4:'E', 5:'F', 6:'G', 7:'H', 8:'I', 9:'J', 10:'K', 11:'L', 12:'M', 13:'N', 14:'O', 15:'P', 16:'Q', 17:'R', 18:'S', 19:'T', 20:'U', 21:'V', 22:'W', 23:'X', 24:'Y', 25:'Z'}

    if request.method == 'POST':
        image_file = request.files["image"]
        if image_file:
            image_location = os.path.join(UPLOAD_FOLDER,
                                          image_file.filename)
            image_file.save(image_location)
            i = load_img(("static/"+image_file.filename), target_size = (64, 64))
            i = img_to_array(i)
            i = np.expand_dims(i, axis=0)
            i = i/255.0
            i = i.reshape(1, 64, 64, 3)
            p = model.predict(i)
            p = np.argmax(p)
            print(p)
            p = dic[p]
            
        return render_template("index.html",prediction = p)
    return render_template("index.html", prediction='-')

if __name__ == '__main__':
    app.run(port=12000, debug=True)
                              