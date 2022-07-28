from flask import Flask, render_template, request
from keras.models import load_model
from keras.preprocessing import image
from matplotlib.pyplot import imshow
import numpy as np
import os

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = "./static/"

dic = {0:'A', 1:'B', 2:'C', 3:'D', 4:'E', 5:'F', 6:'G', 7:'H', 8:'I', 9:'J', 10:'K', 11:'L', 12:'M', 13:'N', 14:'O', 15:'P', 16:'Q', 17:'R', 18:'S', 19:'T', 20:'U', 21:'V', 22:'W', 23:'X', 24:'Y', 25:'Z'}

model = load_model('signs_resnet.h5', compile=False)

print(model)
def predict_label(img_path):
    i = image.load_img(img_path, target_size = (64, 64))
    i = image.img_to_array(i)
    i = np.expand_dims(i, axis=0)
    i = i/255.0
    i = i.reshape(1, 64, 64, 3)
    p = model.predict(i)
    p = np.argmax(p)
    return dic[p]

#routes
@app.route('/', methods = ['GET', 'POST'])
def base():
    if request.method == 'POST':
        try:
            img = request.files['my_image']
            img_path = './static/'+ img.filename
            path = os.path.join(app.config['UPLOAD_FOLDER'], img.filename)
            img.save(path)
        except Exception as e:
            print(e)
        
        #p = predict_label(img_path)
        #return render_template('index.html', prediction = p)
        return "Successful"
    return render_template('index.html')

if __name__ == "__main__":
    app.run()

