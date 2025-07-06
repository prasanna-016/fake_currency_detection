from flask import Flask, request, render_template
import os
import cv2
import numpy as np
import tensorflow as tf
from keras.models import load_model
from werkzeug.utils import secure_filename

# Initialize Flask app
app = Flask(__name__)

# Set upload folder and allowed extensions
UPLOAD_FOLDER = os.path.join(os.getcwd(), 'uploads')
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

# Load model
model = load_model('vgg.h5')

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def process_jpg_image(img):
    img = tf.convert_to_tensor(img[:, :, :3])
    img = tf.image.resize(img, [224, 224])
    img = img / 255.0
    img = np.expand_dims(img, axis=0)
    return img

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return render_template('index.html', prediction_text='No file part in the request')

    file = request.files['file']
    if file.filename == '':
        return render_template('index.html', prediction_text='No file selected')

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)

        img = cv2.imread(file_path)
        if img is None:
            return render_template('index.html', prediction_text='Error reading the image')

        processed_img = process_jpg_image(img)
        prediction = model.predict(processed_img)[0]

        # Assuming model predicts [fake_prob, real_prob]
        fake_percentage = round(prediction[0] * 100, 2)
        real_percentage = round(prediction[1] * 100, 2)

        if real_percentage > fake_percentage:
            result = f"Real ({real_percentage}%)"
        else:
            result = f"Fake ({fake_percentage}%)"

        return render_template(
            'index.html',
            prediction_text=f'The currency is likely: {result}',
            real_score=real_percentage,
            fake_score=fake_percentage
        )

    return render_template('index.html', prediction_text='Invalid file type')

if __name__ == '__main__':
    app.run(debug=True)
