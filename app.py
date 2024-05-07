from flask import Flask, render_template, request, redirect, url_for
from werkzeug.utils import secure_filename
import os
import cv2
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.xception import preprocess_input
import numpy as np
from collections import Counter

app = Flask(__name__)

app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg', 'gif', 'mp4'}
app.config['UPLOAD_FOLDER'] = 'static/uploads'

model = load_model('xception_deepfake_image.h5')

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/image-upload', methods=['GET', 'POST'])
def image_upload():
    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            img_path = filepath
            img = image.load_img(img_path, target_size=(299, 299))  
            x = image.img_to_array(img)
            x = np.expand_dims(x, axis=0)
            x = preprocess_input(x)
            predictions = model.predict(x)
            probability = predictions[0][0]
            result = "FAKE" if probability >= 0.5 else "REAL"
            filepath = filepath.replace("\\","/")
            return render_template("image.html", result=result, filepath=filepath)
    return render_template('image.html')

@app.route('/video-upload', methods=['GET', 'POST'])
def video_upload():
    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            input_video_path = filepath
            output_video_path = 'static/uploads/output.mp4'
            video_capture = cv2.VideoCapture(input_video_path)
            fps = int(video_capture.get(cv2.CAP_PROP_FPS))
            frame_width = int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
            frame_height = int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))
            
            labels = []
            
            while True:
                ret, frame = video_capture.read()
                if not ret:
                    break
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                img = cv2.resize(frame, (299, 299))
                img = image.img_to_array(img)
                img = np.expand_dims(img, axis=0)
                img = preprocess_input(img)
                predictions = model.predict(img)
                probability = predictions[0][0]
                label = "FAKE" if probability >= 0.5 else "REAL"
                labels.append(label)
                text_x = int((frame_width - cv2.getTextSize(f"Prediction: {label} (Probability: {probability:.2f})", cv2.FONT_HERSHEY_SIMPLEX, 1, 2)[0][0]) / 2)
                text_y = 50
                cv2.putText(frame, f"Prediction: {label} (Probability: {probability:.2f})", (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                out.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
            
            video_capture.release()
            out.release()
            label_counts = Counter(labels)
            result = "REAL" if label_counts["REAL"] > label_counts["FAKE"] else "FAKE"
            return render_template('video.html', result=result)
    return render_template('video.html')

if __name__ == "__main__":
    app.run(debug=True)
