from flask import Flask, request, send_file, jsonify
from flask_cors import CORS
from werkzeug.utils import secure_filename
import os
# import json
# import tensorflow as tf
from PIL import Image
import base64
import io
import cv2 as cv
from age_detect import *
from script import *
import glob
from swapface import *

app = Flask(__name__)

# Allow 
CORS(app)


# Path for uploaded images
UPLOAD_FOLDER = 'data/uploads/'
OUTPUT_FOLDER = 'data/result/'

# Allowed file extransions
ALLOWED_EXTENSIONS = set(['txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif'])
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['OUTPUT_FOLDER'] = OUTPUT_FOLDER


def preprocessing_image(image_path):
	img = Image.open(image_path)
	buffer = io.BytesIO()
	img.save(buffer, 'png')
	buffer.seek(0)
	data = buffer.read()
	data = base64.b64encode(data).decode()
	return data


@app.route("/")
def hello():
	return "Hello World!"

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/upload', methods=['GET', 'POST'])
def upload_file():
	if request.method == 'POST':
		print("request data", request.data)
		print("request files", request.files)

		# check if the post request has the file part
		if 'file' not in request.files:
			return "No file part"
		file = request.files['file']


		if file and allowed_file(file.filename):
			filename = secure_filename(file.filename)
			file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))

	# clear faces
	face_files = glob.glob("./data/faces/*")
	human_files = glob.glob("./data/human/*")

	for file in face_files:
		os.remove(file)
	for file in human_files:
		os.remove(file)
		
	converted_image = cv.imread("./data/uploads/img.png")
	cv.cvtColor(converted_image, cv.COLOR_BGR2RGB)
	cv.imwrite("./data/uploads/img.png", converted_image)
	
	image_path, detection_result, class_names, _MODEL_SIZE=object_detection(image_path=[str("./data/uploads/img.png")])

	
	class_names_predict = draw_boxes(image_path, detection_result, class_names, _MODEL_SIZE)
	class_names_predict = ",".join(class_names_predict)
	f = open("./data/class_names.txt", "w")
	f.write(class_names_predict)
	f.close()
	data = preprocessing_image("./data/result/"+os.listdir("./data/result/")[0])
	
	return f'"data:image/png;base64,{data}"'

@app.route('/dowload', methods=['GET', 'POST'])
def dowload_file():
	if request.method == 'POST':
		print("request data", request.data)
		print("request files", request.files)

		# check if the post request has the file part
		if 'file' not in request.files:
			return "No file part"
		file = request.files['file']


		if file and allowed_file(file.filename):
			filename = secure_filename(file.filename)
			file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))	
	return "Success !"


@app.route('/predict', methods=['GET', 'POST'])
def predict():
	faceSwap("./data/uploads/img_1.png","./data/uploads/img_2.png")
	# object_detection(image_path=[str("./data/uploads/"+os.listdir("data/uploads/")[0])])
	
	data = preprocessing_image("./output/"+os.listdir("./output/")[0])
	return f'"data:image/png;base64,{data}"'

@app.route('/test', methods=['GET', 'POST'])
def test():
	f = open("./data/class_names.txt", "r")
	class_names_preddict = f.read()
	print("==========================")
	print(class_names_preddict)
	return class_names_preddict

if __name__ == "__main__":
	app.run(debug=True, host= "0.0.0.0")