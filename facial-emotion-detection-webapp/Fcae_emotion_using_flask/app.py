from flask import Flask, render_template, request 

import cv2
import numpy as np 
from tensorflow.keras.models import load_model
import os

app = Flask(__name__)

model = load_model('model.h5')

faceDetect=cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 1

p1 = None

@app.route('/')
def index():
	return render_template('index.html')

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    #print(os.getcwd())
	image = request.files['select_file']
	print(os.getcwd())

	image.save('Fcae_emotion_using_flask/static/file.jpg')

	image = cv2.imread('Fcae_emotion_using_flask/static/file.jpg')

	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	cascade = cv2.CascadeClassifier('haarcascade_frontalface_alt2.xml')
	
	faces = cascade.detectMultiScale(gray, 1.1, 3)

	for x,y,w,h in faces:
		cv2.rectangle(image, (x,y), (x+w, y+h), (0,255,0), 2)

		cropped = image[y:y+h, x:x+w]


	cv2.imwrite('Fcae_emotion_using_flask/static/after.jpg', image)
	try:
		cv2.imwrite('Fcae_emotion_using_flask/static/cropped.jpg', cropped)

	except:
		pass



	try:
		img = cv2.imread('Fcae_emotion_using_flask/static/cropped.jpg', 0)

	except:
		img = cv2.imread('Fcae_emotion_using_flask/static/file.jpg', 0)

	img = cv2.resize(img, (48,48))
	img = img/255

	img = img.reshape(1,48,48,1)

	model = load_model('model.h5')

	pred = model.predict(img)

	print(pred)
	p1 = pred
	label_map = ['Stress','Neutral' , 'Stress', 'Not Stress', 'Stress', 'Not Stress']
	p2 = max(p1)
	print('p2',p2)
	pred = np.argmax(pred)
	print(pred)
	final_pred = label_map[pred]

	
	return render_template('predict.html', data=final_pred)
print(p1)
if __name__ == "__main__":
	app.run(debug=True)