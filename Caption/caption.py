from pickle import load
from numpy import argmax
from keras.preprocessing.sequence import pad_sequences
from keras.applications.inception_v3 import InceptionV3
# from keras.applications.inception_v3 import preprocess_input
from keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import preprocess_input
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.models import Model
from keras.models import load_model
from keras.layers import Dense,Dropout
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from sys import argv

modelused = "VGG16"
# modelused = "InceptionV3"

# extract features from each photo in the directory
def extract_features(filename):
	# load the model
	if modelused == "InceptionV3":
		model = InceptionV3(weights='imagenet')
		image = load_img(filename, target_size=(299, 299))   # For InceptionV3 model
	else:
		model = VGG16() # For VGG16 mdoel
		image = load_img(filename, target_size=(224, 224)) # For VGG16 model
	# re-structure the model
	model.layers.pop()
	##
	# fdr = Dropout(0.5)(model.layers[-1].output)
	# fden = Dense(256, activation='relu')(fdr)
	##
	model = Model(inputs=model.inputs, outputs=model.layers[-1].output)
	# model = Model(inputs=model.inputs, outputs=fden)

	# model = Model(inputs=model.inputs, outputs=model.layers[-1].output)

	# convert the image pixels to a numpy array
	image = img_to_array(image)
	# reshape data for the model
	image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
	# prepare the image for the VGG model
	image = preprocess_input(image)
	# get features
	feature = model.predict(image, verbose=0)
	return feature

# map an integer to a word
def word_for_id(integer, tokenizer):
	for word, index in tokenizer.word_index.items():
		if index == integer:
			return word
	return None

# generate a description for an image
def generate_desc(model, tokenizer, photo, max_length):
	# seed the generation process
	in_text = 'startseq'
	# iterate over the whole length of the sequence
	for i in range(max_length):
		# integer encode input sequence
		sequence = tokenizer.texts_to_sequences([in_text])[0]
		# pad input
		sequence = pad_sequences([sequence], maxlen=max_length)
		# predict next word
		yhat = model.predict([photo,sequence], verbose=0)
		# convert probability to integer
		yhat = argmax(yhat)
		# map integer to word
		word = word_for_id(yhat, tokenizer)
		# stop if we cannot map the word
		if word is None:
			break
		# append as input for generating the next word
		in_text += ' ' + word
		# stop if we predict the end of the sequence
		if word == 'endseq':
			break
	return in_text

# load the tokenizer
tokenizer = load(open('tokenizer.pkl', 'rb'))
# pre-define the max sequence length (from training)
max_length = 34
# load the model
if modelused == "InceptionV3":
	model = load_model('modelnew_incv3_3.h5')  # For InceptionV3 model
else:
	model = load_model('modelnew_3.h5')  # For VGG16 model
# load and prepare the photograph
image = argv[1]
photo = extract_features(image)
# generate description
description = generate_desc(model, tokenizer, photo, max_length)
print(description)
description2 = description.split(' ')[1:]
if(description2[-1] == "endseq"):
	description2 = description2[:-1]

description2 = ' '.join(description2)
img=mpimg.imread(image)
imgplot = plt.imshow(img)
plt.xticks([])
plt.yticks([])
plt.xlabel(description2,fontsize='18')
plt.show()
#print(description)