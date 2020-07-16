from os import listdir
from pickle import dump
from keras.applications.vgg16 import VGG16
# from keras.applications.vgg16 import preprocess_input
from keras.applications.inception_v3 import InceptionV3
from keras.applications.inception_v3 import preprocess_input
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.models import Model
from keras.layers import Dense,Dropout

# modelused = "VGG16"
modelused = "InceptionV3"

# extract features from each photo in the directory
def extract_features(directory):
    # load the model
    if modelused == "VGG16":
        model = VGG16()  # For VGG16 model
    else:
        model = InceptionV3(weights='imagenet')
    # re-structure the model
    model.layers.pop()

    ##
    # fdr = Dropout(0.5)(model.layers[-1].output)
    # fden = Dense(256, activation='relu')(fdr)
    ##
    model = Model(inputs=model.inputs, outputs=model.layers[-1].output)
    # model = Model(inputs=model.inputs, outputs=fden)
    # summarize
    print(model.summary())
    # extract features from each photo
    features = dict()
    cnt = 0
    for name in listdir(directory):
        # load an image from file
        filename = directory + '/' + name
        if modelused == "VGG16":
            image = load_img(filename, target_size=(224, 224)) # For VGG16 model
        else:
            image = load_img(filename, target_size=(299, 299))
        
        # convert the image pixels to a numpy array
        image = img_to_array(image)
        # reshape data for the model
        image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
        # prepare the image for the VGG model
        image = preprocess_input(image)
        # get features
        feature = model.predict(image, verbose=0)
        # get image id
        image_id = name.split('.')[0]
        # store feature
        features[image_id] = feature
        cnt += 1
        print('> {0} {1}'.format(cnt,name))

    return features


# extract features from all images
directory = 'Flickr8k_Dataset'
features = extract_features(directory)
print('Extracted Features: %d' % len(features))
# save to file
if modelused == "VGG16":
    dump(features, open('features.pkl', 'wb'))
else:
    dump(features, open('features_incv3.pkl', 'wb'))

