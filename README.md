# Image Captioning
  Deep Learning model to generate a caption for a given image. <br />
  See [report.pdf](https://github.com/ramprasadkillari/Image-Captioning/blob/master/report.pdf) for more details about the architecture of model.  <br />
  Used VGG16 and InceptionV3 models to extract features from the Image.

## Datasets:
 To just visualise the results and model's output, download the Caption folder which contains pre-trained model.
 For training the model from scratch download the Flickr Datasets. <br />
  [Flickr 8k](https://forms.illinois.edu/sec/1713398) <br />
  Flickr 30k

## Usage:

### Captioning
  ```python
  cd Caption/
  python3 caption.py dog.jpg
  ```

### Training

  ```python
  cd Training/
  python3 features.py #Saves the features into features.pkl file
  python3 train.py
  ```
## Output:
  ![alt text](https://github.com/ramprasadkillari/Image-Captioning/blob/master/Results/girls_inc3.png?raw=true)
  

## Note:  
  First time you train this model, Keras will download the model weights from the Internet, which are about 500 Megabytes. This may take a few minutes.<br />
  If you don't want to train the model, Just download the Caption Folder, thus saving time. 

## References:
  [machinelearningmastery](https://machinelearningmastery.com/develop-a-deep-learning-caption-generation-model-in-python/) <br />
  [Andrej Karpathy](https://cs.stanford.edu/people/karpathy/cvpr2015.pdf)


