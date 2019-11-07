
import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
import tensorflow.keras.backend as K
import random
from scipy.misc import imsave, imresize
from scipy.optimize import fmin_l_bfgs_b   # https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.fmin_l_bfgs_b.html
from tensorflow.keras.applications import vgg19
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import warnings

random.seed(1618)
np.random.seed(1618)
tf.set_random_seed(1618)

tf.logging.set_verbosity(tf.logging.ERROR)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

CONTENT_IMG_PATH = "fav.jpg"        
STYLE_IMG_PATH = "Starry_Night.jpg"         

CONTENT_IMG_H = 500
CONTENT_IMG_W = 500

STYLE_IMG_H = 500
STYLE_IMG_W = 500

NUM_FILTERS = 512

CONTENT_WEIGHT = 0.1    # Alpha weight.
STYLE_WEIGHT = 1.0      # Beta weight.
TOTAL_WEIGHT = 1.0

TRANSFER_ROUNDS = 3



#=============================<Helper Fuctions>=================================
'''
TODO: implement this.
This function should take the tensor and re-convert it to an image.
'''
def deprocessImage(x):
    x = x.reshape((CONTENT_IMG_H, CONTENT_IMG_W, 3))
    x[:, :, 0] += 103.939
    x[:, :, 1] += 116.779
    x[:, :, 2] += 123.68
    # 'BGR'->'RGB'
    x = x[:, :, ::-1]
    x = np.clip(x, 0, 255).astype('uint8')
    return x


def gramMatrix(x):
    features = K.batch_flatten(K.permute_dimensions(x, (2, 0, 1)))
    gram = K.dot(features, K.transpose(features))
    return gram



#========================<Loss Function Builder Functions>======================

def styleLoss(style, gen):
    return STYLE_WEIGHT * (K.sum(K.square(gramMatrix(style) - gramMatrix(gen))) / (4. * (NUM_FILTERS**2) * ((STYLE_IMG_H * STYLE_IMG_W)**2)))


def contentLoss(content, gen):
    return CONTENT_WEIGHT * K.sum(K.square(gen - content))

# x is (content, style, gen)
def totalLoss(x):
    return CONTENT_WEIGHT * contentLoss(x[0], x[2]) + STYLE_WEIGHT * styleLoss(x[1], x[2])

def totalVariationLoss(x):
    a = K.square(
        x[:, :CONTENT_IMG_H - 1, :CONTENT_IMG_W - 1, :] - x[:, 1:, :CONTENT_IMG_W - 1, :])
    b = K.square(
        x[:, :CONTENT_IMG_H - 1, :CONTENT_IMG_W - 1, :] - x[:, :CONTENT_IMG_H - 1, 1:, :])
    return TOTAL_WEIGHT * K.sum(K.pow(a + b, 1.25))




#=========================<Pipeline Functions>==================================

def getRawData():
    print("   Loading images.")
    print("      Content image URL:  \"%s\"." % CONTENT_IMG_PATH)
    print("      Style image URL:    \"%s\"." % STYLE_IMG_PATH)
    cImg = load_img(CONTENT_IMG_PATH)
    tImg = cImg.copy()
    sImg = load_img(STYLE_IMG_PATH)
    print("      Images have been loaded.")
    return ((cImg, CONTENT_IMG_H, CONTENT_IMG_W), (sImg, STYLE_IMG_H, STYLE_IMG_W), (tImg, CONTENT_IMG_H, CONTENT_IMG_W))



def preprocessData(raw):
    img, ih, iw = raw
    img = img_to_array(img)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        img = imresize(img, (ih, iw, 3))
    img = img.astype("float64")
    img = np.expand_dims(img, axis=0)
    img = vgg19.preprocess_input(img)
    return img


'''
TODO: Allot of stuff needs to be implemented in this function.
First, make sure the model is set up properly. TICK
Then construct the loss function (from content and style loss). TICK
Gradient functions will also need to be created, or you can use K.Gradients().
Finally, do the style transfer with gradient descent.
Save the newly generated and deprocessed images.
'''
def styleTransfer(cData, sData, tData):
    print("   Building transfer model.")
    contentTensor = K.variable(cData)
    styleTensor = K.variable(sData)
    genTensor = K.placeholder((1, CONTENT_IMG_H, CONTENT_IMG_W, 3))
    inputTensor = K.concatenate([contentTensor, styleTensor, genTensor], axis=0)
    model = vgg19.VGG19(include_top=False, weights="imagenet", input_tensor=inputTensor)
    outputDict = dict([(layer.name, layer.output) for layer in model.layers])
    print("   VGG19 model loaded.")
    loss = 0.0
    styleLayerNames = ["block1_conv1", "block2_conv1", "block3_conv1", "block4_conv1", "block5_conv1"]
    contentLayerName = "block5_conv2"
    print("   Calculating content loss.")
    contentLayer = outputDict[contentLayerName]
    contentOutput = contentLayer[0, :, :, :]
    genOutput = contentLayer[2, :, :, :]
    loss += contentLoss(contentOutput,genOutput) 
    print("   Calculating style loss.")
    for layerName in styleLayerNames:
        styleLayer = outputDict[layerName]
        styleOutput = styleLayer[1, :, :, :]
        genOutput = styleLayer[2, :, :, :]
        loss += styleLoss(styleOutput, genOutput)  
    loss += totalVariationLoss(genTensor)

    grads = K.gradients(loss, genTensor)
    print("   Beginning transfer.")
    round_data = tData
    outputs = [loss]
    if isinstance(grads, (list, tuple)):
        outputs += grads
    else:
        outputs.append(grads)

    f_outputs = K.function([genTensor], outputs)

    def loss_func(x):
        x = x.reshape((1, CONTENT_IMG_H, CONTENT_IMG_W, 3))
        #assert isinstance(f_outputs, (list)) == False
        outs = f_outputs([x])        
        loss_value = outs[0]
        print("Loss value: " + str(loss_value))
        return loss_value
    def grad_func(x):
        x = x.reshape((1, CONTENT_IMG_H, CONTENT_IMG_W, 3))
        outs = f_outputs([x])        
        result = None
        if len(outs[1:]) == 1:
            result = outs[1].flatten().astype('float64')
        else:
            result = np.array(outs[1:]).flatten().astype('float64')
        print("Derivatives: " + str(result))
        return result
    #x = preprocess_image(base_image_path)
    for i in range(TRANSFER_ROUNDS):
        
        print("   Step %d." % i)
        #TODO: perform gradient descent using fmin_l_bfgs_b.
        round_data, tLoss, info = fmin_l_bfgs_b(loss_func, round_data.flatten(), fprime=grad_func, maxfun=10)
        print("      Loss: %f." % tLoss)
        img = deprocessImage(round_data)
        saveFile = "iteration_{}.jpg".format(i)  #TODO: Implement.
        imsave(saveFile, img)   #Uncomment when everything is working right.
        print("      Image saved to \"%s\"." % saveFile)
    print("   Transfer complete.")




#=========================<Main>================================================

def main():
    print("Starting style transfer program.")
    raw = getRawData()
    cData = preprocessData(raw[0])   # Content image.
    sData = preprocessData(raw[1])   # Style image.
    tData = preprocessData(raw[2])   # Transfer image.
    styleTransfer(cData, sData, tData)
    print("Done. Goodbye.")



if __name__ == "__main__":
    main()
