from __future__ import absolute_import
from __future__ import print_function
import numpy as np
import tensorflow as tf
import random
from keras.datasets import mnist
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Input, Lambda
from keras.optimizers import RMSprop
from keras import backend as K
import pdb
from os import listdir,chdir
from PIL import Image

# multiple normalize function can be avoided.
# Validate whether if gradient computed in backend is right using K.gradients.
#https://stackoverflow.com/questions/42163697/multi-input-gradients-calculation-on-keras-with-k-function
#https://stackoverflow.com/questions/44444475/accessing-gradient-values-of-keras-model-outputs-with-respect-to-inputs
def binomial_dev(vects,batch_size = 128,loss_weights = 1):
#    pdb.set_trace()
    x,y = vects
    vunit1 = K.l2_normalize(x, axis = 1)
    vunit2 = K.l2_normalize(y, axis = 1)
    kvar3 = K.concatenate([vunit1,vunit2],axis = 0)
    tkvar3 = K.transpose(kvar3)
    vSimilarity = K.dot(kvar3,tkvar3) + K.epsilon()
    vM =np.ones([2*batch_size,2*batch_size])
    np.fill_diagonal(vM,0)
    vW = vM
    vW = 2*vM/batch_size
    #applying cost for negative pair of 2:
    vM = -2*vM               
    for i in range(batch_size):
        if(i%2 == 0):
            vM[i,i+batch_size] = vM[i+batch_size,i] = 0
        else:
            vM[i,i+batch_size] = vM[i+batch_size,i] = 1      
    #pdb.set_trace()   
    vM = K.variable(vM)               
    vW = K.variable(vW)               
    beta = .5
    alpha = 2
    vcost = K.sum(K.log(K.exp((vSimilarity - beta)*vM*-1*alpha) + 1)*vW)
    vcost = K.reshape(vcost,(1,))
    vcost = K.repeat_elements(vcost, batch_size,axis = 0)
    return vcost
 
#output shape for lambda.
def output_shape(shapes):
    shape1, shape2 = shapes
    return (shape1[0], 1)

    #Cosine-Similarity.
def cosine_distance(vects):
    #pdb.set_trace()
    x, y = vects
    vunit1 = K.l2_normalize(x, axis = -1)
    vunit2 = K.l2_normalize(y, axis = -1)
    #return K.sqrt(K.maximum(K.sum(K.square(x - y), axis=1, keepdims=True), K.epsilon()))
    return K.sum(vunit1*vunit2, axis = -1)
    
#Fake-Loss.
def contrastive_loss(y_true, y_pred):
    '''Contrastive loss from Hadsell-et-al.'06
    http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
    '''
    margin = 1
    return K.mean(y_true * K.square(y_pred) +
                  (1 - y_true) * K.square(K.maximum(margin - y_pred, 0)))

def binomial_loss(y_true, y_pred):
    '''Contrastive loss from Hadsell-et-al.'06
    http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
    '''
    return K.sqrt(K.mean(K.square(y_pred)))
    
def create_base_network(input_dim):
    '''Base network to be shared (eq. to feature extraction).
    '''
    seq = Sequential()
    seq.add(Dense(256, input_shape=(input_dim,), activation='relu'))
    seq.add(Dropout(0.2))
    seq.add(Dense(256, activation='relu'))
    seq.add(Dropout(0.2))
    seq.add(Dense(400, activation='relu'))
    return seq  

# Pairing for viper Dataset.
# Change path1 and path2  to folder of Viper Dataset.
def pairing():
    
    pairs = []
    labels = []
    path1 = "C:/Stuff/work/ANU/PersonID/data/VIPeR/cam_a/"
    path2 = "C:/Stuff/work/ANU/PersonID/data/VIPeR/cam_b/"
    cam_a = listdir(path1)
    cam_b = listdir(path2)
#    pdb.set_trace()
    vcount = 0
    for i in cam_a:
        if(vcount>=576):
            break
        img_a = Image.open(path1 + i)
        img_a = img_a.convert('L')
        img_a = np.array(img_a.resize([28,28]))
        img_a = np.ndarray.flatten(img_a)
        img_a = img_a/255
        img_b = Image.open(path2 + cam_b[vcount])
        img_b = img_b.convert('L')
        img_b = np.array(img_b.resize([28,28]))
        img_b = np.ndarray.flatten(img_b)
        img_b = img_b/255
        pairs += [[img_a, img_b]]
        val = random.choice(range(630))
        if(val == i):
            val = val-1
        img_b = Image.open(path2 + cam_b[val])
        img_b = img_b.convert('L')
        img_b = np.array(img_b.resize([28,28]))
        img_b = np.ndarray.flatten(img_b)
        img_b = img_b/255
        pairs += [[img_a, img_b]]
        labels += [1,0]
        vcount += 1
    return pairs,np.array(labels)   

# pair + labels.
tr_pairs, tr_y = pairing()
tr_pairs = np.reshape(tr_pairs,(1152,2,784))
te_pairs = tr_pairs[1024:1152,]
te_y = tr_y[1024:1152,]
tr_pairs = tr_pairs[:1024,]
tr_y = tr_y[:1024,]

# input values
input_dim = 784
epochs = 1


base_network = create_base_network(input_dim)

input_a = Input(shape=(input_dim,))
input_b = Input(shape=(input_dim,))

sess = tf.Session()                    
vinit = tf.global_variables_initializer()
sess.run(vinit)
K.set_session(sess)                    

    
# because we re-use the same instance `base_network`,
# the weights of the network
# will be shared across the two branches
processed_a = base_network(input_a)
processed_b = base_network(input_b)

# Dual Output.
# Co-sine Similarity (Main-Output)
distance = Lambda(cosine_distance,
                  output_shape= output_shape)([processed_a, processed_b])

#Loss-function as an output.
distance2 = Lambda(binomial_dev,
                   output_shape= output_shape)([processed_a, processed_b])


model = Model([input_a, input_b],
              [distance,distance2])  ## 2-input-2-output-model


rms = RMSprop()
model.compile(loss=[contrastive_loss,binomial_loss],
              optimizer=rms,loss_weights = [0,1])

model.fit([tr_pairs[:, 0], tr_pairs[:, 1]], [tr_y,tr_y],
          batch_size=128,
          epochs=epochs,
          validation_data=([te_pairs[:, 0], te_pairs[:, 1]], [te_y,te_y]))

pred = model.predict([tr_pairs[0:128, 0],
                      tr_pairs[0:128, 1]],
                     batch_size = 128)

vpred = pred[0]
vpred[vpred>.5] = 1
vpred[vpred<.5] = 0
vdiff =  vpred - tr_y[:128]
vacc = np.sum(vdiff ==0)/128
print(vacc)



