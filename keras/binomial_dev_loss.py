'''Train a Siamese MLP on pairs of digits from the MNIST dataset.
It follows Hadsell-et-al.'06 [1] by computing the Euclidean distance on the
output of the shared network and by optimizing the contrastive loss (see paper
for mode details).
[1] "Dimensionality Reduction by Learning an Invariant Mapping"
    http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
Gets to 99.5% test accuracy after 20 epochs.
3 seconds per epoch on a Titan X GPU
'''
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
# multiple normalize function can be avoided.
               
def binomial_dev(vects,batch_size = 128,loss_weights = 1):
    pdb.set_trace()
    x,y = vects
#    x = K.reshape(x,(batch_size,10))
#    y = K.reshape(y,(batch_size,10))
##    vbatch = K.cast(K.shape(x),dtype = "float32")
    #batch_size = K.eval(vbatch)
##    batch_size = int(batch_size.tolist()[0])
    vunit1 = K.l2_normalize(x, axis = 1)
    vunit2 = K.l2_normalize(y, axis = 1)
    kvar3 = K.concatenate([vunit1,vunit2],axis = 0)
    tkvar3 = K.transpose(kvar3)
    vconn = K.dot(kvar3,tkvar3) + K.epsilon()
    vmat =np.ones([2*batch_size,2*batch_size])
    np.fill_diagonal(vmat,0)
    vmat2 = vmat
    vmat2 = 2*vmat/batch_size               
    vmat = -1*vmat               
    for i in range(batch_size):
        if(i%2 == 0):
            vmat[i,i+batch_size] = vmat[i+batch_size,i] = 0
        else:
            vmat[i,i+batch_size] = vmat[i+batch_size,i] = 1      
##    pdb.set_trace()   
    vmat = K.variable(vmat)               
    vmat2 = K.variable(vmat2)               
    #vconn = K.variable(vconn)               
    beta = random.uniform(0, 1)
    alpha = random.uniform(0, 1)
    vcost = K.sum(K.log(-1*alpha*(K.exp((vconn - beta)*vmat)) + 1)*vmat2) + K.epsilon()                    
    vcost = K.reshape(vcost,(1,))
    vcost = K.repeat_elements(vcost, batch_size,axis = 0)
    return vcost
 

def eucl_dist_output_shape(shapes):
    shape1, shape2 = shapes
    return (shape1[0], 1)


def cosine_distance(vects):
    #pdb.set_trace()
    x, y = vects
    vunit1 = K.l2_normalize(x, axis = -1)
    vunit2 = K.l2_normalize(y, axis = -1)
    #return K.sqrt(K.maximum(K.sum(K.square(x - y), axis=1, keepdims=True), K.epsilon()))
    return K.mean(vunit1*vunit2, axis = -1)
    
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
    

def create_pairs(x, digit_indices):
    '''Positive and negative pair creation.
    Alternates between positive and negative pairs.
    '''
    pairs = []
    labels = []
    n = min([len(digit_indices[d]) for d in range(10)]) - 1
    for d in range(10):
        for i in range(n):
            z1, z2 = digit_indices[d][i], digit_indices[d][i + 1]
            pairs += [[x[z1], x[z2]]]
            inc = random.randrange(1, 10)
            dn = (d + inc) % 10
            z1, z2 = digit_indices[d][i], digit_indices[dn][i]
            pairs += [[x[z1], x[z2]]]
            labels += [1, 0]
    return np.array(pairs), np.array(labels)


def create_base_network(input_dim):
    '''Base network to be shared (eq. to feature extraction).
    '''
    seq = Sequential()
    seq.add(Dense(128, input_shape=(input_dim,), activation='relu'))
#    seq.add(Dropout(0.2))
    seq.add(Dense(128, activation='relu'))
#    seq.add(Dropout(0.2))
    seq.add(Dense(10, activation='relu'))
    return seq  


def compute_accuracy(predictions, labels):
    '''Compute classification accuracy with a fixed threshold on distances.
    '''
    return labels[predictions.ravel() < 0.5].mean()


# the data, shuffled and split between train and test sets
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.reshape(60000, 784)
x_test = x_test.reshape(10000, 784)
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255
input_dim = 784
epochs = 1

# create training+test positive and negative pairs
digit_indices = [np.where(y_train == i)[0] for i in range(10)]
tr_pairs, tr_y = create_pairs(x_train, digit_indices)
tr_pairs = tr_pairs[:(108400-112)]
tr_y  = tr_y[:(108400-112)]

digit_indices = [np.where(y_test == i)[0] for i in range(10)]
te_pairs, te_y = create_pairs(x_test, digit_indices)
te_pairs = te_pairs[:(17820-156)]
te_y  = te_y[:(17820-156)]

# network definition
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

#pdb.set_trace()
#vloss = 
distance = Lambda(cosine_distance,output_shape= eucl_dist_output_shape)([processed_a, processed_b])
distance2 = Lambda(binomial_dev,output_shape= eucl_dist_output_shape)([processed_a, processed_b])

#my_loss = binomial_dev([processed_a, processed_b])
model = Model([input_a, input_b], [distance,distance2])


##model.Summary()
# train
rms = RMSprop()
model.compile(loss=[contrastive_loss,binomial_loss],optimizer=rms,loss_weights = [0,1])
model.fit([tr_pairs[:, 0], tr_pairs[:, 1]], [tr_y,tr_y],
          batch_size=128,
          epochs=epochs,
          validation_data=([te_pairs[:, 0], te_pairs[:, 1]], [te_y,te_y]))

#compute final accuracy on training and test sets
pred = model.predict([tr_pairs[0:128, 0], tr_pairs[0:128, 1]],batch_size = 128)
tr_acc = compute_accuracy(pred[0], tr_y[:128])
pred = model.predict([te_pairs[0:128:, 0], te_pairs[0:128:, 1]],batch_size = 128)
te_acc = compute_accuracy(pred[0], te_y[:128])

print('* Accuracy on training set: %0.2f%%' % (100 * tr_acc))
print('* Accuracy on test set: %0.2f%%' % (100 * te_acc))

## TO-DO implement the method stated in keras-function-API
## see output of 10 neurons
