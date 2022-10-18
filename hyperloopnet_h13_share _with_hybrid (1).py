# Import Relevant libraries and classes
import scipy.io as sio
import numpy as np
import tqdm
from sklearn.decomposition import PCA
import tensorflow as tf
keras = tf.keras
from keras import backend as K
from keras import regularizers
from keras.models import Model, Sequential
from keras.layers import Dense, Input, Dropout
from keras.layers import Conv2D, Flatten, Lambda, Conv3D, Conv3DTranspose,BatchNormalization,Conv1D, Activation
from keras.layers import Reshape, Conv2DTranspose, Concatenate, Multiply, Add, MaxPooling2D, MaxPooling3D, GlobalAveragePooling2D, GlobalMaxPooling2D
from keras import Model
from keras.datasets import mnist
from keras.losses import mse, binary_crossentropy
#from keras.utils import plot_model
from keras import backend as K
from sklearn.metrics import confusion_matrix

train_patches = np.load('/content/drive/MyDrive/train_patches.npy')
test_patches = np.load('/content/drive/MyDrive/test_patches.npy')

train_labels = np.load('/content/drive/MyDrive/train_labels.npy')-1
test_labels = np.load('/content/drive/MyDrive/test_labels.npy')-1

tr90 = np.empty([2832,11,11,144], dtype = 'float32')
tr180 = np.empty([2832,11,11,144], dtype = 'float32')
#tr270 = np.empty([2832,11,11,145], dtype = 'float32')

for i in tqdm.tqdm(range(2832)):
  tr90[i,:,:,:] = np.rot90(train_patches[i,:,:,:])
  tr180[i,:,:,:] = np.rot90(tr90[i,:,:,:])
#  tr270[i,:,:,:] = np.rot90(tr180[i,:,:,:])

train_patches = np.concatenate([train_patches, tr180], axis = 0)
train_labels = np.concatenate([train_labels,train_labels], axis = 0)

from sklearn.utils import shuffle
train_patches, train_labels = shuffle(train_patches, train_labels, random_state=0)

print(np.shape(train_patches))
print(np.shape(test_patches))

train_vec = np.reshape(train_patches[:,5,5,0:144],[-1,1,144,1])
test_vec = np.reshape(test_patches[:,5,5,0:144],[-1,1,144,1])

lid_train = train_patches[:,:,:,144:145]
lid_test = test_patches[:,:,:,144:145]

def channel_attention(t,fil):
  
 
  channels = t.shape[3]

  


  denseLay  = Dense(channels, activation = 'relu')

  maxPool = GlobalMaxPooling2D()
  avgPool = tf.keras.layers.GlobalAveragePooling2D()

  

  maxPooled = maxPool(t)
  avgPooled = avgPool(t)

  

  maxOut = Reshape((1,1,channels))(maxPooled)
  avgOut = Reshape((1,1,channels))(avgPooled)

  maxConv = Conv2D(filters=fil, kernel_size=1,  padding = 'valid', activation = 'relu')(maxOut)
  avgConv = Conv2D(filters=fil, kernel_size=1,  padding = 'valid', activation = 'relu')(avgOut)

  maxConc = denseLay(maxConv)
  avgConc = denseLay(avgConv)

  concatenated = tf.math.add(
      maxConc, avgConc, name=None
  )

  final = tf.math.multiply(
      t, concatenated, name=None
  )

  return final

def spatial_attention(t):

  


  maxPooled = tf.reduce_max(t, axis = 3)
  avgPooled = tf.reduce_mean(t, axis = 3)

  maxOut = Reshape((t.shape[1],t.shape[2],1))(maxPooled)
  avgOut = Reshape((t.shape[1],t.shape[2],1))(avgPooled)
 

  concat = tf.keras.layers.Concatenate(axis=3)([maxOut, avgOut])
  convolved = Conv2D(filters=1, kernel_size=7,  padding = 'same', activation = 'relu')(concat)

  final = tf.math.multiply(
      t, convolved, name=None
  )

  return final

def hybrid_attention(x,fil):
  channelAtt = channel_attention(x,fil)
  final = spatial_attention(channelAtt)
  return final




  


def cross_atac(x,h,l):
  
  a = Multiply()([x,h])
  b = Multiply()([x,l])
  ad1 = Concatenate(axis = 3)([a,b,x])
  return ad1

def my_conv(x,l):

  c1 = l(x)
  c1 = BatchNormalization()(c1)
  
  return c1
def my_conv_with_hybrid(x,l):
  x = hybrid_attention(x,16)
  c1 = l(x)
  c1 = BatchNormalization()(c1)
  return c1

def block(x,k):

  fil = 32
  #k = 3

  x2 = Conv2D(filters=fil, kernel_size=1,  padding = 'valid', 
                       activation = 'relu')(x)

  l1 = Conv2D(filters=fil, kernel_size=k,  padding = 'same', 
                       activation = 'relu')
  l2 = Conv2D(filters=fil, kernel_size=k,  padding = 'same', 
                       activation = 'relu')
  l3 = Conv2D(filters=fil, kernel_size=k,  padding = 'same', 
                       activation = 'relu')
  l4 = Conv2D(filters=fil, kernel_size=k,  padding = 'same', 
                       activation = 'relu')


  # Stage 1
  
  cv1 = my_conv(x2, l1)
  c1 = Add()([x2,cv1])
  cv2 = my_conv(c1, l2)
  c2 = Add()([x2,cv1,cv2])
  cv3 = my_conv(c2, l3)  
  c3 = Add()([x2,cv1,cv2,cv3])
  cv4 = my_conv_with_hybrid(c3, l4)  

  # Stage 2

  c4 = Add()([cv2,cv3,cv4])
  cv1 = my_conv(c4, l1)
  c5 = Add()([cv1,cv3,cv4])
  cv2 = my_conv(c5, l2)
  c6 = Add()([cv1,cv2,cv4])
  cv3 = my_conv(c6, l3)
  c7 = Add()([cv1,cv2,cv3])
  cv4 = my_conv(c7, l4)

  conc1 = Concatenate(axis = 3)([cv1,cv2,cv3, cv4])
  gap1 = GlobalAveragePooling2D()(conc1)

  return conc1, gap1

def ext(x):

  conc1, gap1 = block(x,3) 
  conc2, gap2 = block(x,5) 
  conc3, gap3 = block(x,7)  

  gp = Concatenate(axis = 1)([gap1, gap2, gap3])
  
  c6 =  Dense(15, activation = 'softmax')(gp)
  return Reshape([15])(c6)

x = Input(shape=(11,11,144), name='inputA')

outfinal = ext(x)

optim = keras.optimizers.Nadam(0.0002) 

model = Model(x, outfinal, name = 'model')

# Compiling the model
model.compile(loss='categorical_crossentropy', optimizer=optim, metrics=['accuracy'])

model.summary()

keras.utils.plot_model(model)

for p in range(5):

  x = Input(shape=(11,11,144), name='inputA')

  outfinal = ext(x)

  optim = keras.optimizers.Nadam(0.0002) 

  model = Model(x, outfinal, name = 'model')

  # Compiling the model
  model.compile(loss='categorical_crossentropy', optimizer=optim, metrics=['accuracy'])
  ep = 0
  k=0
  import gc
  for epoch in range(500): 
    gc.collect()
    model.fit(x = train_patches[:,:,:,0:144],
                    y = my_ohc(np.expand_dims(train_labels, axis = 1)),
                    epochs=1, batch_size = 64, verbose = 1)
    
    preds2 = model.predict(test_patches[:,:,:,0:144], batch_size = 64, verbose = 2) 

    conf = confusion_matrix(test_labels, np.argmax(preds2,1))
    ovr_acc, _, _, _, _ = accuracies(conf)

    print(epoch)
    print(np.round(100*ovr_acc,2))
    if ovr_acc>=k:
      model.save('/content/gdrive/My Drive/Projects/CliqueNet/H13/ablation/cn_share_ms_ab16/model'+str(p))
      k = ovr_acc
      ep = epoch
      np.save('/content/gdrive/My Drive/Projects/CliqueNet/H13/ablation/cn_share_ms_ab16/ep',epoch)
    print('acc_max = ', np.round(100*k,2), '% at epoch', ep)

    if epoch%5==0:
      model.save('/content/gdrive/My Drive/Projects/CliqueNet/H13/ablation/cn_share_ms_ab16/model_temp')
      np.save('/content/gdrive/My Drive/Projects/CliqueNet/H13/ablation/cn_share_ms_ab16/ep_temp',epoch)

preds = np.zeros([12197,5])
conf = np.zeros([15,15,5])
for i in range(5):
  model = keras.models.load_model('/content/gdrive/My Drive/Projects/CliqueNet/H13/ablation/cn_relu_1_0/model'+str(i))
  preds[:,i] = np.argmax(model.predict(test_patches[:,:,:,0:144]),1)
  conf[:,:,i] = confusion_matrix(test_labels, preds[:,i])
  tr = np.trace(conf[:,:,i])
  ss = np.sum(conf[:,:,i])
  print(np.round(100*tr/ss,5))

cm_s = np.mean(conf,2)
tr = np.trace(cm_s)
ss = np.sum(cm_s)
print(np.round(100*tr/ss,2))
