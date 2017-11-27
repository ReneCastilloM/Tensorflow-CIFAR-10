#Importamos todas las librerias que utilizaremos
import numpy as np
import tensorflow as tf
from sklearn.metrics import confusion_matrix
from time import time
from data import get_data_set
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

"""
Como los datos los obtngo de una fuente externa no puedo utilizar la funcion next_batch de tensorflow
Por lo mismo forme mi propia funcion next_batch, la cual te regresa un conjunto aleatorio de registros
"""
def next_batch(num, data, labels):
    '''
    Recibe el numero de registros, el array de inputs y el de outputs
    Regresa un total de `num` inputs y outputs random. 
    '''
    idx = np.arange(0 , len(data))
    np.random.shuffle(idx)
    idx = idx[:num]
    data_shuffle = [data[ i] for i in idx]
    labels_shuffle = [labels[ i] for i in idx]

    return np.asarray(data_shuffle), np.asarray(labels_shuffle)

#Se cargan los datos
print("Cargando los datos...")
train_x, train_y, train_l = get_data_set("train")
test_x, test_y, test_l = get_data_set("test")

#Iniciamos la interactiveSession de Tensorflow
sess = tf.InteractiveSession()
#Creamos los nodos para las entradas y las salidas
#En x son 3072 porque son imagenes a color de 32*32 pixeles
#En y son 10 pr el numero de clases que hay
x = tf.placeholder(tf.float32, shape=[None, 3072])
y_ = tf.placeholder(tf.float32, shape=[None, 10])

#Definimos los pesos
W = tf.Variable(tf.zeros([3072,10]))
b = tf.Variable(tf.zeros([10]))

#funciones para inicializar los pesos
def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)

def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)

#Convolucion y pooling
def conv2d(x, W):
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

#Reacomodamos la imagen en 32x32 con 3 canales de entrada (RGB)
x_image = tf.reshape(x, [-1, 32, 32, 3])

#Primera capa convolucional / La convolucion encontrara 32 caracteristicas para cada zona de 5x5
#La imagen se reducira a una de 16x16 con la funcion max_pool_2x2
W_conv1 = weight_variable([5, 5, 3, 32])
b_conv1 = bias_variable([32])
h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)
#Segunda capa convolucional / La convolucion encontrara 64 caracteristicas por cada zona de 5x5
#La imagen se reducira a una de 8x8 con la funcion max_pool_2x2
W_conv2 = weight_variable([5, 5, 32, 64])
b_conv2 = bias_variable([64])
h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)
#Se agrega una capa fully-conected de 1024 neuronas para procesar la imagen entera
W_fc1 = weight_variable([8 * 8 * 64, 1024])
b_fc1 = bias_variable([1024])
h_pool2_flat = tf.reshape(h_pool2, [-1, 8*8*64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
#Aplicamos el metodo dropout antes de la ultima capa
keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)
#Se agrega una capa final
W_fc2 = weight_variable([1024, 10])
b_fc2 = bias_variable([10])
y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

#Agregamos los pesos a la coleccion vars para utiizarlas desde el test
tf.get_collection("vars")
tf.add_to_collection("vars", W_conv1)
tf.add_to_collection("vars", b_conv1)
tf.add_to_collection("vars", W_conv2)
tf.add_to_collection("vars", b_conv2)
tf.add_to_collection("vars", W_fc1)
tf.add_to_collection("vars", b_fc1)
tf.add_to_collection("vars", W_fc2)
tf.add_to_collection("vars", b_fc2)
#Iniciamos un saver
saver = tf.train.Saver()

print("Datos cargados")

#Entrenamiento y evaluacion del modelo
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

with tf.Session() as sess:
  sess.run(tf.global_variables_initializer()) #Inicializamos las variables
  for i in range(20000):
    x_batch, y_batch =next_batch(50, train_x, train_y) #Aplicamos la funcion batch
    if i % 100 == 0:
      train_accuracy = accuracy.eval(feed_dict={
          x: x_batch, y_: y_batch, keep_prob: 1.0})
      print('step %d, training accuracy %g' % (i, train_accuracy))
    train_step.run(feed_dict={x: x_batch, y_: y_batch, keep_prob: 0.5})

  print('test accuracy %g' % accuracy.eval(feed_dict={
      x: test_x, y_: test_y, keep_prob: 1.0}))
  #Guardamos el modelo
  saver.save(sess, './EasyNet_Trained.ckpt')