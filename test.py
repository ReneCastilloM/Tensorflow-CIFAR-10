#Importamos las librerias necesarias
import numpy as np
import tensorflow as tf
from sklearn.metrics import confusion_matrix
from time import time
from data import get_data_set
import random
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

#Iniciamos el reloj
tic = time()

#Cargamos los datos
print("Cargando los datos...")
train_x, train_y, train_l = get_data_set("train")
test_x, test_y, test_l = get_data_set("test")
print("Datos cargados\n")

#Iniciamos la session de tensorflow
sess = tf.Session()
#Importamos el modelo
new_saver = tf.train.import_meta_graph('EasyNet_Trained.ckpt.meta')
new_saver.restore(sess, tf.train.latest_checkpoint('./'))
#Accesamos a la coleccion vars
all_vars = tf.get_collection('vars')
w_conv1 = sess.run(all_vars[0])
b_conv1 = sess.run(all_vars[1])
w_conv2 = sess.run(all_vars[2])
b_conv2 = sess.run(all_vars[3])
w_fc1 = sess.run(all_vars[4])
b_fc1 = sess.run(all_vars[5])
w_fc2 = sess.run(all_vars[6])
b_fc2 = sess.run(all_vars[7])

#Elegimos un numero aleatorio entre 0 y 10000 y obtenemos los inputs y outputs de ese numero de registro
indice = random.randint(0, 10000)
registro = test_x[indice]
respuesta_correcta = test_y[indice]

"""
  Aplicamos el mismo proceso que en el train solo que sin evaluar con las variables obtenidas de la coleccion vars
"""

def conv2d(x, W):
  return tf.nn.conv2d(tf.cast(x, tf.float32), tf.cast(W, tf.float32), strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

x_image = tf.reshape(registro, [-1, 32, 32, 3])

h_conv1 = tf.nn.relu(conv2d(x_image, w_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)

h_conv2 = tf.nn.relu(conv2d(h_pool1, w_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)

h_pool2_flat = tf.reshape(h_pool2, [-1, 8*8*64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, w_fc1) + b_fc1)

h_fc1_drop = tf.nn.dropout(h_fc1, 0.5)

y_conv = tf.matmul(h_fc1_drop, w_fc2) + b_fc2

#Paramos el reloj
tac = time() - tic

print("\nEl registro aleatorio del test elegido fue el " + str(indice))

#iniciamos las variables
model = tf.global_variables_initializer()

with tf.Session() as session:
    session.run(model)
    print("\nLa clase que se predice es:")
    #Predecimos los valores para este caso
    print(session.run(y_conv))

#Imprimimos los datos correctos
print("\nLa clase correcta es:")
print(respuesta_correcta)
print("\nEl tiempo requerido fue de " + str(round(tac, 2)) + " segundos")