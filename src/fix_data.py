# -*- coding: utf-8 -*-
# fix_data.py
# flipea y rota los numeros para crear nuevo dataset


import pandas as pd                    # dataframe
import numpy as np                     # numerical python, algebra lineal
import matplotlib.pyplot as plt        # plots, graficos




# Se cargan los datos
print ('Leyendo datos...')
datos=pd.read_csv('emnist-byclass-test.csv')        # imagenes de letras
print ('Datos leidos.')
cols=['e']
for i in range(784):
	cols.append(i+1)
datos.columns=cols
#print (datos.columns)
#print(datos)
matriz_datos=datos.values 
print (matriz_datos.shape)           
#print(matriz_datos)




'''
plt.figure(figsize=(5,5))
for i in range(5):
	plt.figure(i+1)
	imagen=matriz_datos[i+13000,1:].reshape(28,28)
	plt.imshow(imagen) 
plt.show()
'''


matriz_datos_fix=np.zeros(shape=matriz_datos.shape)
for i in range(len(datos)):
	print (i)
	matriz_datos_fix[i,0]=matriz_datos[i,0]
	imagen=matriz_datos[i,1:].reshape(28,28)
	imagen=np.rot90(imagen, 3)
	imagen=np.flip(imagen, axis=1)
	matriz_datos_fix[i,1:]=imagen.reshape(1,784)


df=pd.DataFrame(matriz_datos_fix, columns=datos.columns)
df.to_csv('emnist-byclass-test-fixed.csv', index=False)
print ('\nFixed.')

print (max([datos['e'][i] for i in range(len(datos))]))
print (datos.head())

plt.figure(figsize=(5,5))
for i in range(5):
	plt.figure(i+1)
	imagen=matriz_datos_fix[i+13000,1:].reshape(28,28)
	plt.title(matriz_datos_fix[i+13000,0])
	plt.imshow(imagen) 
plt.show()






