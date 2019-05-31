# -*- coding: utf-8 -*-
# clasificador_letras.py
# aprende letras escritas a mano (1vsALL)


import sys
#sys.stdout=open('resultado_optimizacion.txt','wt')  # registro

import time                                     # para tiempo
inicio=time.time()                              # momento inicial
print ('Comenzando...\n')
print ('Modelo SoftMax (MLR) : EMNIST Letras')    
import pandas as pd                             # dataframe
import numpy as np                              # numerical python, algebra lineal
import matplotlib.pyplot as plt                 # plots, graficos
import seaborn as sns                           # plots
from scipy.optimize import minimize             # minimizar, opt
from numba import jit                           # compilacion y paralelizacion, velocidad
from sklearn.metrics import confusion_matrix    # metricas, matriz de confusion
import warnings                                 # avisos
warnings.filterwarnings("ignore", category=RuntimeWarning)  # elimino un warning por valores NaN en logaritmos o /0





# Se crean las funciones de las ecuaciones principales del modelo


def f(X,a):                                 # funcion logistica, sigmoide, funcion del modelo, con z=X*alfa, el producto escalar
    return 1.0/(1.0+np.exp(-np.dot(X,a)))   # Boltzmann con pivote, alfa[i]=0


@jit()
def coste(X,a,y):              # funcion de perdida o coste, funcion a minimizar 
    return -(np.sum(np.log(f(X,a)))+np.dot((y-1).T,(np.dot(X,a))))/y.size+lambda_reg/(2.0*y.size)*np.dot(a[1:],a[1:])

    
@jit()
def grad_coste(X,a,y,lambda_reg):          # gradiente de la funcion de perdida con regularizacion
    return (np.dot(X.T,(f(X,a)-y)))/y.size+lambda_reg/(2.0*y.size)*np.concatenate(([0], a[1:])).T





# Se cargan los datos
datos=pd.read_csv('emnist-byclass-test-fixed.csv')        # imagenes de letras
print ('Datos leidos...')
#print(datos)
matriz_datos=datos.values 
#print (matriz_datos.shape)   # (116322, 785)        
#print(matriz_datos)


n_etiq=int(max([datos['e'][i] for i in range(len(datos))]))   # numero de etiquetas

n_train=int(0.8*len(datos))  # numero registros de entrenamiento



# Creacion de la matriz Y (variable dependiente, a predecir), (onehot)
Y=np.zeros((matriz_datos.shape[0],n_etiq))   
#print (Y.shape)           # (116322, 61)            
for i in range(n_etiq):
	Y[:,i]=np.where(matriz_datos[:,0]==i,1,0)
#print(Y[0:10,:]) # 10 primeras filas	



# Se separan las columnas etiqueta y se quitan las columnas que sean todo ceros
etiquetas=matriz_datos[:,0]        # etiqueta, el numero en si, 42000 etiquetas


X=matriz_datos[:,1:]               # datos numericos de los pixeles, cada columna es un pixel (variables indep)
#print (X.shape)                   # dimension=(116322,784)
X=X[:,X.sum(axis=0)!=0]            # se quitan las columnas=0 (la suma de los elementos es no nulo, no hay informacion)
#print (X.shape)                   # dimension=(116322,749)



# Se dividen los datos en train y test
X_train, Y_train=X[0:n_train,:], Y[0:n_train,:]        # datos de entranamiento
X_test, Y_test=X[n_train:,:], Y[n_train:,:]            # datos de test
#print (X_train.shape, Y_train.shape)                  # dimensiones==> X_train(93057,749), Y_train(93057,61)
#print (X_test.shape, Y_test.shape)                    # dimensiones==> X_test(23265,708), Y_test(23265,61)



# etiquetas train y test
etiquetas_train=etiquetas[0:n_train]       # etiquetas para entranamiento
etiquetas_test=etiquetas[n_train:]         # etiquetas para test





@jit()
def normalizador(X):                     # normalizador de X
    X_media=X.mean(axis=0)               # media de X
    X_std=X.std(axis=0)                  # desviacion estandar de X
    X_std[X_std==0]=1.0                  # si hay alguna std=0 ponla a 1
    X=(X-X_media)/X_std                  # normaliza
    X=np.insert(X,0,1, axis=1)           # esta linea añade una columna de 1, feature engineering [1, f1, f2....., fn] (mejora un 10%)
    return X




X_train=normalizador(X_train)
X_test=normalizador(X_test)
#X=normalizador(X)
print ('Datos normalizados.')



# Se buscan los parametros optimos para los modelos

val_inicial=np.random.rand(X_train.shape[1])     # valores iniciales de los parametros alfa

A_opt=np.zeros((X_train.shape[1],n_etiq))        # se crea la matriz para los parametros optimizados, alfas


lambda_reg=100.            # valor obtenido del gridsearching  


inicio_opt=time.time()                       # inicio optimizacion
for i in range(n_etiq):
	print ('\n\nOptimizando {} frente al resto'.format(i))
	
	def opt_coste(a):                        # funcion a minimizar, para ello solo tiene un input (val_inicial)
		return coste(X_train, a, Y_train[:,i]) 
	
	def opt_grad_coste(a):                   # gradiente 
		return grad_coste(X_train, a, Y_train[:,i], lambda_reg)	
		
	# metodo Nelder-Mead, Powell, CG, BFGS, Newton-CG, L-BFGS-B, TNC, COBYLA, SLSQP, trust-constr, 
	# dogleg, trust-ncg, trust-exact, trust-krylov (tambien custom)            
	metodo='trust-constr'
	print ('Optimizacion {}...'.format(metodo)) # minimizacion, optimizacion	
	i_opt=time.time()                           # inicio de cada optimizacion
	modelo=minimize(opt_coste, val_inicial, method=metodo, jac=opt_grad_coste, hess=None, tol=1e-6, options={'disp':True})
	print ('Hecho.')
	print ("Tiempo optimizacion: {:.2f} segundos" .format(time.time()-i_opt))  
	A_opt[:,i]=modelo.x  # se guardan los parametros obtenidos
	
print ("\nTiempo total optimizacion: {:.2f} segundos\n" .format(time.time()-inicio_opt))  # tiempo desde inicio hasta final minimizacion



# Ahora se chequea el modelo en cada paquete de datos
y_pred=[]                  # etiquetas predichas
y_prob=[]                  # probabilidades de las etiquetas predichas

def resumen(datos):        # testeo
	for e in datos:
		nombre, etiqueta, Xs=e         
		etiq=etiqueta.size
		probs=np.zeros((n_etiq,2))    # etiquetas con su probabilidad
		cuenta=0                      # conteo de aciertos
		for muestra in range(etiq): 
			for n in range(n_etiq):
				alfa=A_opt[:,n]       # parametros de softmax
				probs[n,0]=n
				probs[n,1]=f(Xs[muestra,:],alfa)      # evaluacion de la prediccion
				
			probs=probs[probs[:,1].argsort()[::-1]]	  # se pone la prob mas alta al principio
			y_pred.append(probs[0,0])
			y_prob.append(probs[0,1])
			if probs[0,0]==etiqueta[muestra]:         # si se acierta +1
				cuenta+=1
		print ("\n{}".format(nombre))
		print ("{} correctos de {} ==> {:.4}% correcto".format(cuenta, etiq, cuenta/etiq*100))
		
		
	
# Resultados train, validacion y test	
resumen([('Entranamiento  :', etiquetas_train, X_train)])
resumen([('Test  :', etiquetas_test, X_test)])	
print ('')
print ("Tiempo final : {:.2f} segundos".format(time.time()-inicio))
print ('')
print ('')


confusion=confusion_matrix(etiquetas_test, y_pred[n_train:])
sns.heatmap((confusion/len(y_pred)*100),annot=False)         # plot matriz confusion
plt.xlabel('Prediction')
plt.ylabel('Truth')
plt.title('Confusion Matrix (%)')
plt.savefig('Confusion Matrix.png')
plt.show()



#print (A_opt)	
print ('Dimensiones matriz de parametros={}'.format(A_opt.shape))		
df=pd.DataFrame(A_opt, columns=[i+1 for i in range(A_opt.shape[1])])  # se guardan los parametros softmax en csv
df.to_csv('alfas.csv', index=False)
		
		
		
		
		
		
		
	
