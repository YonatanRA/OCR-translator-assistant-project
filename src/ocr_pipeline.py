# ocr_pipeline.py

import cv2                           # libreria OpenCV para vision por computador
from PIL import Image                # para imagen
import numpy as np                   # numerical python
import pandas as pd                  # dataframe (datos parametros)
from google_speech import Speech     # para hablar
from googletrans import Translator   # para traducir




def captura():                                      # funcion captura video por webcam (para sacar foto)
	cam=cv2.VideoCapture(0)                         # inicia captura
	while True:           
		ret, frame=cam.read()                       # lee la camara frame a frame
		cv2.imshow("Captura de imagen", frame)      # muestra imagen por pantalla con nombre
		if ret==False:                              # ret=retorno (booleano)
			break
		key=cv2.waitKey(1)                          # espera por OnClick de una tecla
		if key%256==27:                             # presiona escape para salir
			break
		elif key%256==32:                           # presiona espacio para capturar
			img_name="captura.png"                  # nombra png
			cv2.imwrite(img_name, frame)            # guarda la imagen
			print("{} guardada".format(img_name))
			break
	cam.release()                                   # cierra la pantalla de captura
	cv2.destroyAllWindows()                         # destruye todas las ventanas de imagen






def contraste():                                    # funcion para pasar imagen a blanco y negro 
	image=cv2.imread('captura.png')                 # lee la captura de imagen         
	im=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)       # pasa a escala de grises
	im=(255-im)                                     # foto en negativo
	umbral=170
	img=np.zeros(shape=im.shape)                    # pasa a blanco y negro puros (umbral en 170)
	for i in range(im.shape[0]):
		for j in range(im.shape[1]):
			if im[i][j]>umbral: img[i][j]=255
			else: img[i][j]=0
	#cv2.imshow('img', img)                          # muestra la imagen
	cv2.imwrite('b&w.png', img)                     # guarda imagen en blanco y negro
	cv2.waitKey(1)                                  # espera por tecla (si se comenta esta linea no se muestra la imagen)                





def contorno():                                 # funcion captura de contorno, captura letras
	umbral_fino=10                              # umbral fino para deteccion de contornos
	image=cv2.imread('b&w.png')                 # lee la imagen
	im=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)   # pasa a grises
	im=(255-im)                                 # pasa a negativo
	thresh=cv2.adaptiveThreshold(im, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)      # umbral adaptativo
	rect_kernel=cv2.getStructuringElement(cv2.MORPH_RECT, (30, 10))                                          # elemento estructural (rectangular) 
	threshed=cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, rect_kernel)                                          # transformacion morfologica
	contorno, _ =cv2.findContours(threshed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)                      # encuentra los contornos
	s_contorno=sorted(contorno, key=lambda x: cv2.boundingRect(x)[1]+cv2.boundingRect(x)[0]*image.shape[1])  # ordena por orden lateral (y+xh)
	idx=0
	for cnt in s_contorno:                                   # contornos
		idx+=1
		x, y, w, h=cv2.boundingRect(cnt)
		roi=im[y:y+h, x:x+w]                                 # region de interes
		if h<umbral_fino or w<umbral_fino:
			continue
		cv2.imwrite(str(idx) + '.png', roi)
		cv2.rectangle(im, (x, y), (x+w, y+h), (200, 0, 0), 2)
	#cv2.imshow('imagen B-N', im)
	cv2.waitKey(1)
	return idx




def normalizador(X):               # normalizador de los datos de letra
    X_media=X.mean()               
    X_std=X.std()                  
    X=(X-X_media)/X_std            
    X=np.insert(X,0,1)             
    return X



def f(X,a):                                 # funcion logistica, sigmoide, funcion del modelo, con z=X*alfa, el producto escalar
    return 1.0/(1.0+np.exp(-np.dot(X,a)))   # Boltzmann con pivote, alfa[i]=0




def interpreta(idx, A_opt):
	resultado=''
	alfabeto={1:'A', 2:'B', 3:'C', 4:'D', 5:'E', 6:'F', 7:'G', 8:'H', 9:'I', 10:'J',
	          11:'K', 12:'L', 13:'M', 14:'N', 15:'O', 16:'P', 17:'Q', 18:'R', 19:'S', 20:'T',
	          21:'U', 22:'V', 23:'W', 24:'X', 25:'Y', 26:'Z'}
	          
	for i in range(idx):
		nombre=str(i+1)+'.png'           # nombre de la imagen
		img=cv2.imread(nombre)           # lee imagen
		shape=img.shape                  # creo un marco blanco, zoom out
		w=shape[1]
		h=shape[0]
		ancho_marco=h+20,w+20,3                                
		datos=np.zeros(ancho_marco,dtype=np.uint8)
		cv2.rectangle(datos,(0,0),(w+20,h+20),(255,255,255),30)
		datos[10:h+10,10:w+10]=img 
		datos=cv2.cvtColor(datos,cv2.COLOR_BGR2GRAY)
		datos=np.array(Image.fromarray(datos).resize((28,28)))
		#datos=misc.imresize(datos, (28, 28))
		datos=(255-datos)
		datos=datos.reshape(784,)
		datos=normalizador(datos)
		n_etiq=A_opt.shape[1]
		probs=np.zeros((n_etiq,2)) 
		for n in range(n_etiq):
			alfa=A_opt[:,n]               # parametros de softmax
			probs[n,0]=n                  # evaluacion de la prediccion
			probs[n,1]=f(datos,alfa)      
		probs=probs[probs[:,1].argsort()[::-1]]	 # primero el de mas probabilidad
		etiqueta=int(probs[0,0])                 # etiqueta de la letra
		resultado+=alfabeto[etiqueta]

	return resultado




def habla(texto, leng='es'):
	voz=Speech(texto, leng)
	return voz.play()





def traduce(texto, leng='en'):
	traductor=Translator()
	traduccion=traductor.translate(texto, dest=leng).text
	pronunciacion=traductor.translate(texto, dest=leng).pronunciation
	return traduccion



