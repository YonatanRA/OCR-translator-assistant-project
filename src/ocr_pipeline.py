# ocr_pipeline.py

import cv2               # libreria OpenCV para vision por computador
import numpy as np       # numerical python






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
			
	cv2.imshow('img', img)                          # muestra la imagen
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
	cv2.imshow('imagen B-N', im)
	cv2.waitKey(1)
	return idx





if __name__=='__main__':
	captura()
	contraste()
	idx=contorno()
	print (idx)




