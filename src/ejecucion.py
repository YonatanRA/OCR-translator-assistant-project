# ejecucion.py


from ocr_pipeline import *



if __name__=='__main__':
	captura()
	contraste()
	idx=contorno()
	#palabra=interpreta_softmax(idx).lower()        # con modelo softmax
	palabra=interpreta_cnn(idx).lower()             # con modelo convolucional
	print (palabra)
	habla(palabra, leng='es')
	idioma='en'
	traduccion=(traduce(palabra.lower(), leng=idioma))
	print (traduccion)
	habla(traduccion, leng=idioma)















