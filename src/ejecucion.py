# ejecucion.py


from ocr_pipeline import *



if __name__=='__main__':
	A_opt=pd.read_csv('alfa-letras.csv').values
	captura()
	contraste()
	idx=contorno()
	palabra=interpreta(idx, A_opt).lower()
	print (palabra)
	habla(palabra, leng='es')
	idioma='en'
	traduccion=(traduce(palabra.lower(), leng=idioma))
	print (traduccion)
	habla(traduccion, leng=idioma)










