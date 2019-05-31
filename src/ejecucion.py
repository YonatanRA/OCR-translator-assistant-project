# ejecucion.py


from ocr_pipeline import *



if __name__=='__main__':
	A_opt=pd.read_csv('alfa-letras.csv').values
	captura()
	contraste()
	idx=contorno()
	palabra=interpreta(idx, A_opt)
	print (palabra)
	habla(palabra, lang='es')











