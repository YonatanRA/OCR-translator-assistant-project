# ejecucion.py


from ocr_pipeline import *



if __name__=='__main__':    # inicializa el asistente
	while 1:
		trigger=activacion()     # palabra activacion
		if trigger=='escucha':
			habla('Hola. Soy Mamba, ¿que puedo hacer por ti?')
			while 1:
				datos=escucha()
				flag=mamba(datos)
				if flag: break





