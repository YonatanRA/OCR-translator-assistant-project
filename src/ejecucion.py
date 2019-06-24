# ejecucion.py


from ocr_pipeline import *



if __name__=='__main__':    # inicializa el asistente
	while 1:
		trigger_word=activacion()     # palabra activacion
		if trigger_word=='escucha':
			trigger.update_one({"a":'0'}, {"$set":{"a": "1"}})
			habla('Hola. Cuentame.')
			trigger.update_one({"a":'1'}, {"$set":{"a": "0"}})
			while 1:
				datos=escucha()
				flag=mamba(datos)
				if flag: break





