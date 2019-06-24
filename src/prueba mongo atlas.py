# prueba mongo atlas.py



import pymongo


def token(token):         
    with open(token, 'r') as f:
        t=f.readlines()[0].split('\n')[0]
    return t


cliente=pymongo.MongoClient('mongodb+srv://Yonatan:{}@mambacluster-v9uol.mongodb.net/test?retryWrites=true&w=majority'.format(token('mongoatlas.txt')))
db=cliente.test

try:
    print('MongoDB version is %s' % cliente.server_info()['version'])
except pymongo.errors.OperationFailure as error:
    print(error)
    quit(1)


original=db.original
traduccion=db.traduccion

cursor=original.find()
cursor2=traduccion.find()


#original.insert_one({'palabra':'uva'})
#traduccion.insert_one({'palabra':'grapes'})
for item in cursor:
    print(item['palabra'])

for item2 in cursor2:
    print(item2['palabra'])
'''
cont=0
for item in cursor:
    if item['palabra']=='hola': cont+=1
if cont==0: original.insert_one({'palabra':'hola'})



cont2=0
for item2 in cursor2:
    if item2['palabra']=='hello': cont2+=1
if cont2==0: traduccion.insert_one({'palabra':'hello'})
'''





















