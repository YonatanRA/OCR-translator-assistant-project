import numpy as np
import pyqtgraph.opengl as gl
from pyqtgraph.Qt import QtCore, QtGui
import struct
import pyaudio
import sys
import requests
import pymongo

def token(token):                # lectura de txt, token
    with open(token, 'r') as f:
        t=f.readlines()[0].split('\n')[0]
    return t
cliente=pymongo.MongoClient('mongodb+srv://Yonatan:{}@mambacluster-v9uol.mongodb.net/test?retryWrites=true&w=majority'.format(token('mongoatlas.txt')))
db=cliente.test
trigger=db.trigger
cursor3=trigger.find()


class Voice():
    def __init__(self, trigger=0):  # inicio

        # configura la ventana de visualizacion
        self.app=QtGui.QApplication(sys.argv)
        self.window=gl.GLViewWidget()
        self.window.setWindowTitle('Mamba')
        self.window.setGeometry(0, 110, 1920, 1080)
        self.window.setCameraPosition(distance=65, elevation=25)
        self.window.show()

        # constantes y arrays
        self.nsteps=1.3
        self.offset=0
        self.ypoints=np.arange(-20, 20 + self.nsteps, self.nsteps)
        self.xpoints=np.arange(-20, 20 + self.nsteps, self.nsteps)
        self.nfaces=len(self.ypoints)

        self.RATE=44100    # frecuencia de muestreo
        self.CHUNK=len(self.xpoints)*len(self.ypoints)

        self.p=pyaudio.PyAudio()    # para audio
        self.trigger=trigger        # disparador
        self.stream=self.p.open(
            format=pyaudio.paInt16,
            channels=1,
            frames_per_buffer=self.CHUNK,
            rate=self.RATE,
            input=True,
            output=True
        ) # captura de audio microfono



        # crea array de vertices
        verts, faces, colors=self.mesh()

        self.mesh1=gl.GLMeshItem(
            faces=faces,
            vertexes=verts,
            faceColors=colors,
            drawEdges=True,
            smooth=False,
        )
        self.mesh1.setGLOptions('additive')
        self.window.addItem(self.mesh1)

    def mesh(self, offset=0, height=2.5, wf_data=None):   # malla

        if wf_data is not None:
            wf_data=struct.unpack(str(2 * self.CHUNK) + 'B', wf_data)
            wf_data=np.array(wf_data, dtype='b')[::2] + 128
            wf_data=np.array(wf_data, dtype='int32') - 128
            wf_data=wf_data * 0.04
            wf_data=wf_data.reshape((len(self.xpoints), len(self.ypoints)))
        else:
            wf_data=np.array([1] * 1024)
            wf_data=wf_data.reshape((len(self.xpoints), len(self.ypoints)))
            
 
        amp=np.frombuffer((self.stream.read(self.CHUNK)),dtype=np.int16)[0] if self.trigger==1 else 0  # amplitud de la malla

        faces=[]
        colors=[]
        verts=np.array([[x, y, wf_data[xid][yid] * amp/2000] for xid, x in enumerate(self.xpoints) for yid, y in enumerate(self.ypoints)], 
                         dtype=np.float32)
        
        for yid in range(self.nfaces-1):
            yoff=yid*self.nfaces
            for xid in range(self.nfaces-1):
                faces.append([
                    xid + yoff,
                    xid + yoff + self.nfaces,
                    xid + yoff + self.nfaces + 1,
                ])
                faces.append([
                    xid + yoff,
                    xid + yoff + 1,
                    xid + yoff + self.nfaces + 1,
                ])
                colors.append([
                    xid / self.nfaces, 1 - xid / self.nfaces, yid / self.nfaces, 0.7
                ])
                colors.append([
                    xid / self.nfaces, 1 - xid / self.nfaces, yid / self.nfaces, 0.8
                ])

        faces=np.array(faces, dtype=np.uint32)
        colors=np.array(colors, dtype=np.float32)

        return verts, faces, colors

    def update(self): # actualiza malla
		
        wf_data=self.stream.read(self.CHUNK)
        self.trigger=int([e['a'] for e in cursor3][0])
        cursor3.rewind()
        verts, faces, colors=self.mesh(offset=self.offset, wf_data=wf_data)
        self.mesh1.setMeshData(vertexes=verts, faces=faces, faceColors=colors)
        self.offset-=0.05
        


    def start(self):  # empieza el dibujo
        if (sys.flags.interactive != 1) or not hasattr(QtCore, 'PYQT_VERSION'):
            QtGui.QApplication.instance().exec_()

    def animation(self, frametime=10): # llama al update en bucle
        timer= QtCore.QTimer()
        timer.timeout.connect(self.update)
        timer.start(frametime)
        self.start()


if __name__ == '__main__':
	voz=Voice().animation()

