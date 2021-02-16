Ejemplo de exportación de modelos de Pytorch a ONNX

Instalación

Instalar las dependencias con PIP

$ pip install -r requirements.txt

Uso

Arrancar el servidor de visdom para obsevar la curva de error en http://localhost:8097

$ python -m visdom.server

Entrenar el modelo

$ python run.py


El modelo entrenado se serializa a ONNX en el fichero lenet.onnx. La arquitectura del grafo de inferencia de lenet.onnx puede ser observado en Netron (https://github.com/lutzroeder/netron).

Credits: https://github.com/activatedgeek/LeNet-5
