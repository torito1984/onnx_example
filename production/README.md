Generar ficheros de ejemplo

sbt> runMain uimp.mnist.preprocess.MnistPreprocess -i /home/david/projects/uimp_onnx/training/data/mnist/MNIST/raw/t10k-images-idx3-ubyte -l /home/david/projects/uimp_onnx/training/data/mnist/MNIST/raw/t10k-labels-idx1-ubyte -o ./samples/ -r true -n 10

Ejecutar

sbt> runMain uimp.mnist.MNISTPrediction ../training/lenet.onnx samples/_f65181_0_1.png
