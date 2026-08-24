Coloque aqui o modelo TensorFlow Lite com o nome model.tflite.
Opcionalmente coloque labels.txt com uma classe por linha.

O runtime lê a forma do tensor de entrada diretamente do modelo. O pipeline atual espera entrada RGB [1,height,width,3] e saída YOLO em [1,N,attributes] ou [1,attributes,N].
Se o modelo usar outra codificação de saída, o parser deverá ser ajustado com base nos metadados reais do modelo; não assumir classes, quantização ou sigmoid sem verificar o modelo.
