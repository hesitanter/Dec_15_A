
% https://www.mathworks.com/help/deeplearning/ug/define-custom-layer-with-multiple-inputs.html
clear;
[XTrain,YTrain] = digitTrain4DArrayData;
layers = [
    imageInputLayer([28 28 1],'Name','in')
    convolution2dLayer(5,20,'Name','conv1')
    reluLayer('Name','relu1')
    convolution2dLayer(3,20,'Padding',1,'Name','conv2')
    reluLayer('Name','relu2')
    convolution2dLayer(3,20,'Padding',1,'Name','conv3')
    reluLayer('Name','relu3')
    weightedAdditionLayer(2,'add')
    fullyConnectedLayer(10,'Name','fc')
    softmaxLayer('Name','softmax')
    classificationLayer('Name','classoutput')];

lgraph = layerGraph(layers);
lgraph = connectLayers(lgraph, 'relu1', 'add/in2');
options = trainingOptions('adam','MaxEpochs',10);
net = trainNetwork(XTrain,YTrain,lgraph,options);


layer = recurrent_layer(4, "rnn");
validInputSize  = [2 4 1];
checkLayer(layer,validInputSize ,'ObservationDimension',2)






