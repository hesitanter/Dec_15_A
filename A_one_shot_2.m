


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% train with possitive and negative data
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clear;
str = 'data_file_train_591.txt';
[data, time] = load_data(str);        % load data, standardization
[data, time] = sliding_mean(data, time);
[p, n] = generate_p_n(data);
[data, label] = get_data_label(p, n);

plots = "training-progress";
plotRatio = 16/9;
if plots == "training-progress"
    trainingPlot = figure;
    trainingPlot.Position(3) = plotRatio*trainingPlot.Position(4);
    trainingPlot.Visible = 'on';
    
    trainingPlotAxes = gca;
    
    lineLossTrain = animatedline(trainingPlotAxes);
    xlabel(trainingPlotAxes,"Iteration")
    ylabel(trainingPlotAxes,"Loss")
    title(trainingPlotAxes,"Loss During Training")
end

inputSize = 2;
layers = [...
          sequenceInputLayer(8, 'Name', 'Input')
          recurrent_layer(1, 'rnn')];
lgraph = layerGraph(layers);
dlnet = dlnetwork(lgraph);
fcWeights = dlarray(0.01*randn(1,4));
fcBias = dlarray(0.01*randn(1,1));
fcParams = struct(...
    "FcWeights",fcWeights,...
    "FcBias",fcBias);

vel_1 = [];
vel_2 = [];
for i = 1:size(data, 2)
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % dlarray format ?????????
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    dlx1 = dlarray(single(data(1:8, i)), 'CT');
    dlx2 = dlarray(single(data(9:16, i)), 'CT');
    [gradientsSubnet, gradientsParams,loss] = dlfeval(@modelGradients,dlnet,fcParams,dlx1,dlx2,label(i, 1));
    lossValue = double(gather(extractdata(loss)));
    % updata network parameters
    [dlnet, vel_1] = sgdmupdate(dlnet,gradientsSubnet,vel_1);
    [fcParams, vel_2] = sgdmupdate(fcParams,gradientsParams,vel_2);
    
    % Update the training loss progress plot.
    if plots == "training-progress"
        addpoints(lineLossTrain,i,lossValue);
    end
    drawnow;
end





function [data, time] = load_data(str)
y = load(str);
pre_data = y(:, 1:3);
time = y(:, 4:9);
data = mapminmax(pre_data(:, 1:2)')';
data(:, 3) = pre_data(:, 3);
end

function [x, y] = generate_p_n(data)
n = size(data, 1);
count_x = 1;
count_y = 1;
for i = 4:n
   negative = length(find(data(i-3:i, 3) == -1));
   positive = length(find(data(i-3:i, 3) == 1));
   
   if positive == 4
      x(:, :, count_x) = data(i-3:i, :);
      vec(count_x) = i;
      count_x = count_x + 1;
   end
   if negative == 4
      y(:, :, count_y) = data(i-3:i, :);
      count_y = count_y + 1;
   end
end





end

function Y = forwardnet(dlnet,fcParams,d1x1,d1x2)
F1 = forward(dlnet, d1x1);
F1 = sigmoid(F1);

F2 = forward(dlnet, d1x2);
F2 = sigmoid(F2);

Y = abs(F1-F2);
Y = fullyconnect(Y, fcParams.FcWeights, fcParams.FcBias);
Y = sigmoid(Y);
end

function [gradientsSubnet,gradientsParams,loss] = modelGradients(dlnet,fcParams,d1x1,d1x2,label)
Y = forwardnet(dlnet, fcParams, d1x1, d1x2);
loss = binarycrossentropy(Y, label);
[gradientsSubnet,gradientsParams] = dlgradient(loss,dlnet.Learnables,fcParams);
end

function [data, label] = get_data_label(p, n)
sz = size(p,3);
for i = 1:sz
   data(1:8, i) = reshape(p(1:4,1:2,i), [8,1]); 
   data(9:16, i) = reshape(n(1:4,1:2,i), [8,1]);
   label(i, 1) = 0; 
   
   data(1:8, i+sz) = reshape(n(1:4,1:2,i+sz), [8,1]);
   data(9:16, i+sz) = reshape(n(1:4,1:2,i+sz), [8,1]);
   label(i+sz,1) = 1;
end
end

function loss = binarycrossentropy(Y,pairLabels)
    % binarycrossentropy accepts the network's prediction Y, the true
    % label, and pairLabels, and returns the binary cross-entropy loss value.
    
    % Get precision of prediction to prevent errors due to floating
    % point precision    
    precision = underlyingType(Y);
      
    % Convert values less than floating point precision to eps.
    Y(Y < eps(precision)) = eps(precision);
    %convert values between 1-eps and 1 to 1-eps.
    Y(Y > 1 - eps(precision)) = 1 - eps(precision);
    asdf = 100;
    % Calculate binary cross-entropy loss for each pair
    loss = -pairLabels*log(Y) - (1 - pairLabels)*log(1 - Y);
    asdf = 100;
end






