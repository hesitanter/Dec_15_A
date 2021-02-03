
function A_one_shot_6_backup(input)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% train with negative data, no bias in fully connected layer
% modified: label, when same class->0, different classes-> 1 Dec/28
% modified: test data sets   Dec/29
% modified: add gsr data, skin temperature for training Dec/30
% modified: "get_pair_label" function error Dec/31
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
str = 'data_file_train_'+string(input)+'.txt';
str_test = 'data_file_test_'+string(input)+'.txt';

[data1, time] = load_data(str);        % load data, standardization
[data2, time] = sliding_mean(data1, time);
data2 = mapminmax(data2')';

[p, n] = generate_p_n(data2);
%[data, label] = get_pair_label(n);
[data, label] = get_pair_label(p, n);


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
          sequenceInputLayer(16, 'Name', 'Input')
          recurrent_layer_modified(1, 'rnn')];
lgraph = layerGraph(layers);
dlnet = dlnetwork(lgraph);
fcWeights = dlarray(0.01*randn(1,4));
fcBias = dlarray(0.01*randn(1,1));
fcParams = struct(...
    "FcWeights",fcWeights,...
    "FcBias",fcBias);


vel_1 = [];
vel_2 = [];
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% training
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
for i = 1:size(data, 2)
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % dlarray format ?????????
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    dlx1 = dlarray(single(data(1:16, i)), 'CT');
    dlx2 = dlarray(single(data(17:32, i)), 'CT');
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



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% testing
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

[test_data, test_label] = load_test_data(str_test); % load test data
disp(length(test_data));
for i = 1:size(test_data, 2)
    dlx1 = dlarray(single(test_data(1:16, i)), 'CT');
    dlx2 = dlarray(single(test_data(17:32, i)), 'CT');

    dlY(i) = predict_one_shot(dlnet,fcParams,dlx1,dlx2);
    Y(i) = gather(extractdata(dlY(i)));
end
Y = extractdata(Y);
plot(Y);
xlabel('samples');
ylabel('probability indicates different');


function [data, time] = load_data(str)
y = load(str);
data = y(:, 1:5);
time = y(:, 6:11);
end

function [x, y] = generate_p_n(data)
n = size(data, 1);
count_x = 1;
count_y = 1;
for i = 4:n
   negative = length(find(data(i-3:i, 5) == -1));
   positive = length(find(data(i-3:i, 5) == 1));
   
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
if count_x == 1
    x = 0;
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

function [data, label] = get_pair_label(p, n)
sz = size(p,3);
for i = 1:sz
   data(1:16, i) = reshape(p(1:4,1:4,i), [16,1]); 
   data(17:32, i) = reshape(n(1:4,1:4,i), [16,1]);
   label(i, 1) = 1; 
   
   data(1:16, i+sz) = reshape(n(1:4,1:4,i+2*sz), [16,1]);
   data(17:32, i+sz) = reshape(n(1:4,1:4,i+3*sz), [16,1]);
   label(i+sz,1) = 0;
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
    % Calculate binary cross-entropy loss for each pair
    loss = -pairLabels*log(Y) - (1 - pairLabels)*log(1 - Y);
end

function Y = predict_one_shot(dlnet,fcParams,x1, x2)
F1 = predict(dlnet, x1);
F1 = sigmoid(F1);
F2 = predict(dlnet, x2);
F2 = sigmoid(F2);
Y = abs(F1-F2);
Y = fullyconnect(Y,fcParams.FcWeights,fcParams.FcBias);
Y = sigmoid(Y);
end

function [test_data, test_label] = load_test_data(str)
y = load(str);
pre_data = y(:, 1:5);
test_time = y(:, 6:11);
test_data(:, 1:4) = mapminmax(pre_data(:, 1:4)')';
test_data(:, 5) = pre_data(:, 5);
[test_data, test_time] = sliding_mean(test_data, test_time);

[p, n] = generate_p_n(test_data);
if length(p) == 1 && p == 0
    [test_data, test_label] = get_pair_label(n(:,:,1:30), n(:,:,31:size(n, 3)));
else
    [test_data, test_label] = get_pair_label(p, n);
end
end

end
