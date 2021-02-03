
function A_one_shot_6_backup(input, thre)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% train with negative data, no bias in fully connected layer
% modified: label, when same class->0, different classes-> 1 Dec/28
% modified: test data sets   Dec/29
% modified: add gsr data, skin temperature for training Dec/30
% modified: "get_pair_label" function error Dec/31
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

str = 'data_file_train_'+string(input)+'.txt';
str_test = 'data_file_test_'+string(input)+'.txt';
str_2 = 'C:\Users\Ke Ma\Dropbox\dage&laodi\OhioT1DM-testing\OhioT1DM-testing\'+string(input)+'-ws-testing.xml';
%[test_data, test_label, test_time] = load_test_data(str_test);
[data1, time] = load_data(str);        % load data, standardization
[data2, time] = sliding_mean(data1, time);
data2 = mapminmax(data2')';


[truth, truth_time] = read_in_truth(str_2);

[p, n] = generate_p_n(data2);
anchor = generate_anchor(n);
[data, label] = get_pair_label(p, n);


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



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% training
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
vel_1 = [];
vel_2 = [];
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
end



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% testing
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

[test_data, test_label, test_time, pre_data] = load_test_data(str_test); % load test data
count = 1;
result = zeros(1, 6) + 1000;
for i = 1:size(test_data, 2)
    dlx1 = dlarray(single(test_data(1:16, i)), 'CT');
    dlx2 = dlarray(single(reshape( anchor(1:4,1:4), [16,1])), 'CT');

    dlY(i) = predict_one_shot(dlnet,fcParams,dlx1,dlx2);
    Y(i) = gather(extractdata(dlY(i)));
    
    if (Y(i) > thre)
       result(count, :) = test_time(i, :); 
       count = count + 1;
    end
end

%%%%%%%%%%%%%%%%%%%%%%%%%
% draw picture
%%%%%%%%%%%%%%%%%%%%%%%%%
Y = extractdata(Y);

%plot_bad_detection(Y, thre, test_label, pre_data);

h = figure;
plot(Y, 'DisplayName', "test example");
xlabel('samples');
ylabel('probability of different');
hold on
plot(test_label*max(Y), 'DisplayName', "ground truth");
plot(ones(length(test_label), 1)*thre, 'DisplayName', 'threshold');
legend
hold off
saveas(h, sprintf('A_%d.fig', input));


result_analysis(result, input, truth, truth_time);
%%%%%%%%%%%%%%%%%%%%%%%%%
% function
%%%%%%%%%%%%%%%%%%%%%%%%%

function plot_bad_detection(Y, thre, test_label, pre_data)
over_thre = find(Y>thre);
over_thre(test_label(over_thre) == 1) = [];

scatter(over_thre, pre_data(over_thre, 1), 'DisplayName', 'step');
hold on
scatter(over_thre, pre_data(over_thre, 2), 'DisplayName', 'heart rate');
scatter(over_thre, pre_data(over_thre, 3), 'DisplayName', 'gsr');
scatter(over_thre, pre_data(over_thre, 4), 'DisplayName', 'skin temperature');
legend
end

function result_analysis(result, input, truth, truth_time)
    sz = size(result);
    if length(result) > 6
        fp = fopen('A_result.txt', 'a');
        fprintf(fp, "\r\n");
        fprintf(fp, "Patient_%d\r\n", input);
        
        j = 1;
        count = 1;
        count_num = 1;
        jj = 1; % truth index
        t_sz = length(truth);
        for i = 1:sz
         
            

%fprintf("%s; %s; %d %d \r\n", convert2string(result(i,:)), truth_time(jj), interval_2(truth_time(jj), convert2string(result(i,:))), truth(jj));
if which_bigger(convert2string(result(i,:)), truth_time(jj)) == 1
    continue;
elseif which_bigger(truth_time(jj), convert2string(result(i,:))) == 1 && interval_2(truth_time(jj), convert2string(result(i,:))) <= truth(jj)
    %fprintf("here");
    count_num = count_num + 1;
else
    if jj < t_sz
        temp1 = convert2string(result(i,:));
        temp2 = truth_time(jj+1);
        if temp1{1,1}(1:10) == temp2{1,1}(1:10)
            jj = jj + 1;
            if which_bigger(truth_time(jj), convert2string(result(i,:))) == 1 && interval_2(truth_time(jj), convert2string(result(i,:))) <= truth(jj)
                %fprintf("there: %s; %s; %d %d \r\n", convert2string(result(i,:)), truth_time(jj), interval_2(truth_time(jj), convert2string(result(i,:))), truth(jj));
                count_num = count_num + 1;
            end
        end
    end
end           
%disp(count_num);
              
            
            if (result(j, 1) == result(i, 1))
               continue; 
            else
                time_slip(count, 1) = interval_2( convert2string(result(j,:)), convert2string(result(i-1,:)) );
                fprintf(fp, "%d-%d-%d %d-%d-%d  ", result(j,1),result(j,2),result(j,3),result(j,4),result(j,5),result(j,6));
                fprintf(fp, "%d\r\n", time_slip(count, 1));
                j = i;
                count = count + 1;
            end
        end
        time_slip(count, 1) = interval_2( convert2string(result(j,:)), convert2string(result(i-1,:)) );
        fprintf(fp, "%d-%d-%d %d-%d-%d  ", result(j,1),result(j,2),result(j,3),result(j,4),result(j,5),result(j,6));
        fprintf(fp, "%d\r\n", time_slip(count, 1));
        fclose(fp);
        %disp(length(result));
        %disp(count_num);
    end

end

function output = convert2string(in)
    a = "0"+num2str(in(1, 4));
    b = "0"+num2str(in(1, 5));
    c = num2str(in(1, 1));
    d = num2str(in(1, 2));
    if strlength(a) == 3
       a = a{1,1}(2:3);
    end
    if strlength(b) == 3
       b = b{1,1}(2:3); 
    end
    if strlength(c) == 1
        c = "0"+c;
    end
    if strlength(d) == 1
        d = "0"+d;
    end   
    output = c+"-"+d+"-"+num2str(in(1, 3))+" "+a+":"+b+":"+"00";
end

function out = generate_anchor(in)
    size_in = size(in, 3);
    out = zeros(4, 5);
    for i = 1:100
        j = round(rand(1,1)*size_in);
        out = out + in(:,:,j);
    end
    out = out/100;
end

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

function [output, test_label, test_time, pre_data] = load_test_data(str)
y = load(str);
pre_data = y(:, 1:5);
test_time = y(:, 6:11);

index = find(pre_data(:, 5) == 1); 
step = sum(pre_data(index, 1))/length(index);
hear = sum(pre_data(index, 2))/length(index);
gsr  = sum(pre_data(index, 3))/length(index);
skin = sum(pre_data(index, 4))/length(index);
fprintf("%d, %d, %d, %d\r\n", step, hear, gsr, skin);

test_data(:, 1:4) = mapminmax(pre_data(:, 1:4)')';
test_data(:, 5) = pre_data(:, 5);
[test_data, test_time] = sliding_mean(test_data, test_time);

test_label = test_data(:, 5);
test_label(test_label == -1) = 0; 
test_time = test_time(4:size(test_time, 1), :);

n = size(test_data, 1);
for i = 4:n
   xx = test_data(i-3:i, :);
   output(:, i-3) = reshape(xx(:, 1:4), [16,1]);
end
end

function draw_plot(Y, test_label, input, thre)
Y = extractdata(Y);
h = figure;
plot(Y, 'DisplayName', "test example");
xlabel('samples');
ylabel('probability of different');
hold on
plot(test_label*max(Y), 'DisplayName', "ground truth");
plot(ones(length(test_label), 1)*thre, 'DisplayName', 'threshold');
legend
hold off
saveas(h, sprintf('A_%d.fig', input));
end


end
