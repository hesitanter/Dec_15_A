clear;
str = 'data_file_train_591.txt';
[data, time] = load_data(str);        % load data, standardization
[data, time] = sliding_mean(data, time);
[p, n] = generate_p_n(data);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% forward propagation + back propagation 
% rnn --> subtraction --> 1_output --> sigmoid --> prediction 
% a = act(w1*x1)
% b = act(w2(x2+a))
% c = act(w3(x3+b))
% d = act(w4(x4+c))
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
w1 = [0.5, 0.5, 0.5, 0.5];
w2 = [0.5;0.5;0.5;0.5;0.5;0.5;0.5;0.5];
lr = 0.1;
pair = generate_pair(n, p); % pair: p, n, n,n 
size_p = size(p, 3);        % [1,159]p, [160,159*2]n, 
iteration = size(pair, 3)/2;

for k = 1:3000 
    for i = 1:iteration
        a1 = tanh(pair(1, 1:2, i)*w1(1));
        b1 = tanh((pair(2, 1:2, i)+a1)*w1(2));
        c1 = tanh((pair(3, 1:2, i)+b1)*w1(3));
        d1 = tanh((pair(4, 1:2, i)+c1)*w1(4));
        cat1 = [a1 b1 c1 d1];

        a2 = tanh(pair(1, 1:2, i+2*size_p)*w1(1));
        b2 = tanh((pair(2, 1:2, i+2*size_p)+a2)*w1(2));
        c2 = tanh((pair(3, 1:2, i+2*size_p)+b2)*w1(3));
        d2 = tanh((pair(4, 1:2, i+2*size_p)+c2)*w1(4));
        cat2 = [a2 b2 c2 d2];

        vec = [pair(1:4, 1:2, i);pair(1:4, 1:2, i+2*size_p)];

        sub = abs(cat2 - cat1);
        out = sigmoid(sub*w2);
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % back propagation
        % prediction = 0, dissimilar; prediciton = 1, similar
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        if pair(1, 3, i) == pair(1, 3, i+2*size_p)
            label = 1;
        else
            label = 0;
        end
        loss(i+(k-1)*iteration) = bce(out, label);
        w2 = w2 - lr*delta_w2(label, out, sub, w2);
        w1 = w1 - lr*delta_w1(label, out, sub, vec, w1, w2, cat1, cat2);
    end

end

plot(loss)





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

function x = generate_pair(n, p)
size_p = size(p, 3);
x(:, :, :) = p;
x(:, :, size_p+1:size_p*4) = n(:, :, 1:3*size_p);
end

function x = sigmoid(y)
x = 1/(1+exp(-y));
end

function loss = bce(Y,label)
    % binarycrossentropy accepts the network's prediction Y, the true
    % label, and pairLabels, and returns the binary cross-entropy loss value.
    
    % Get precision of prediction to prevent errors due to floating
    % point precision    
    precision = class(Y);
      
    % Convert values less than floating point precision to eps.
    Y(Y < eps(precision)) = eps(precision);
    %convert values between 1-eps and 1 to 1-eps.
    Y(Y > 1 - eps(precision)) = 1 - eps(precision);
    
    % Calculate binary cross-entropy loss for each pair
    loss = -label*log(Y) - (1 - label)*log(1 - Y);
end

function x = delta_w2(label,out,sub,w2)
temp = sub*w2;
diff_loss_out = (label-1)/(out-1) - label/out;
diff_out_w2 = 1/(1+exp(-temp))*(1-1/(1+exp(-temp)))*sub;
x = diff_loss_out * diff_out_w2;
x = x';
end

function y = delta_w1(label, out, sub, vec, w1, w2, cat1, cat2)
diff_loss_out = (label-1)/(out-1) - label/out;
temp = sub*w2;
for i = 1:4
    diff_out_sub = 1/(1+exp(-temp))*(1-1/(1+exp(-temp)))*w2(i);
    if i == 1
        diff_sub_w = abs( vec(i,1)*d_tanh(vec(i,1)*w1(i)) - vec(i+4,1)*d_tanh(vec(i+4,1)*w1(i)) );
    else
        diff_sub_w = abs( vec(i,1)*d_tanh(vec(i,1)*w1(i)+cat1(i-1)) - vec(i+4,1)*d_tanh(vec(i+4,1)*w1(i)+cat2(i-1)) );
    end
    y(i) = diff_loss_out * diff_out_sub * diff_sub_w;
end

end

function y = d_tanh(x)
tanh = (exp(x)-exp(-x))/(exp(x)+exp(-x));
y = 1-tanh^2;
end



