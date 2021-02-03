
% train the mlp
clear;
data = load('data_file_train_591.txt');           % load data, with label
[features, labels] = extract_features(data);  % extract features, min-max normalized
vec = [0.5;0.5;0.5;0.5;0.5];                  % generate weights

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% train with all data, made optimization derivation compare to main11
% back propogation 
% binary cross entroy(BCE)
% y = w1*x1 + w2*x2 + w3*x3 + w4*x4 + w5*x5, 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

n = length(features);
alpha = 0.1;                             % learning rate
for i = 1:n
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % forward propagation
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    Y1(i) = y1(features(i, :), vec);     % objective function
    Y2(i) = y2(Y1(i));                   % activate
    loss(i) = Loss(Y2(i), labels(i));    % loss
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % back propogation
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%    
    vec = vec -  delta(features(i, :), Y1(i), Y2(i), labels(i, 1))* alpha;
    %disp(i);
end
%plot(loss)

fp = fopen('full_train_weithgs_591.txt', 'w+');
fprintf(fp, '%d\r\n' , vec(1, 1));
fprintf(fp, '%d\r\n' , vec(2, 1));
fprintf(fp, '%d\r\n' , vec(3, 1));
fprintf(fp, '%d\r\n' , vec(4, 1));
fprintf(fp, '%d\r\n' , vec(5, 1));
fclose(fp);

%{
function out = y1(x, w)
% y=w1*x1 + w2*x2 + w3*x3 + w4*x4 + w5*x5, 
out = x*w;
end
function out = y2(Y1)
% sigmoid function
out = 1/(1+exp(-Y1));
end
function out = Loss(Y2, label)
% loss function
out = log(1 - Y2)*(label - 1) - label*log(Y2);
end

function out = delta(x, Y1, Y2, label)
% dloss_dy2: (label - 1)/(Y - 1) - label/Y
out1 = (label - 1)/(Y2 - 1) - label/Y2;
% dy2_dy1: exp(-in)/(exp(-in) + 1)^2
out2 = exp(-Y1)/(exp(-Y1) + 1)^2;
% dy1_dw: x
out3 = x';
out = out1*out2*out3;
end
%}
%{
N = 1:12000;
figure(5);
subplot(5,1,1);
plot(N, features(N, 1));
subplot(5,1,2);
plot(N, features(N, 2));
subplot(5,1,3);
plot(N, features(N, 3));
subplot(5,1,4);
plot(N, features(N, 4));
subplot(5,1,5);
plot(N, features(N, 5));
%}
