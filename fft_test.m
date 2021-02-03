

clear;
data = load('data_file_train_591.txt');           % load data, with label

X = data(5:10, 1);
Fs = 1/300;            % Sampling frequency                    
T = 1/Fs;             % Sampling period       
L = 6;             % Length of signal
t = (0:L-1)*T;        % Time vector
Y = fft(X);
P2 = abs(Y/L);
P1 = P2(1:L/2+1);
P1(2:end-1) = 2*P1(2:end-1);
f = Fs*(0:(L/2))/L;
plot(f,P1) 
title('Single-Sided Amplitude Spectrum of X(t)')
xlabel('f (Hz)')
ylabel('|P1(f)|')
z = ifft(Y);
