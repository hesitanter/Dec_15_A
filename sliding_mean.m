function [data, time] = sliding_mean(vec, t)
n = length(vec);
for i = 2:n-1
   data(i-1, 1) = (vec(i-1, 1) + vec(i, 1) + vec(i+1, 1))/3; 
   data(i-1, 2) = (vec(i-1, 2) + vec(i, 2) + vec(i+1, 2))/3; 
   data(i-1, 3) = vec(i, 3);
   time(i-1, :) = t(i, :);
end