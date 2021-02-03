function out = delta(x, Y1, Y2, label)
% dloss_dy2: (label - 1)/(Y - 1) - label/Y
out1 = (label - 1)/(Y2 - 1) - label/Y2;
% dy2_dy1: exp(-in)/(exp(-in) + 1)^2
out2 = exp(-Y1)/(exp(-Y1) + 1)^2;
% dy1_dw: x
out3 = x';
out = out1*out2*out3;
end