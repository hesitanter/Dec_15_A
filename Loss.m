function out = Loss(Y2, label)
% loss function
out = log(1 - Y2)*(label - 1) - label*log(Y2);
end
