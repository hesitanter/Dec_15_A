function [features, label] = extract_features(data)
n = size(data, 1);
for i = 4:n
    heart = data(i-3:i, 2);
    step = data(i-3:i, 1);
    
    mean_h = mean(heart);
    mean_s = mean(step);
    
    %a = step(step >= 2.0142);     % 20
    %b = step(step >= 3.2178);     % 30
    %c = heart(heart >= 1.5891);   % 85
    a = step(step >= 20);
    b = step(step >= 30);
    c = heart(heart >= 85);
    
    features(i-3, 1) = length(a) * length(b);
    features(i-3, 2) = length(a) * length(c);
    features(i-3, 3) = norm(heart) * norm(step);
    features(i-3, 4) = heart'*step;
    features(i-3, 5) = mean_h*mean_s;
    
    if size(data, 2) == 3
        ex =   data(i-3:i, 3);
        d = ex(ex >= 1);
        if length(d) >= 3
            label(i-3, 1) = 1;
        else
            label(i-3, 1) = 0;
        end
    else
        label(i-3) = 0;
    end
end

features = mapminmax(features')';




