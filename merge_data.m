function [value, time] = merge_data(heart, h_time, exercise, e_time)



% IMPORTANT!!!! Assume size(glucose) > size(insulin)
n = size(exercise, 2); 
l = size(heart, 2);

count = 1; % value index
i = 1; % exercise time index
j = 1; % heart rate time index
value = zeros(2, 100); % exercise + hearate
time = strings(1, 100);
[i, j] = neat_time(e_time, i, h_time, j);   % neat time of exercise and heart rate

% merge heart rate and basis_step
while i < n && j < l
    % a is the smaller one
    if (which_bigger(h_time(j), e_time(i)) == 1)
        a = h_time(j);
        b = e_time(i);
    else
        a = e_time(i);
        b = h_time(j);
    end
    % c is the bigger one
    if (which_bigger(e_time(i), h_time(j+1)) == 1)
        c = h_time(j+1);
        d = e_time(i);
    else
        c = e_time(i);
        d = h_time(j+1);
    end
    
    if interval_2(a, b) <= 3
        value(1, count) = exercise(i);
        value(2, count) = heart(j);
        time(1, count) = e_time(i); 
        i = i + 1;
        j = j + 1;
        count = count + 1;
    elseif interval_2(d, c) <= 3
        value(1, count) = exercise(i);
        value(2, count) = heart(j);
        time(1, count) = e_time(i); 
        i = i + 1;
        j = j + 2;
        count = count + 1;
    else
        i = i + 1;
        [i, j] = neat_time(e_time, i, h_time, j);
    end
end
value = value';
time = time';