function [output, output_t] = merge_data_3(heart, h_time, exercise, e_time, gsr, gsr_time, temp, temp_time)
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


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% merge value(heart+step) with gsr
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
n = size(value, 2); 
l = size(gsr, 2);

count = 1; % value index
i = 1; % value time index
j = 1; % gsr  time index
value_f = zeros(3, 100); % exercise + hearate
time_f = strings(1, 100);
[i, j] = neat_time(time, i, gsr_time, j);   % neat time of exercise and heart rate

while i < n && j < l
    % a is the smaller one
    if (which_bigger(gsr_time(j), time(i)) == 1)
        a = gsr_time(j);
        b = time(i);
    else
        a = time(i);
        b = gsr_time(j);
    end
    % c is the bigger one
    if (which_bigger(time(i), gsr_time(j+1)) == 1)
        c = gsr_time(j+1);
        d = time(i);
    else
        c = time(i);
        d = gsr_time(j+1);
    end
    
    if interval_2(a, b) <= 3
        value_f(1:2, count) = value(:, i);
        value_f(3, count) = gsr(j);
        time_f(1, count) = time(i); 
        i = i + 1;
        j = j + 1;
        count = count + 1;
    elseif interval_2(d, c) <= 3
        value_f(1:2, count) = value(:, i);
        value_f(3, count) = gsr(j);
        time_f(1, count) = time(i); 
        i = i + 1;
        j = j + 2;
        count = count + 1;
    else
        i = i + 1;
        [i, j] = neat_time(time, i, gsr_time, j);
    end
end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% merge value(heart+step+gsr) with temperature
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
n = size(value_f, 2); 
l = size(temp, 2);

count = 1; % value index
i = 1; % value time index
j = 1; % gsr  time index
output = zeros(3, 100); % exercise + hearate
output_t = strings(1, 100);
[i, j] = neat_time(time_f, i, temp_time, j);   % neat time of exercise and heart rate

while i < n && j < l
    % a is the smaller one
    if (which_bigger(temp_time(j), time_f(i)) == 1)
        a = temp_time(j);
        b = time_f(i);
    else
        a = time_f(i);
        b = temp_time(j);
    end
    % c is the bigger one
    if (which_bigger(time_f(i), temp_time(j+1)) == 1)
        c = temp_time(j+1);
        d = time_f(i);
    else
        c = time_f(i);
        d = temp_time(j+1);
    end
    
    if interval_2(a, b) <= 3
        output(1:3, count) = value_f(:, i);
        output(4, count) = temp(j);
        output_t(1, count) = time_f(i); 
        i = i + 1;
        j = j + 1;
        count = count + 1;
    elseif interval_2(d, c) <= 3
        output(1:3, count) = value_f(:, i);
        output(4, count) = temp(j);
        output_t(1, count) = time_f(i); 
        i = i + 1;
        j = j + 2;
        count = count + 1;
    else
        i = i + 1;
        [i, j] = neat_time(time_f, i, temp_time, j);
    end
end

output = output';
output_t = output_t';

