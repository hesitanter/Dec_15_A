clear;
str = 'C:\Users\Ke Ma\Dropbox\dage&laodi\OhioT1DM-training\OhioT1DM-training\591-ws-training.xml';
[exercise, e_time] = read_in_exercise(str);
[heart, h_time] = read_in_hearate(str);
[gsr, g_time] = read_in_gsr(str);
[truth, t_time] = read_in_truth(str);
[temp, t_temp] = read_in_skin_temperature(str);

[data, time] = merge_data_3(heart, h_time, exercise, e_time, gsr, g_time, temp, t_temp);

%[data1, time1] = merge_data_2(heart, h_time, exercise, e_time, temp, t_temp);
%[data, time] = sliding_mean(data, time);
%[data, time] = meage_label(data, time, truth, t_time);


figure(4);
subplot(4,1,1);
plot(data(:, 1));
subplot(4,1,2);
plot(data(:, 2));
subplot(4,1,3);
plot(data(:, 3));
subplot(4,1,4);
plot(data(:, 4));





