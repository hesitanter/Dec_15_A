
% write data in a txt
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% heart rate, step, label
% eliminate night data
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function write_train_data(input)
str = 'C:\Users\Ke Ma\Dropbox\dage&laodi\OhioT1DM-training\OhioT1DM-training\'+string(input)+'-ws-training.xml';
sleep = read_in_sleep_data(str);
[exercise, e_time] = read_in_exercise(str);
[heart, h_time] = read_in_hearate(str);
[truth, t_time] = read_in_truth(str);
[data, time] = merge_data(heart, h_time, exercise, e_time);
[data, time] = meage_label(data, time, truth, t_time);


fp = fopen('data_file_train_'+string(input)+'.txt', 'w+');
count = 1;
for i = 1:size(data,1)
    if which_bigger(time(i), sleep(count, 2))
        fprintf(fp, '%d %d %d ' , data(i, :));
        day = str2double(time{i,1}(1:2));
        month = str2double(time{i,1}(4:5));
        year = str2double(time{i,1}(7:10));
        hour = str2double(time{i,1}(12:13));
        min = str2double(time{i,1}(15:16));
        sec = str2double(time{i,1}(18:19));
        fprintf(fp, '%d %d %d %d %d %d\r\n', day, month, year, hour, min, sec);
    elseif which_bigger(sleep(count, 2), time(i)) && which_bigger(time(i), sleep(count, 1))
        continue;
    elseif which_bigger(sleep(count, 1), time(i)) && count+1< size(sleep,1) && which_bigger(time(i), sleep(count+1, 2))
        fprintf(fp, '%d %d %d ' , data(i, :));
        day = str2double(time{i,1}(1:2));
        month = str2double(time{i,1}(4:5));
        year = str2double(time{i,1}(7:10));
        hour = str2double(time{i,1}(12:13));
        min = str2double(time{i,1}(15:16));
        sec = str2double(time{i,1}(18:19));
        fprintf(fp, '%d %d %d %d %d %d\r\n', day, month, year, hour, min, sec);
    elseif count+1<= size(sleep,1) 
        count = count + 1;
    end
end
fclose(fp);



