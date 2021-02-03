function output = interval_2(time_i, time_j)
if time_i{1,1}(7:10) == time_i{1,1}(7:10) % same year
    if time_i{1,1}(1:10) == time_j{1,1}(1:10)    % same day
        output = same_day(time_i, time_j);
    elseif str2double(time_i{1,1}(4:5)) == str2double(time_j{1,1}(4:5)) && str2double(time_i{1,1}(1:2)) ~= str2double(time_j{1,1}(1:2))     % day2 - day1 > 1
        % same month, diff day
        output = same_month(time_i, time_j);
    else % diff month 
        output = diff_m_diff_d(time_i, time_j);
    end
else % different year
    output = diff_m_diff_d(time_i, time_j);
end

function n = same_day(time_i, time_j)
if str2double(time_i{1,1}(18:19)) ~= 0 || str2double(time_i{1,1}(15:16)) ~= 0 ||str2double(time_j{1,1}(18:19)) ~= 0 || str2double(time_j{1,1}(15:16)) ~= 0    % second don't have to account
    sec = (60 - str2double(time_i{1,1}(18:19)) + str2double(time_j{1,1}(18:19)))/60;
    if sec ~= 0
        min = 60 - str2double(time_i{1,1}(15:16)) + str2double(time_j{1,1}(15:16))-1;
    else 
        min = 60 - str2double(time_i{1,1}(15:16)) + str2double(time_j{1,1}(15:16));
    end
    hou = (- str2double(time_i{1,1}(12:13)) - 1 + str2double(time_j{1,1}(12:13)))*60;
    n = sec+min+hou;
else
    hou = (0-str2double(time_i{1,1}(12:13)) + str2double(time_j{1,1}(12:13)))*60;
    n = hou;
end 

function n = same_month(time_i, time_j) % same month different day
% minite ~= 00
if str2double(time_i{1,1}(18:19)) ~= 0 || str2double(time_i{1,1}(15:16)) ~= 0 ||str2double(time_j{1,1}(18:19)) ~= 0 || str2double(time_j{1,1}(15:16)) ~= 0   % second don't have to account
    sec = (60 - str2double(time_i{1,1}(18:19)) + str2double(time_j{1,1}(18:19)))/60;
    if sec ~= 0
        min = 60 - str2double(time_i{1,1}(15:16)) + str2double(time_j{1,1}(15:16))-1;
    else 
        min = 60 - str2double(time_i{1,1}(15:16)) + str2double(time_j{1,1}(15:16));
    end
    hou = (24 - str2double(time_i{1,1}(12:13)) - 1 + str2double(time_j{1,1}(12:13)))*60;
    n = sec+min+hou + 60*24*(-str2double(time_i{1,1}(1:2)) + str2double(time_j{1,1}(1:2))-1);
% minite == 00
else
    hou = (24 - str2double(time_i{1,1}(12:13)) + str2double(time_j{1,1}(12:13)) + 24*(-str2double(time_i{1,1}(1:2)) + str2double(time_j{1,1}(1:2))-1))*60;
    n = hou;
end 

function n = diff_m_diff_d(time_i, time_j)
% minite ~= 00
if str2double(time_i{1,1}(18:19)) ~= 0 || str2double(time_i{1,1}(15:16)) ~= 0 ||str2double(time_j{1,1}(18:19)) ~= 0 || str2double(time_j{1,1}(15:16)) ~= 0   % second don't have to account
    sec = (60 - str2double(time_i{1,1}(18:19)) + str2double(time_j{1,1}(18:19)))/60;
    if sec ~= 0
        min = 60 - str2double(time_i{1,1}(15:16)) + str2double(time_j{1,1}(15:16))-1;
    else 
        min = 60 - str2double(time_i{1,1}(15:16)) + str2double(time_j{1,1}(15:16));
    end
    hou = (24 - str2double(time_i{1,1}(12:13)) - 1)*60;
    m = sec+min+hou;
% minite == 00
else
    hou = (24 - str2double(time_i{1,1}(12:13)))*60;
    m = hou;
end 

if str2double(time_i{1,1}(4:5)) == 1|| str2double(time_i{1,1}(4:5)) == 3|| str2double(time_i{1,1}(4:5)) == 5 || str2double(time_i{1,1}(4:5)) == 7 || str2double(time_i{1,1}(4:5)) == 8 || str2double(time_i{1,1}(4:5)) == 10 || str2double(time_i{1,1}(4:5)) == 12
    n = m + 24*60*(31-str2double(time_i{1,1}(1:2))+str2double(time_j{1,1}(1:2))-1);
    n = n + str2double(time_j{1,1}(12:13))*60;
else
    if str2double(time_i{1,1}(4:5)) == 2
        if mod(str2double(time_i{1,1}(7:10)),4) == 0 && mod(str2double(time_i{1,1}(7:10)),100) ~= 0
            n = m + 24*60*(29-str2double(time_i{1,1}(1:2))+str2double(time_j{1,1}(1:2))-1);
        else
            n = m + 24*60*(28-str2double(time_i{1,1}(1:2))+str2double(time_j{1,1}(1:2))-1);
        end
        n = n + str2double(time_j{1,1}(12:13))*60;
    else
        n = m + 24*60*(30-str2double(time_i{1,1}(1:2))+str2double(time_j{1,1}(1:2))-1);
        n = n + str2double(time_j{1,1}(12:13))*60;
    end
end
        
        
        