% a <= b --> c = 1
function c = which_bigger(a, b) 
if str2double(a{1,1}(7:10)) < str2double(b{1,1}(7:10)) 
    c = 1;
elseif str2double(a{1,1}(7:10)) == str2double(b{1,1}(7:10))
    if str2double(a{1,1}(4:5)) < str2double(b{1,1}(4:5))
        c = 1;
    elseif str2double(a{1,1}(4:5)) == str2double(b{1,1}(4:5))
        if str2double(a{1,1}(1:2)) < str2double(b{1,1}(1:2))
            c = 1;
        elseif str2double(a{1,1}(1:2)) == str2double(b{1,1}(1:2))
            if str2double(a{1,1}(12:13)) < str2double(b{1,1}(12:13))
                c = 1;
            elseif str2double(a{1,1}(12:13)) == str2double(b{1,1}(12:13))
                if str2double(a{1,1}(15:16)) < str2double(b{1,1}(15:16))
                    c = 1;
                elseif str2double(a{1,1}(15:16)) == str2double(b{1,1}(15:16))
                        if str2double(a{1,1}(18:19)) <= str2double(b{1,1}(18:19))
                            c = 1;
                        else
                            c = 0;
                        end
                else
                    c = 0;
                end
            else
                c = 0;
            end
        else
            c = 0;
        end
    else
        c = 0;
    end
else
    c = 0;
end
                        
             