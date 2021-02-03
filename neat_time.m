% dui qi glucose and HR de shi jian
function [x, y] = neat_time(g_time, x, h_time, y)
m = size(g_time, 2); 
n = size(h_time, 2);
%g_time(x)
%h_time(y)
if which_bigger(g_time(x), h_time(y)) == 1  % g_time smaller
    while x < m
        if which_bigger(g_time(x), h_time(y)) && which_bigger(h_time(y), g_time(x+1))
            x = x + 1;
            break;
        end
        x = x + 1;
    end
else    % h_time smaller
   while y < n
       if which_bigger(h_time(y), g_time(x)) && which_bigger(g_time(x), h_time(y+1))
          break; 
       end
       y = y + 1;
   end
end
