



% read in basal values and corresponding timing:
% xml dataset: <event ts="30-11-2021 00:00:00" value="1.05"/>

function [values, times] = read_in_exercise(str)
dom = xmlread(str);
nodes = dom.getElementsByTagName("basis_steps");
node = nodes.item(0);
data = node.getElementsByTagName('event');
len = data.getLength;


values = linspace(1, len, 1);
for i = 1 : len
    element = data.item(i-1);
    time = element.getAttribute('ts');
    value = element.getAttribute('value');
    times(i) = string(time);
    values(i) = str2double(value);
end
%disp(size(times));
%plot(values(1, 1:100))
