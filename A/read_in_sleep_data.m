



% read in basal values and corresponding timing:
% xml dataset: <event ts="30-11-2021 00:00:00" value="1.05"/>

function sleep = read_in_sleep_data(str)
dom = xmlread(str);
nodes = dom.getElementsByTagName("sleep");
node = nodes.item(0);
data = node.getElementsByTagName('event');
len = data.getLength;


for i = 1 : len
    element = data.item(i-1);
    time1 = element.getAttribute('ts_begin');
    time2 = element.getAttribute('ts_end');
    sleep(i, 1) = string(time1);
    sleep(i, 2) = string(time2);
end
