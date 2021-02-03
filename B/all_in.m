

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% difference from A
% change the testing part, test with all testing data
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clear;
a = [559,563,570,575,588,591];
b = [0.15, 0.034, 0.011, 0.45, 0.086, 0.081];
for i = 1:6
    %write_test_data(563);
    %write_train_data(a(i));
    %A_one_shot_6_backup(a(i), b(i));
end
A_one_shot_6_backup(a(3), b(3));

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% patient 563 data isn't collected by sensor
% 
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
