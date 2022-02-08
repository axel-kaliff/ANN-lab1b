function [ratio]=verifyBackprop(input_classA,input_classB, targets,w,v)
%TODO add MSE to output

patterns = [input_classA, input_classB];                        % X (input patterns)
%[~,class_A_len] = size(input_classA);
%[~,class_B_len] = size(input_classB);

[~, ndata] = size(targets);

    hin = w * [patterns ; ones(1,ndata)];             %input to hidden layer
    hout = [2 ./ (1+exp(-hin)) - 1 ; ones(1,ndata)];  %output from hidden layer
    oin = v * hout;                                     %input to next layer
    out = 2 ./ (1+exp(-oin)) - 1;                       %final output
   result = heaviside(max((out)));

ratio = 1 - sum(abs(heaviside(out)-heaviside(targets))) / ndata;


