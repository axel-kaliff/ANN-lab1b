function [w,v,MSE]=orderedBackprop(input_classA,input_classB,eta,epoch, Nhidden, nclasses, alpha)
%TODO add MSE to output

patterns = [input_classA, input_classB];                        % X (input patterns)
[~,class_A_len] = size(input_classA);
[~,class_B_len] = size(input_classB);
classA_targets = zeros(1,class_A_len) + 1;    
classB_targets = zeros(1,class_B_len) - 1;
targets = [classA_targets,classB_targets];          % targets (without bias)

[~, ndata] = size(targets);

w = randn(Nhidden, nclasses+1);                     %hidden layer nodes
v = randn(nclasses, Nhidden + 1);                   %output layer nodes
dw = zeros(size(w));                                %delta hidden layer weights
dv = zeros(size(v));                                %delta output layer weights

for epo=1:epoch

    error = 0;

    hin = w * [patterns ; ones(1,ndata)];             %input to hidden layer
    hout = [2 ./ (1+exp(-hin)) - 1 ; ones(1,ndata)];  %output from hidden layer
    oin = v * hout;                                     %input to next layer
    out = 2 ./ (1+exp(-oin)) - 1;                       %final output

    %calculate deltas backwards
    delta_o = (out - targets) .* ((1 + out) .* (1 - out)) * 0.5;        %delta output layer
    delta_h = (v' * delta_o) .* ((1 + hout) .* (1 - hout)) * 0.5;       %delta hidden layer
    delta_h = delta_h(1:Nhidden, :);                                    %remove bias from delta hidden layer (WHY???)

    %apply delta, first to hidden then to output
    dw = (dw .* alpha) - (delta_h * [patterns;ones(1,ndata)]') .* (1-alpha);                 
    dv = (dv .* alpha) - (delta_o * hout') .* (1-alpha);

    w = w + dw .* eta;
    v = v + dv .* eta;

    e = out-targets;
    error = sum(0.5.*max(e).^2) / (ndata);
    MSE(epo) = error;
    
end

% disp([heaviside(max(out));heaviside(targets)])
% 
% result = 1 - sum(abs(heaviside(max(out))-heaviside(targets))) / ndata


%fprintf('Number of nodes: %s, final MSE: %s \n', Nhidden, MSE(epo));

