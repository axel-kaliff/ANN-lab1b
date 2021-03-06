function [w,v,MSE, final_out]=backprop(input_patterns, input_targets,eta,epoch, Nhidden, nclasses, alpha)
%TODO add MSE to output

[~, ndata] = size(input_targets);

w = randn(Nhidden, nclasses+1);                     %hidden layer nodes
v = randn(1, Nhidden + 1);                   %output layer nodes
dw = zeros(size(w));                                %delta hidden layer weights
dv = zeros(size(v));                                %delta output layer weights

for epo=1:epoch
    patterns_and_targets = [input_patterns;input_targets];

    random_inputs = patterns_and_targets(:, randperm(size(patterns_and_targets, 2))); % target row
    
    patterns = random_inputs(1:2,:); % pattern rowsssss
    targets = random_inputs(3,:); % target row

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
    error = sum(0.5.*e.^2) / (ndata);
    MSE(epo) = error;
    
end

hin = w * [input_patterns ; ones(1,ndata)];             %input to hidden layer
hout = [2 ./ (1+exp(-hin)) - 1 ; ones(1,ndata)];  %output from hidden layer
oin = v * hout;                                     %input to next layer
out = 2 ./ (1+exp(-oin)) - 1;                       %final output

final_out = out;

end
