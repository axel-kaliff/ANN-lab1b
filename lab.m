% LAB1B

% generate data
n = 100;
mA = [ 0.3, 0.5]; sigmaA = 0.5;
mB = [-0.3, 0.7]; sigmaB = 0.5;
classA(1,:) = randn(1,n) .* sigmaA + mA(1);
classA(2,:) = randn(1,n) .* sigmaA + mA(2);
classB(1,:) = randn(1,n) .* sigmaB + mB(1);
classB(2,:) = randn(1,n) .* sigmaB + mB(2);

labelA = ones(1,size(classA,2));
labelB = zeros(1,size(classB,2))-1;
f1 = figure;
scatter(classA(1,:), classA(2,:), 20,'r','o');
hold on;
scatter(classB(1,:), classB(2,:), 20,'b','x');
hold on;

Nhidden = 4;

% Combine classA and classB as X_input
X_in = [classA,classB];
%X_bias = ones(1,size(X_in,2));
%X_in = [X_in;X_bias];

patterns = X_in;
% Length of input
[~,ndata] = size(X_in);

%targets
classA_targets = zeros(1,n) + 1;
classB_targets = zeros(1,n) - 1;
targets = [classA_targets,classB_targets];

w = randn(Nhidden,3);%hidden layer weights
v = randn(2, Nhidden + 1);%output layer weights

% initialize delta weights
dw = zeros(Nhidden,3);
dv = zeros(Nhidden,2);
alpha = 0.9;
eta = 0.0001;

% forward pass
hin = w * [patterns ; ones(1,ndata)];
hout = [2 ./ (1+exp(-hin)) - 1 ; ones(1,ndata)];

disp(size(v));
disp(size(hout));

oin = v * hout;
out = 2 ./ (1+exp(-oin)) - 1;
%disp(out);
%disp(size(out));

% backward pass
delta_o = (out - targets) .* ((1 + out) .* (1 - out)) * 0.5;

fprintf('delta_o %s' , size(delta_o));

delta_h = (transpose(v) * delta_o) .* ((1 + hout) .* (1 - hout)) * 0.5;
%remove bias row
disp(size(delta_h));
%delta_h = delta_h(1:Nhidden, :);

%disp(size(dw));
%disp(size(delta_h));
%disp(size(transpose(patterns)));
%disp(size((delta_h * transpose(patterns))));

% update weights
disp(size(dw));
disp(size(delta_h));
disp(size(transpose(patterns)));

dw = (dw .* alpha) - (delta_h * transpose(patterns)) .* (1-alpha);
dv = (dv .* alpha) - (delta_o * transpose(hout)) .* (1-alpha);

disp(size(w));
disp(size(dw .* eta));


W = w + dw .* eta;
V = v + dv .* eta;



% Apply and compare perceptron learning with the Delta learning rule in
% batch mode on the generated dataset. Adjust the learning rate and study
% the convergence of the two algorithms
