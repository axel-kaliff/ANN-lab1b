
nx = 1000;
nclasses = 2;

alpha = 0.9; %how much the previous delta should change the next one
eta = 0.0001; % learning rate
epoch = 1000;
mA = [ 1.0, 0.3]; sigmaA = 0.2;
mB = [ 0.0, -0.1]; sigmaB = 0.3;
classA(1,:) = [ randn(1,round(0.5*nx)) .* sigmaA - mA(1), ...
randn(1,round(0.5*nx)) .* sigmaA + mA(1)];
classA(2,:) = randn(1,nx) .* sigmaA + mA(2);
classB(1,:) = randn(1,nx) .* sigmaB + mB(1);
classB(2,:) = randn(1,nx) .* sigmaB + mB(2);
figure('color','w');
scatter(classA(1,:),classA(2,:),'o','b'); hold on;
scatter(classB(1,:),classB(2,:),'x','r'); hold on;
legend('classA','classB');

% Randomly remove 25% from each class
[trainA, trainB, verifyA, verifyB] = splitData(classA, classB, 0.25, 0.25);

[~,ta_len] = size(trainA);[~,tb_len] = size(trainB);[~,va_len] = size(verifyA);[~,vb_len] = size(verifyB);

trainA_targets = zeros(1,ta_len) + 1;trainB_targets = zeros(1,tb_len) - 1;vA_targets = zeros(1,va_len) + 1;vB_targets = zeros(1,vb_len) - 1;

nhidden = [5 16 32 64];
[~,n_experiments] = size(nhidden);
figure(1);
figure('color','w');
for i=1:n_experiments
    [w,v, MSE, out] = backprop([trainA,trainB],[trainA_targets,trainB_targets], eta, epoch, nhidden(i), nclasses, alpha);
    plot(1:epoch,MSE); hold on;
    [score] = verifyBackprop(verifyA, verifyB,[vA_targets, vB_targets], w,v);
    fprintf('Node count: %d, final MSE training: %d, verification accurracy: %d \n',nhidden(i), MSE(epoch), score);
end
legend('Hidden nodes: 5', 'Hidden nodes:16','Hidden nodes:32','Hidden nodes:64');
title('Learning curves 75% training');

% remove 50% from classA
[trainA, trainB, verifyA, verifyB] = splitData(classA, classB, 0.5, 1);

[~,ta_len] = size(trainA);
[~,tb_len] = size(trainB);
[~,va_len] = size(verifyA);
[~,vb_len] = size(verifyB);

trainA_targets = zeros(1,ta_len) + 1;    
trainB_targets = zeros(1,tb_len) - 1;
vA_targets = zeros(1,va_len) + 1;    
vB_targets = zeros(1,vb_len) - 1;

nhidden = [16 32 64];
[~,n_experiments] = size(nhidden);
figure(1);
figure('color','w');
for i=1:n_experiments
    [w,v, MSE, out] = backprop([trainA,trainB],[trainA_targets,trainB_targets], eta, epoch, nhidden(i), nclasses, alpha);
    
    plot(1:epoch,MSE); hold on;
    score = verifyBackprop(verifyA, verifyB,[vA_targets, vB_targets], w,v);
end
legend('Hidden nodes:16','Hidden nodes:32','Hidden nodes:64');
title('Learning curves, 50% from classA');

%Special case
index_p = find(classA(1,:)>0);  index_n = find(classA(1,:)<0);
classA_p = classA(:,index_p);  classA_n = classA(:,index_n);
subclass_p = subsample(0.2,classA_p); subclass_n = subsample(0.8,classA_n); 
subclassA = [subclass_p,subclass_n];
% Plot the samples

[trainA, trainB, verifyA, verifyB] = splitData(subclassA, classB, 1, 1);

[~,ta_len] = size(trainA);
[~,tb_len] = size(trainB);
[~,va_len] = size(verifyA);
[~,vb_len] = size(verifyB);

trainA_targets = zeros(1,ta_len) + 1;    
trainB_targets = zeros(1,tb_len) - 1;
vA_targets = zeros(1,va_len) + 1;    
vB_targets = zeros(1,vb_len) - 1;

nhidden = [16 32 64];
[~,n_experiments] = size(nhidden);
figure(1);
figure('color','w');
for i=1:n_experiments
    [w,v, MSE, out] = backprop([trainA,trainB],[trainA_targets,trainB_targets], eta, epoch, nhidden(i), nclasses, alpha);
    
    plot(1:epoch,MSE); hold on;
end
legend('Hidden nodes:16','Hidden nodes:32','Hidden nodes:64');
title('Remove 20% from classA(1,:)<0 and 80% from classA(1,:)>0');

