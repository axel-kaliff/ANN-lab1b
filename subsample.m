function subclass = subsample(rate,class)
% Randomly selcet #rate of data from input samples
% subclass -- subsampled data
% rate -- sample rate, i.e.0.75
% class -- originial input data set

% n is size of original input data set
[~,n] = size(class);

subclass = zeros(2,round(n*rate));
index = randperm(n,round(n*rate));

for i=1:n*rate
    subclass(:,i) = class(:,index(i));
end

end