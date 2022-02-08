%generate data
x=[-5:0.5:5]';
y=[-5:0.5:5]'
z=exp(-x.*x*0.1) * exp(-y.*y*0.1)' - 0.5;

[~, ~,~, ~] = trainFunction(x, y, z, 1000, 0.001, 0.9, 1:2:25);
drawnow;

x = sort(subsample(0.8, x'));
x = x(1,:);
subclass = sort(x, 1);
x = x';
y = x
z = exp(-x.*x*0.1) * exp(-y.*y*0.1)' - 0.5;

[lowestMSE, w,v, m] = trainFunction(x, y, z, 1000, 0.001, 0.9, 1:2:20);
drawnow;

%verify

x = sort(subsample(0.2, x'));
x = x(1,:);
subclass = sort(x, 1);
x = x';
y = x
z = exp(-x.*x*0.1) * exp(-y.*y*0.1)' - 0.5;

mesh(x, y, z);
[x_len, ~]= size(x);
[y_len, ~]= size(y);
ndata = x_len*y_len;

targets = reshape (z, 1, ndata);
[xx, yy] = meshgrid (x, y);
patterns = [reshape(xx, 1, ndata); reshape(yy, 1, ndata)];

[~, ndata] = size(targets);

    hin = w * [patterns ; ones(1,ndata)];             %input to hidden layer
    hout = [2 ./ (1+exp(-hin)) - 1 ; ones(1,ndata)];  %output from hidden layer
    oin = v * hout;                                     %input to next layer
    out = 2 ./ (1+exp(-oin)) - 1;                       %final output
figure(1);
figure('color','w');

    gridsize = x_len;
    zz = reshape(out, gridsize, gridsize);
    mesh(x,y,zz);
    axis([-5 5 -5 5 -0.7 0.7]);
   
    drawnow();
