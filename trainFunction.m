function [lowestMSE, w,v, m] = trainFunction(x,y,z, epoch, eta, alpha, hidden)

mesh(x, y, z);
[x_len, ~]= size(x);
[y_len, ~]= size(y);
ndata = x_len*y_len;

targets = reshape (z, 1, ndata);
[xx, yy] = meshgrid (x, y);
patterns = [reshape(xx, 1, ndata); reshape(yy, 1, ndata)];

[~, n_experiments] = size(hidden);
figure(1);
figure('color','w');
lowestMSE = 1;
for i=1:n_experiments
    [w,v, MSE, out] = backprop(patterns, targets, eta, epoch,hidden(i), 2, alpha);
    fprintf('MSE %d hidden nodes: %d \n', hidden(i), MSE(epoch));
    
    gridsize = x_len;
    zz = reshape(out, gridsize, gridsize);
    mesh(x,y,zz);
    axis([-5 5 -5 5 -0.7 0.7]);
    rubrik = sprintf('Hidden nodes: %d \n MSE: %d', hidden(i), MSE(epoch));
    title(rubrik);
    m(i) = getframe;
    drawnow();

    if MSE(epoch) < lowestMSE
        lowestMSE = MSE(epoch);
        lowestMSEnode = hidden(i);
    end
end




end