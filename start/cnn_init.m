opts = []
opts.batch = 200 ;
opts.numEpochs = 10 ;
opts.learningRate = 0.01 ;
img = load('vdata.mat')
%process function to compute everything.
net1 = process(net,opts,'train');
