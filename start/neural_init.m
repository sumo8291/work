rng('default');
% implement batch normalization.
f=1/100 ;
net.layers = {} ;

net.layers{end+1} = struct('type', 'conv', ...
                           'weights', {{f*randn(7,7,1,20, 'single'), zeros(1, 20, 'single')}}, ...
                           'stride', 1, ...
                           'pad', 3);

net.layers{end+1} = struct('type', 'relu') ;


net.layers{end+1} = struct('type', 'pool', ...
                           'method', 'max', ...
                           'pool', [2 2], ...
                           'stride', 2, ...
                           'pad', 0) ;

net.layers{end+1} = struct('type', 'conv', ...
                           'weights', {{f*randn(5,5,20,50, 'single'),zeros(1,50,'single')}}, ...
                           'stride', 1, ...
                           'pad', 2) ;

net.layers{end+1} = struct('type', 'pool', ...
                           'method', 'max', ...
                           'pool', [2 2], ...
                           'stride', 2, ...
                           'pad', 0) ;

net.layers{end+1} = struct('type', 'conv', ...
                           'weights', {{f*randn(4,4,50,500, 'single'),  zeros(1,500,'single')}}, ...
                           'stride', 1, ...
                           'pad', 0) ;

net.layers{end+1} = struct('type', 'relu') ;

net.layers{end+1} = struct('type', 'softmaxloss') ;

% Meta parameters
net.meta.inputSize = [128 48 3] ;
net.meta.learningRate = 0.001 ;
net.meta.numEpochs = 10 ;
net.meta.batchSize = 100 ;

% Fill in defaul values
net = vl_simplenn_tidy(net) ;
opts.networkType = 'simplenn' ;

opts = []
opts.batch = 100 ;
opts.numEpochs = 10 ;
opts.learningRate = 0.01 ;
img = load('vdata.mat');
img = img.filename;
rng('default');

val = randperm(length(img), round(length(img)/2));
opts.train = img(val);
opts.test = img;
opts.test(val) = [];

% Split to train and test and store in opts.
%process function to compute everything.
[net1,res] = process(net,opts,'train');
[net2,res2] = process(net,opts,'test');
