opts = []
opts.train.batch = 200 ;
opts.train.numEpochs = 10 ;
opts.train.continue = true ;
opts.train.gpus = [] ;
opts.train.learningRate = 0.01 ;
%opts.train.expDir = [vl_rootnn '/data/toy'] ;
%opts.dataDir = [vl_rootnn '/data/toy-dataset'] ;

vroot = 'C:/Stuff/work/ANU/PersonID/examples'
opts.train.expDir =[ vroot '/toy'];
opts.dataDir = [vroot '/toy-dataset'] ;

opts.imdbPath = [opts.train.expDir '/imdb.mat'] ;
opts.epochs = 10
imdb = load(opts.imdbPath) ;
%[img, labels] = params.getBatch(params.imdb, batch) ;
opts.imdb = imdb


%% add solver; binomial deviance.
opts.solver = []
opts.backPropDepth = +inf ;
opts.sync = false ;
opts.cudnn = true ;
opts.train = imdb.images(:,:,:,1:4500)
opts.val = imdb.images(:,:,:,4501:4950)





f = 1/100 ;
net.layers = {} ;
net.layers{end+1} = struct('type', 'conv', ...
                           'weights', {{f*randn(5,5,1,5, 'single'), zeros(1, 5, 'single')}}) ;
net.layers{end+1} = struct('type', 'pool', ...
                           'method', 'max', ...
                           'pool', [2 2], ...
                           'stride', 2) ;
net.layers{end+1} = struct('type', 'conv', ...
                           'weights', {{f*randn(5,5,5,10, 'single'),zeros(1,10,'single')}}) ;
net.layers{end+1} = struct('type', 'pool', ...
                           'method', 'max', ...
                           'pool', [2 2], ...
                           'stride', 2) ;
net.layers{end+1} = struct('type', 'conv', ...
                           'weights', {{f*randn(5,5,10,3, 'single'),  zeros(1,3,'single')}}) ;
net.layers{end+1} = struct('type', 'softmaxloss') ;
net = vl_simplenn_tidy(net) ;
net.layers{end-1}.precious = 1; % do not remove predictions, used
                                % for error
%%net.layers{end}.class = labels ;



%% include parameter sharing between two Conv Nets.(averaging of weights).




%% NEXT STOP USE CNN_TRAIN.
%[net, stats] = cnn_train(net, imdb, @(imdb, batch) getBatch(imdb, batch, use_gpu), ...
%  'train', find(imdb.set == 1), 'val', find(imdb.set == 2), opts.train) ;


[net_train,res] = process(net,opts,'train') ;
[net_val,res] = process(net, opts,'val') ;
%predictions = gather(res(end-1).x) ;
        
    
    
    
    
