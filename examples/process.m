function [net,res] = process(net,opts,mode)
% PROCESS - 
%   
    opts.batch = 300
    params = opts
    res = [] ;
    error = [] ;
    params.epoch = 10

    for i=1:params.epoch
        rng(params.epoch) ;
        disp i
        params.train = opts.train(randperm(numel(opts.train))) ; %
                                                                 % shuffle
                                                                 %params.train = params.train(1:min(opts.epochSize, numel(opts.train)));
        params.val = opts.val(randperm(numel(opts.val))) ;
        params.imdb = opts.imdb ;
        for t= 1:params.batch:length(opts.train)
            vstart = t;
            
            vend = t + params.batch
            if(vend >=1500)
                vend = 1500
            end
            [im, labels] = getBatch(params.imdb,vstart:vend ) ;
            if strcmp(mode, 'train')
                dzdy = 1 ;
                evalMode = 'normal' ;
            else
                dzdy = [] ;
                evalMode = 'test' ;
            end
            net.layers{end}.class = labels ;
            res = vl_simplenn(net, im, dzdy, res, ...
                              'mode', evalMode, ...
                              'backPropDepth', params.backPropDepth, ...
                              'sync', params.sync, ...
                              'parameterServer', []) ;
            if(vend>=1500)
                break
            end
            

        end
    end
end



function [images, labels] = getBatch(imdb, batch)
% --------------------------------------------------------------------
% This is where we return a given set of images (and their labels) from
% our imdb structure.
% If the dataset was too large to fit in memory, getBatch could load images
% from disk instead (with indexes given in 'batch').

    images = imdb.images(:,:,:,batch) ;
    labels = imdb.labels(batch) ;
end