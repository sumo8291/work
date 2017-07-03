
%% TO-DO:
%1.) apply cosine connection on FC layer, 
%2.)weight sharing for res and res2
%3.) loss function in res.

function [net,res] = process(net,opts,mode)
% PROCESS - trains CNN for Custom imdb
%   
    opts.batch = 100;
    params = opts;
    res = [] ;
    error = [] ;
    params.epoch = 10;
    if strcmp(mode, 'train')
        dzdy = 1 ;
        evalMode = 'normal' ;       
        vdata = opts.train
    else
        dzdy = [] ;
        evalMode = 'test' ;
        vdata = opts.test;
    end
    qend = length(vdata);
    
    for i=1:params.epoch
        rng(params.epoch) ;
        disp(i)
                                                                 % shuffle
                                                                 %params.train = params.train(1:min(opts.epochSize, numel(opts.train)));
        for t= 1:params.batch:length(vdata)
            vstart = t;
            vend = t + params.batch-1;
            
            if(vend >=qend)
                vend = qend;
            end
            im_a = vdata(vstart:vend).img_a;
            im_b = vdata(vstart:vend).img_a;
            labels = vdata(vstart:vend).labels;
            
            net.layers{end}.class = labels ;
            res1 = vl_simplenn(net, im_a, dzdy, res, ...
                              'mode', evalMode, ...
                              'backPropDepth', params.backPropDepth, ...
                              'sync', params.sync, ...
                              'parameterServer', []) ;
            res2 = vl_simplenn(net, im_a, dzdy, res, ...
                              'mode', evalMode, ...
                              'backPropDepth', params.backPropDepth, ...
                              'sync', params.sync, ...
                              'parameterServer', []) ;
            %% weight sharing, loss function,connection cosine function 
            if(vend>=qend)
                break
            end
            

        end
    end
end
