% times function for element wise
function [val,derv] = loss_function(alpha, opts,beta)
% LOSS_FUNCTION - binomial deviance.
%     W*ln(exp(-a(S-b)*M) + 1)
    W = opts.W
    M = opts.M
    S = opts.S
    X = opts.X
    vinit =times(a*(b-S),M)
    vexp =  exp(vinit) + 1
    val = W*ln(vexp)
    vA = -1*a*times(times(W,M),exp(vinit)/(exp(vinit) + 1))
    vB = 1/(sqrt(opts.X.a*transpose(opts.X.a)*opts.X.a*transpose(opts.X.a)))
    vC = vB*transpose(opts.X.a)*opts.X.b*inverse(transpose(opts.X.a)*opts.X.a)
    vD = vB*transpose(opts.X.a)*opts.X.b*inverse(transpose(opts.X.a)*opts.X.a)
    derv = X*((vA*vB) + transpose(vA*vB)) - times(X,(repmat(vA*vC) + repmat(vA*vD)))
    
end
