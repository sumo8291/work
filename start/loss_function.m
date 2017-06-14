function [cost] = loss_function(alpha, beta,S,M)
% LOSS_FUNCTION - binomial deviance.
%     W*ln(exp(-a(S-b)*M) + 1)

    vinit = a*(b-S)
    vexp =  exp(times(vinit,M)) + 1
    cost = W*ln(vexp)
    
    
    
