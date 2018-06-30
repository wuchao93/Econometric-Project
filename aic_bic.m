function [ aic,bic,Rsquared ] = aic_bic( p,time_series )
%UNTITLED3 Summary of this function goes here
%   Detailed explanation goes here
% e = zeros(252,1);
% sse = zeros(10,1);
% sst = zeros(10,1);
% Rsquared = zeros(10,1);
for i = 1:length(p)
    theory_model = arima('ARlags',1:p(i),'D',1,0);
    [Estmdl,~,logL,info] = estimate(theory_model, time_series);
    numParam = length(info.X);
    [aic(i),bic(i)] = aicbic(logL,numParam,252);
    [e,~]= infer(Estmdl,time_series); % computing R squared
    sse = sum(e.^2);
    mean_ts = mean(time_series);
    sst = sum((time_series - mean_ts).^2);
    n = 252;
    Rsquared(i) = 1 - (n-1)/(n - numParam) * sse/ sst;
    
end




end

