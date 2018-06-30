clc;
clear all;
close all;

%% IMPORT DATA
path = '';
inputFile = [path 'Track 7.xlsx'];
formatDate = 'dd/mm/yyyy';
data = xlsread(inputFile, 'a1:c253');
date = x2mdate(data(:,1));
cds_ois = data(:,2:3);
cds = data(:,2);
ois = data(:,3);

%% MEAN, STANDARD DEVIATION AND CORRELATION MATIRX
mean_cds = mean(cds); % mean
mean_ois = mean(ois);
std_cds = std(cds) ;% standard deviation
std_ois = std(ois);
lack_cds = sum(isnan(cds)); % lacking data
lack_ois = sum(isnan(ois));
cov = nancov(cds_ois);
corr_matrix = corrcov(cov) ;% correlation matrix
plot(date,cds,'k', date,ois,'m');
datetick('x',formatDate);
set(gca, 'FontSize', 16);
xlabel('time');
ylabel('basis point');
legend('cds','ois');

%% Test for unit root for cds
[h,p] = adftest(cds, 'model','ARD','lags',0:10)
difference_cds = diff(cds);%
[h1,p1] = adftest(difference_cds, 'model','ARD','lags',0:10)

%% Unit root test for ois
[h_ois,p_ois] = adftest(ois,'model','TS','lags',0:10);
difference_ois = diff(ois);%
[h1_ois,p1_ois] = adftest(difference_ois, 'model','TS','lags',0:10);

%% EXPORT DATA
adf = zeros(11,4);
adf(:,1)=h';
adf(:,2)=p';
adf(:,3)=h_ois';
adf(:,4)=p_ois';
adf_diff= zeros(11,5);
adf_diff(:,1)=[0:10]';
adf_diff(:,2)=h1';
adf_diff(:,3)=p1';
adf_diff(:,4)=h1_ois';
adf_diff(:,5)=p1_ois';

%% aicbic test
p = [0:10];
cds_info = zeros(11,3);
[cds_info(:,1),cds_info(:,2),cds_info(:,3)]= aic_bic(p,cds);
filename = 'cds_info.csv';
xlswrite(filename , cds_info);
%% 
logL = zeros(10,1); % Preallocate loglikelihood vector
a = zeros(10,1);
b = zeros(10,1);
for i = 1:10
EstMdl1 = arima('ARLags',1:i);
[~,~,logL] = estimate(EstMdl1,cds,'print',false);
[a(i),b(i)]=aicbic(logL,i+2,252)
end
%% 
p = [0:10];
ois_info = zeros(11,3);
[ois_info(:,1),ois_info(:,2),ois_info(:,3)]= aic_bic(p,ois);
filename = 'ois_info.csv';
xlswrite(filename , ois_info);

%% Fit model
theory_model = arima('ARlags',1,'D',1);
Estmdl_cds = estimate(theory_model, cds);
Estmdl_ois = estimate(theory_model, ois);
simu = zeros(252,10000);
aver_simu =zeros(252,1);

for i = 1:10000
simu(:,i)=simulate(Estmdl_cds,252);

end
for i = 1:252
    aver_simu(i,1)=mean(simu(i,:));
end

plot(aver_simu);
grid on;




%% Infer the residuals
e_cds = infer(Estmdl_cds,cds);
plot(date,e_cds);
datetick('x',formatDate);
set(gca, 'FontSize', 16);
xlabel('CDS residuals');
ylabel('basis point');

e_ois = infer(Estmdl_ois,ois);
plot(date,e_ois);
datetick('x',formatDate);
set(gca, 'FontSize', 16);
xlabel('OIS residuals');
ylabel('basis point');



%% residual correlation, jbtest
cds_ois_e = zeros(length(e_cds),2);
cds_ois_e(:,1)= e_cds;
cds_ois_e(:,2)= e_ois;
cov_r = nancov(cds_ois_e);
corr_matrix = corrcov(cov_r);
[h_e_cds,p_e_cds,tcds,ccds] = jbtest(e_cds)
[h_e_ois,p_e_ois,tois,cois] = jbtest(e_ois)


%% residual Heteroscedasticity test
[h_e_cds,p_e_cds,tcds,ccds] = lbqtest(e_cds.^2);
[h_e_ois,p_e_ois,tois,cois] = lbqtest(e_ois.^2);
[h_arch_cds,p_arch_cds,tcds,ccds] = archtest(e_cds)
[h_arch_ois,p_arch_ois,tois,cois] = archtest(e_ois)

%% Estimate VAR(P)
est= varm(2,1);
EstMdl= estimate(est,[cds,ois]);
summarize(EstMdl);
Y = [cds,ois];

[h1,pValue1,~,~,mles] = jcitest(Y,'model','H1*','display','params');

%[~,~,~,~,mles]=jcontest(Y,1,'Bvec',{[1,1]'});
[F,c_v] = granger_cause(diff(cds),diff(ois),0.05,1)
%% GARCH 
clc;
ToEstMdl = arima('ARlags',1,'D',1,'Variance',garch(1,1));
[EstMdl_cds] = estimate(ToEstMdl,cds)
[EstMdl_ois] = estimate(ToEstMdl,ois)
% options = optimoptions(@fmincon,'Diagnostics','on','Algorithm',...
%     'sqp','Display','off','ConstraintTolerance',1e-7)
