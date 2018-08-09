% Denomination of data: EDEQ delta, s52_4 delta, bmi delta, s52_14 delta,
% s26_3 delta, fc_ofc_delta, edeq_responder
% Labels_clean = data_clean(:,7);
% threshold= median(data_clean(:,1))  % Cut-off value is median change of EDEQ
% response=data_clean(:,1)>threshold; 
% sum(response)/length(response)

%response=(data_clean(:,2)<data_clean(:,1)*0.6);
data=importdata('data_mrtrix.csv')
response=data.data(:,3);

perc_responder=sum(response)/length(data.data)
no_responder=sum(response)
no_nonresponder=sum(~response)
% Labels_clean=
Labels_clean_cell={};
for s=1:length(response)
    if response(s),
        Labels_clean_cell{1}(s,1)={'yes'};
    else
        Labels_clean_cell{1}(s,1)={'no'};
    end
end

data_clean=data.data;
% Labels_clean_cell = mat2cell(Labels_clean);
X_clean=data_clean(:,[5:6]);



%%  LOOCV
folds = nchoosek(1:length(data.data),1);
indicator = true(length(data.data),1);
%ps will hold the predictions of the test folds, ls the true labels
[ps,ls] = deal(nan(size(folds)));
for n = 1 : size(folds,1)
    %we knock out the test fold
    indicator(folds(n,:)) = 0;
    b = glmfit(X_clean(indicator,:),response(indicator),'binomial','link','logit'); % Logistic regression
    ls(n,:) = response(~indicator);
    ps(n,:) = glmval(b,X_clean(~indicator,:),'logit');
    %we return the test fold to the pool
    indicator(folds(n,:)) = 1;
end

[X,Y,~,AUC] = perfcurve(ls(:),ps(:),1);
hold on
plot(X,Y,'linewidth',3)
xlabel('False positive rate'); ylabel('True positive rate')
title('ROC for classification by logistic regression (LOOCV)')

[aa bb]=size(ps);
FIT_shuffled=reshape(ps, aa*bb,1);


FIT_binary=FIT_shuffled>0.5; 
RES_shuffled=reshape(ls, aa*bb,1);
a=sum((FIT_binary).*(RES_shuffled==1)); %true positive
b=sum((~FIT_binary).*(RES_shuffled==1)); %false negative
c=sum((FIT_binary).*(RES_shuffled==0)); %false positive
d=sum((~FIT_binary).*(RES_shuffled==0)); %true negative

average_diagnostics=AUC;
average_sensitivity_alt = a/(a+b)
average_specificity_alt = d/(c+d)
DOR=a/c*d/b
DOR_SE=10^((1/a+1/b+1/c+1/d)^0.5)
DOR_95CI=[DOR-1.96*DOR_SE DOR+1.96*DOR_SE] 