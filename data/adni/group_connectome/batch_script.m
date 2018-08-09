connectome_type={'aparc.a2009s+aseg_count','aparc.a2009s+aseg_length',...
    'aparc+aseg_count','aparc+aseg_length'};

% list=dir(['connectome_' connectome_type{1} '_*.txt']);
%  t=import_connectome(list(1).name);
%  [a,b]=size(t);
 
% eval(sprintf('ret%d',g))
%connectome_aparc2009_count=repmat([nan],a,b,length(list));

for ct=1:length(connectome_type)
v=genvarname(['connectome_' connectome_type{ct}]);

list=dir(['connectome_' connectome_type{ct} '_*.csv']);
 if ct < 3
     t=import_connectome_164(list(1).name);
 else 
     t=import_connectome_84(list(1).name);
 end
 [a,b]=size(t);
 
 
eval([v '=repmat([nan],a,b,length(t));']);


% for s=1:length(list)
%     connectome_aparc2009_count(:,:,s)=import_connectome(list(1).name);
% end
if ct < 3
    for s=1:length(list)
        eval([v '(:,:,s)=import_connectome_164(list(s).name);' ])
    end
else 
    for s=1:length(list)
    eval([v '(:,:,s)=import_connectome_84(list(s).name);' ])
    end
end

end

% %% rearrange connectome matrix (vectorization)
% load('data_connectome.mat') %order is (1)aparc_count (2)aparc_length (3)aparc2009_count (4)aparc2009_length
% 
% %cnt_a1_c = zeros(211,84*84);
% sz=[84*84];
% %vector = reshape(connectome_aparc_count, [sz,211]);
% cnt_a1_c = zeros(211,sz); 
% for s = 1:211
%     cnt_a1_c(s,:)=connectome_aparc_count(sz*(s-1)+1:s*sz);
% end
% 
% 
% cnt_a1_c_table=array2table(cnt_a1_c)
% 
% %connectome_aparc_count_vector = array2table(zeros(211,84*84));
% 
% % for i=1:84*84
% %     v=genvarname(['atlas1_count_var' int2str(i)]);
% %     connectome_aparc_count_vector.Properties.VariableNames{i}=v;
% % end
% 
% writetable(connectome_aparc_count_vector, 'test.csv','Delimiter',',')