% ADNI DATA COMBINATION 

%connectome data first; 
clc
clear all
cd('/Users/posnerlab/Documents/bitbucket/idp_jiook_light/data/adni/group_connectome')
load('connectome_aparc_count.mat')
load('connectome_aparc_length.mat')
load('connectome_aparc2009_count.mat')
load('connectome_aparc2009_length.mat')

A=reshape(connectome_aparc0x2Baseg_count, [84*84,179]);
A=A';

B=reshape(connectome_aparc0x2Baseg_length, [84*84,179]);
B=B';

C=reshape(connectome_aparc0x2Ea2009s0x2Baseg_count, [164*164,179]);
C=C';

D=reshape(connectome_aparc0x2Ea2009s0x2Baseg_length, [164*164,179]);
D=D';

T=horzcat(A,B,C,D);
T=A; % LET's take only A matrix for now

% read the 84*84 feature names 
%FS2009=importdata('fs_a2009s.txt') ;
%FS=importdata('fs_default.txt');

% read 164*164 feature names 

% remove 0 columns 
T( :, ~any(T,1) ) = [];  %columns

%csvwrite('conn_raw.csv',T)
csvwrite('conn_raw_aparc_count.csv',T)
%%
% add the demographic information 

[ID, text,head]=importdata('connectome_RID_179.csv');% corresponding to the data above

% read the raw demographic and rank according to ID
load('connectome_demo.mat')
Raw=connectomedemo;
M=connectomedemo;
 %M(1,:)=Raw(1,:);
for i=1:length(ID.data)

  for j=1:179
      if  ID.data(i,1)==Raw{j,1}
          M(i,:)=Raw(j,:);
      else
      end
  end
  
end
T1=array2table(T);
T_ID=array2table(ID.data);T_ID.Properties.VariableNames={'RID'};
% C=outerjoin(M,T1,'MergeKeys', true)

combined_table=[T_ID, T1];
writetable(combined_table,'conn_aparc_count.csv')