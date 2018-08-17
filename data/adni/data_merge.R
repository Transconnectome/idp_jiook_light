setwd("/Users/posnerlab/GitHub/idp_jiook_light/data/adni")

d1=read.csv("combine_MConly_wo_dropouts_1_demo.csv",heade=T, skip=0)
d2=read.csv("combine_new_biomarker_correct_wo_dropouts.csv",heade=T, skip=0)
d2_select<-d2[1:6]

data<-Reduce(function(...) merge(..., all = F,by="RID",all.x=T),
                  list(d1, d2_select))
write.csv(data,file="combine_MConly_wo_dropouts_0_biomarker.csv",row.names=F)