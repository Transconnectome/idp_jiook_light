d_conn_aparc_count=read.csv("conn_aparc_count.csv",header=T,skip =0)
d_mor=read.csv("data_adni_mor.csv",header=T,skip=0)
d_mor_conn=merge(d_mor,d_conn_aparc_count,by="RID",all.x=F)

d_combine_new_biomarker_correct_1=read.csv("combine_new_biomarker_correct_1.csv",header=T)
write.csv(file="d_combine_new_biomarker_correct_1.csv", d_combine_new_biomarker_correct_1,row.names=F)

d_combine_new_biomarker_correct_mor2=merge(d_combine_new_biomarker_correct_1,d_mor,by="RID",all.x=F)
write.csv(file="d_combine_new_biomarker_correct_mor2.csv", d_combine_new_biomarker_correct_mor2,row.names=F)


d_combine_new_biomarker_correct_mor_conn3=merge(d_combine_new_biomarker_correct_1,d_mor_conn,by="RID",all.x=F)
write.csv(file="d_combine_new_biomarker_correct_mor_conn3.csv", d_combine_new_biomarker_correct_mor_conn3,row.names=F)
