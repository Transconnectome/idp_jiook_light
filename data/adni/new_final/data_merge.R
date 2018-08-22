d_conn_aparc_count=read.csv("conn_aparc_count.csv",header=T,skip =0)
d_mor=read.csv("data_adni_mor.csv",header=T,skip=0)
d_mor_conn=merge(d_mor,d_conn_aparc_count,by="RID",all.x=F)

d_combine_new_biomarker_correct_1=read.csv("combine_new_biomarker_correct_1.csv",header=T)
d_combine_new_biomarker_correct_1=d_combine_new_biomarker_correct_1[,]
write.csv(file="d_combine_new_biomarker_correct_1.csv", na = "NaN",d_combine_new_biomarker_correct_1,row.names=F)

d_combine_new_biomarker_correct_mor_2=merge(d_combine_new_biomarker_correct_1,d_mor,by="RID",all.x=F)
write.csv(file="d_combine_new_biomarker_correct_mor_2.csv", na = "NaN",d_combine_new_biomarker_correct_mor_2,row.names=F)


d_combine_new_biomarker_correct_mor_conn_3=merge(d_combine_new_biomarker_correct_1,d_mor_conn,by="RID",all.x=F)
write.csv(file="d_combine_new_biomarker_correct_mor_conn_3.csv", na = "NaN", d_combine_new_biomarker_correct_mor_conn_3,row.names=F)
