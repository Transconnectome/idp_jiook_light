data=read.csv('data_seonjoo_dictionary.csv',header=T)
rid_179=read.csv('RID_179.csv')
data_179=merge(data,rid_179,by="RID")
write.csv(data_179,file="demo_connectome.csv",row.names = FALSE)


biomarker=read.csv(file="UPENNBIOMK_MASTER.csv",header=T,sep=",")
biomarker_bl=biomarker[biomarker$VISCODE=='bl',]
biomarker_bl_median=biomarker_bl[biomarker_bl$BATCH=='MEDIAN',]

connectome_demo_biomarker=merge(data_179,biomarker_bl_median,by="RID",all.x=T)
write.csv(file="freesurfer_demo2_biomarker.csv", freesurfer_demo_biomarker,row.names=F)
