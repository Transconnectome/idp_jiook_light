setwd("/Volumes/Pegasus/Bigdata/ADNI/1018_DTI_MRI_ADNI2_screening_baseline/fs/group_freesurfer")
filelist = list.files(path=".",pattern="*txt")

as.data.frame(read.delim(file=filelist[1]))

data<-merge(as.data.frame(read.delim(file=filelist[1])),
            as.data.frame(read.delim(file=filelist[2])),
            by="RID")
data<-merge(data,as.data.frame(read.delim(file=filelist[3])),by="RID")
data<-merge(data,as.data.frame(read.delim(file=filelist[4])),by="RID")
data<-merge(data,as.data.frame(read.delim(file=filelist[5])),by="RID")
data<-merge(data,as.data.frame(read.delim(file=filelist[6])),by="RID")
data<-merge(data,as.data.frame(read.delim(file=filelist[7])),by="RID")
data<-merge(data,as.data.frame(read.delim(file=filelist[8])),by="RID")
data<-merge(data,as.data.frame(read.delim(file=filelist[9])),by="RID")
data<-merge(data,as.data.frame(read.delim(file=filelist[10])),by="RID")
data<-merge(data,as.data.frame(read.delim(file=filelist[11])),by="RID")
data<-merge(data,as.data.frame(read.delim(file=filelist[12])),by="RID")
data<-merge(data,as.data.frame(read.delim(file=filelist[13])),by="RID")
data<-merge(data,as.data.frame(read.delim(file=filelist[14])),by="RID")
data<-merge(data,as.data.frame(read.delim(file=filelist[15])),by="RID")
data<-merge(data,as.data.frame(read.delim(file=filelist[16])),by="RID")
data<-merge(data,as.data.frame(read.delim(file=filelist[17])),by="RID")
data<-merge(data,as.data.frame(read.delim(file=filelist[18])),by="RID")

write.csv(file="group_freesurfer_.csv",data,row.names=F)

RID=read.csv(file="freesurfer_RID.csv")
demo=read.csv(file="data_seonjoo_dictionary.csv")
freesurfer_demo=merge(RID,demo,by="RID")
write.csv(file="freesurfer_demo.csv", freesurfer_demo,row.names=F)

biomarker=read.csv(file="UPENNBIOMK_MASTER.csv",header=T,sep=",")
biomarker_bl=biomarker[biomarker$VISCODE=='bl',]
biomarker_bl_median=biomarker_bl[biomarker_bl$BATCH=='MEDIAN',]

freesurfer_demo_biomarker=merge(freesurfer_demo,biomarker_bl_median,by="RID",all.x=T)
write.csv(file="freesurfer_demo2_biomarker.csv", freesurfer_demo_biomarker,row.names=F)




####
data_adni_freesurfer=read.csv("freesurfer_all.csv")
data_adni_freesurfer_demo=read.csv("freesurfer_demo2_biomarker.csv")
data_adni_mor=merge(data_adni_freesurfer_demo,data_adni_freesurfer,by="RID")
write.csv(file="data_adni_mor.csv",data_adni_mor,row.names=F)