# Template

workdirs='
Gather/tjubd6/Personalized/Compare
'
worklevel=1
summary_path='Summary/Personalized/Compare'
summary_name='all'
metrics='Test Error,Err0'
valid_loss='Valid_Loss'
# order='method'
column='+method peoples label_source 200'
new_column='method peoples label_source Error'
# replace=''

python Stat/Summary.py  $workdirs -o $summary_path -v $valid_loss -l $worklevel
python Stat/stat_base.py $summary_path -m "$metrics"
python Stat/clean_summary.py -p $summary_path/Summary_table.csv -n $summary_name \
                            -c $column -nc $new_column # -r $replace

# workdirs='
# Gather/tjubd6/Compare/AdaptingRoNIN
# Gather/tjubd6/Compare/CoTeachingRoNIN
# Gather/tjubd6/Compare/AUXRoNIN
# Output/Compare/BaseRes
# Gather/tjubd6/Pretrain/RoNIN
# Output/Pretrain/RoNIN
# '
# summary_path='Summary/Compare/Base'
# summary_name='all'
# metrics='Test Error,Err0'
# valid_loss='Valid_Loss'
# # order='method'
# column='exp_run peoples label_source 200'
# new_column='method peoples label_source Error'
# replace='method:Gather/tjubd6/Compare/: method:Output/Compare/: label_source:default:PhoneLabel method:Gather/tjubd6/Pretrain/: method:Output/Pretrain/:'

# python Stat/Summary.py  $workdirs -o $summary_path -v $valid_loss
# python Stat/stat_base.py $summary_path -m "$metrics"
# python Stat/clean_summary.py -p $summary_path/Summary_table.csv -n $summary_name \
#                             -c $column -nc $new_column -r $replace
                            
# workdirs='
# Gather/tjubd6/Pretrain/Base
# Gather/tjubd6/Pretrain/RoNIN
# Output/Pretrain/Base
# Output/Pretrain/LIONet
# Output/Pretrain/RoNIN
# '
# summary_path='Summary/Pretrain/Base'
# summary_name='all'
# metrics='Err0'
# valid_loss='Valid_Loss'
# # order='method'
# column='exp_run peoples label_source 200'
# new_column='method peoples label_source Error'
# replace='method:Gather/tjubd6/Pretrain/: method:Output/Pretrain/: label_source:default:PhoneLabel'

# python Stat/Summary.py  $workdirs -o $summary_path -v $valid_loss
# python Stat/stat_base.py $summary_path -m "$metrics"
# python Stat/clean_summary.py -p $summary_path/Summary_table.csv -n $summary_name \
#                             -c $column -nc $new_column -r $replace

# workdirs='
# Gather/tjubd5/AdaptV2.2/Adapting
# '
# summary_path='Summary/AdaptV2.2'
# summary_name='all'
# metrics='Test Error'
# valid_loss='Train_Weighted_Pred_Loss'
# order='method'
# column='+method dataset 200'
# new_column='method dataset Error'

# python Stat/Summary.py  $workdirs -o $summary_path -v $valid_loss
# python Stat/stat_base.py $summary_path -m "$metrics"
# python Stat/clean_summary.py -p $summary_path/Summary_table.csv -n $summary_name --order +method $order \
#                             -c $column -nc $new_column

# workdirs='
# Gather/tjubd12/CJYLall_Clean_NoDA/Base
# '
# summary_path='Summary/CJYLall_Clean_NoDA'
# summary_name='all'
# metrics='Err0'
# valid_loss='Valid_Loss'
# order='method'
# column='+method max_err_limit 200'
# new_column='method max_err_limit Error'
# table='Error max_err_limit'
# # replace='method:BaseModel_GT:BaseModel-GT method:Plus:+ method:NAL:AUX method:CAUX:CNAL method:CoAdapt:EasyTrack method:EasyTrack_wo_:no method:CoTeaching:Co-teaching'

# python Stat/Summary.py  $workdirs -o $summary_path -v $valid_loss
# python Stat/stat_base.py $summary_path -m "$metrics"
# python Stat/clean_summary.py -p $summary_path/Summary_table.csv -n $summary_name --order +method $order \
#                             -c $column -nc $new_column -t $table #-r $replace

# workdirs='
# Gather/autoDL/AdaptV2/Adapting
# Gather/autoDL2/AdaptV2/Adapting
# Gather/tjubd5/AdaptV2/Adapting
# Gather/tjubd6/AdaptV2/Adapting
# Gather/autoDL2/AdaptV1/Adapting
# Gather/autoDL2/AdaptV1/CoTeaching
# Gather/autoDL3/AdaptV1/CoTeaching
# Gather/tjubd6/AdaptV1/LIONet
# Gather/tjubd6/AdaptV1/RoNIN
# Gather/autoDL/AdaptV1/Base
# '
# summary_path='Summary/AdaptV2'
# summary_name='all'
# metrics='Test Error,Err0'
# valid_loss='Train_Weighted_Pred_Loss Train_Weighted_Pred_Loss Train_Weighted_Pred_Loss Train_Weighted_Pred_Loss Valid_Loss Valid_Loss Valid_Loss Valid_Loss Valid_Loss Valid_Loss'
# order='method BaseModel_GT CoAdapt CoTeaching CoTeachingPlus JoCoR NAL BaseModel'
# column='+method dataset 200'
# new_column='method dataset Error'
# table='Error dataset'
# replace='method:BaseModel_GT:BaseModel-GT method:Plus:+ method:NAL:AUX method:CAUX:CNAL method:CoAdapt:EasyTrack method:EasyTrack_wo_:no method:CoTeaching:Co-teaching'

# # python Stat/Summary.py  $workdirs -o $summary_path -v $valid_loss
# # python Stat/stat_base.py $summary_path -m "$metrics"
# python Stat/clean_summary.py -p $summary_path/Summary_table.csv -n $summary_name --order +method $order \
#                             -c $column -nc $new_column -t $table -r $replace

# workdirs='
# Gather/autoDL/AdaptV1/Base
# Gather/autoDL3/AdaptV1/Adapting
# Gather/autoDL2/AdaptV1/CoTeaching
# Gather/tjubd5/AdaptV1/Adapting
# Gather/tjubd12/AdaptV1/Adapting
# Gather/tjubd6/AdaptV1/LIONet
# Gather/tjubd6/AdaptV1/RoNIN
# '
# summary_path='Summary/AdaptV1'
# summary_name='all'
# metrics='Test Error,Err0'
# valid_loss='Valid_Loss Train_Weighted_Pred_Loss Valid_Loss Train_Weighted_Pred_Loss Train_Weighted_Pred_Loss Valid_Loss Valid_Loss'
# order='method BaseModel_GT CoAdapt CoTeaching CoTeachingPlus JoCoR BaseModel'
# column='+method dataset 200'
# new_column='method dataset Error'
# table='Error dataset'
# #replace=''

# python Stat/Summary.py  $workdirs -o $summary_path -v $valid_loss
# python Stat/stat_base.py $summary_path -m "$metrics"
# python Stat/clean_summary.py -p $summary_path/Summary_table.csv -n $summary_name --order +method $order \
#                             -c $column -nc $new_column -t $table #-r $replace
