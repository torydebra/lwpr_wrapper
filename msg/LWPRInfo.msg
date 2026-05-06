#About data seen by the whole model
int32 n_data
## for each input dimension 
float64[] mean_data
float64[] var_data

#About LAST prediction
float64[] prediction
float64[] prediction_conf
float64[] prediction_maxw

## RFS (vector because one group of rf for each output)
int32[] num_rfs

lwpr_wrapper/RfsInfo[] rfs_info #SIZE: number of output
