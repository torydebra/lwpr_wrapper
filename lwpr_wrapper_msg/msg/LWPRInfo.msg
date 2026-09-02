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
lwpr_wrapper_msg/RfsInfo[] rfs_info_per_output #SIZE: number of output
