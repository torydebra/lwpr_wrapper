# data from the LWPR_ReceptiveFieldObject

float64 weight

#PLS
## number of PLS regression directions 
int32 n_reg
## weighted number of training data each PLS direction has seen  has seen (size: nReg)
float64[] num_data

# About data seen by this single RFl
## the weighted mean of the input data, as seen by the receptive field (nIn)
float64[] mean_data #size: number of input of lwpr
float64[] var_data

# RF shape
## the center vector of the receptive field (nIn) (normalized!)
float64[] center
## the distance metric of the receptive field which is nIn x nIn 
## and geometrically (by its eigenvalues) is the shape of the gaussian/bisquare (normalized!)
float64[] d

# Other
## whether this receptive field is trustworthy (has seen sufficient data)
bool trustworthy


