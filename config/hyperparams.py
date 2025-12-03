EPOCHS = 200
LR = 5e-4
ORDER = 2
V_MAX = 15.0         
A_MAX = 0.5           
U_MAX = 0.2          
A_NU_MAX = 2.0
# forbid backward motion
LAMBDA_BACK = 200.0      
# make velocity smooth
LAMBDA_SMOOTH = 50.0     
#  match true velocity
LAMBDA_MATCH = 50.0       
# penalize v > V_MAX
LAMBDA_VMAX = 50.0
# Prefer monotonicity       
LAMBDA_MONO = 200
