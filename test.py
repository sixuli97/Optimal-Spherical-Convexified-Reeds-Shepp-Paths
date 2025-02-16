from CRS_on_sphere_opt import opt_path_gen, CRS_plot
import numpy as np
from scipy.spatial.transform import Rotation as R

seed = 50156

rng = np.random.default_rng(seed)  # Create a random number generator with the seed

# Generate a single random rotation matrix
terminal_config = R.random(random_state=rng).as_matrix()
U_max = 3
ini_config = np.eye(3)

feas,opt = opt_path_gen(U_max, ini_config, terminal_config)    

CRS_plot(feas, opt, ini_config, terminal_config)


    