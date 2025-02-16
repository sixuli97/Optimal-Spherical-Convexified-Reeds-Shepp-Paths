from CRS_on_sphere_opt import opt_path_gen, CRS_plot
import numpy as np
terminal_config = np.array([[0.804977,   -0.59221622,  0.03594413],
 		            [-0.56946107, -0.75420323,  0.32694278],
 		            [-0.16651164, -0.2836502,  -0.94436033]]) # 🎯 Random desired terminal configuration on a sphere (in SO(3))
ini_config = np.eye(3)  # 🏁 Example start configuration on a sphere (in SO(3))
U_max = 3 # ⚙️ Example maximum truning rate

feas,opt = opt_path_gen(U_max, ini_config, terminal_config) # ✅ Solve for the optimal path and all feasible paths

CRS_plot(feas, opt, ini_config, terminal_config) # 🎨 Plot the optimal path and all feasible paths on a unit sphere