
import OptimalPolicyPI
import OptimalPolicyVI
from HelpersFunctions import FourRoom, plot_grid_world


def _plotGridWorld():
	goal = 81
	gamma = 0.9
	env = FourRoom(goal)
    
    #### Different optimal policy algorithms
	#Pi, v = OptimalPolicyPI.policyIteration(env, gamma)
	Pi, v = OptimalPolicyVI.valueIteration(env, gamma)  

	print(f"State-value for the goal state is {v[goal, 0]:.2f}")
	plot_grid_world(env, Pi, v)
    


if __name__ == "__main__":

	_plotGridWorld()
