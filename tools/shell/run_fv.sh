#!/usr/bin/bash
# 02-Nov-2022
# Runs the Fleming-Viot estimator of the blocking probability triggered by run_FV.py on a set of parameters of interest.
# Goal: Analyze how different parameters affect the estimation performance.
# TO BE RUN AT THE IRIT CLUSTER where the python executable is called `python3` (NOT `python`)
#
# Example:
# ./run_fv.sh 1 N 5 "0.2 0.4 0.6 0.8" 3 IDK5
# 
# Ref:
# - bash documentation: https://devdocs.io/bash
# - bash documentation about arrays: https://devdocs.io/bash/arrays --> Here I learned how to declare an array and thus be able to get its number of elements when input from the command line!

declare -a J_FACTORS

# Input parameters
NSERVERS="$1"		# Number of servers in the system to simulate
ANALYSIS_TYPE="$2"      # Type of analysis: either "N" for the impact of number of particles in the estimation of Phi(t,K) or "T" for the impact of the number of arrival events in the estimation of E(T_A)
K="$3"		       	# K: capacity of the system
J_FACTORS=($4)		# J factors to consider such that J = round( factor*K ). It should be given as a blank separated set of values (e.g. "0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8").
#N="$5"			# N: number of particles in the FV system. This must be only ONE. If we want to simulate on more than one value of N, set their values when calling analyze_convergence()
R="$5"			# Number of replications to run for each experiment
ID="$6"			# ID to use for this run (so that we can easily put together all the output result files for analysis) (e.g. "ID1")


errors_rel=(1.0 0.8 0.6 0.4 0.2 0.1 0.05)	# These relative error values were taken from the ones used in run_FV.py when defining the values of the variable to analyze (look for `errors_rel` variable)

# Reference about FOR loops in bash: https://www.geeksforgeeks.org/bash-scripting-for-loop/
n_factors="${#J_FACTORS[@]}"	# This does not work as expected... it always returns 1... WHY?? Ref for length of array: https://www.cyberciti.biz/faq/finding-bash-shell-array-length-elements/
n_errors="${#errors_rel[@]}"
let n_cases=$n_factors*$n_errors
i=0
for JF in ${J_FACTORS[@]}
do
	# Iterate on each expected error value defined in the above array
	for err in ${errors_rel[@]}
	do
		let i++
		echo
		echo "*********************"
		echo "Running case $i of $n_cases..."
		echo "Parameters:"
		echo "nservers=$NSERVERS"
		echo "analysis_type=$ANALYSIS_TYPE"
		echo "K=$K"
		echo "JF=$JF"
		echo "error=$err"
		echo "R=$R"
		echo "ID=$ID"
		python3 ~/projects/Python/lib/run_FV.py $NSERVERS $ANALYSIS_TYPE $K $JF $err $R 1 True 10 5 2 save False $ID False
		## Note: the final parameters above, after $R, are:
		## - test number to run
		## - whether to discard parameter values that are smaller than the minimum from the set of parameters to try (e.g. N_required < N_min based on the expected error requirement)
		##   Note that parameter values larger than the maximum allowed are always discarded (because o.w. the execution time would be too long).
		## - BITS: Burn-in time steps to consider until stationarity can be assumed for the estimation of expectations (e.g. E(T) (return cycle time) in Monte-Carlo, E(T_A) (reabsorption cycle time) in Fleming-Viot)
		## - MINCE: Minimum number of cycles to be used for the estimation of expectations (e.g. E(T) in Monte-Carlo and E(T_A) in Fleming-Viot)
		## - Number of methods to run: 1 (only FV), 2 (FV & MC)
		## - Either "nosave" or anything else for saving the results and log
		## - Whether to use a suffix indicating the execution date time in the results filenames
		## - The ID to identify the output files with, in addition to the execution datetime (if requested)
		##
		## Example of ONE execution of the Python program run_FV.py:
		## python run_FV.py 1 N 5 0.2 0.4 3 1 10 5 2 save False ID1
	done
done
