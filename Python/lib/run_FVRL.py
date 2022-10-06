# -*- coding: utf-8 -*-
"""
Created on Sun Jul 11 08:42:57 2022

@author: Daniel Mastropietro
@description: Runs the FVRL algorithm to learn the optimum parameter of a parameterized policy.
"""

import os
import sys
import warnings
import tracemalloc

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.ticker import MaxNLocator
from timeit import default_timer as timer

from Python.lib.agents.learners.continuing.fv import LeaFV
from Python.lib.agents.learners.continuing.mc import LeaMC
from Python.lib.agents.learners.policies import LeaPolicyGradient

from Python.lib.agents.policies.parameterized import PolQueueTwoActionsLinearStep

from Python.lib.agents.queues import AgeQueue

from Python.lib.environments.queues import Actions, rewardOnJobRejection_ExponentialCost

from Python.lib.simulators import LearningMethod, define_queue_environment_and_agent
from Python.lib.simulators.queues import compute_nparticles_and_nsteps_for_fv_process, \
    compute_rel_errors_for_fv_process, LearningMode, SimulatorQueue

from Python.lib.utils.basic import aggregation_bygroups, is_scalar, show_exec_params
from Python.lib.utils.computing import stationary_distribution_birth_death_process
import Python.lib.utils.plotting as plotting


PLOT_GRADIENT = False                   # Whether to plot the estimated gradient of the average reward at each learning step
PLOT_TRAJECTORY = True                  # Whether to show the theta-learning trajectory
# Constants that are used for information purposes only: each section started with checking the value of the following
# constants are currently (11-Jul-2022) written to be executed manually, as the data to plot is read from files specified each time.
PLOT_RESULTS_TOGETHER = False           # Whether to plot the theta-learning trajectories of the FVRL and the MC algorithms on the same graph
PLOT_RESULTS_PAPER = False              # Whether to generate the plots with the theta learning for the paper


# ---------------------------- Auxiliary functions ---------------------------#
def run_simulation_policy_learning(simul, dict_params_simul, dict_info,
                                   dict_params_info: dict = {'plot': False, 'log': False},
                                   params_read_from_benchmark_file=False, seed=None, verbose=False):
    set_required_entries_info = {'case', 'ncases', 'learning_method', 'exponent',
                                 'rhos', 'K_true', 'K', 'error_rel_phi', 'error_rel_et',
                                 'alpha_start', 'adjust_alpha', 'min_time_to_update_alpha', 'alpha_min'}
    if not set_required_entries_info.issubset(dict_info.keys()):
        raise ValueError("Missing entries in the dict_info dictionary: {}" \
                         .format(set_required_entries_info.difference(dict_info.keys())))

    if not params_read_from_benchmark_file:
        error_rel_phi_real, error_rel_et_real = compute_rel_errors_for_fv_process(dict_info['rhos'],
                                                                                  dict_info['K'],
                                                                                  dict_params_simul['nparticles'],
                                                                                  dict_params_simul['t_sim'],
                                                                                  dict_params_simul[
                                                                                      'buffer_size_activation_factor'])

        print("\n--> CASE {} of {}: theta_true={:.3f} (K_true={}), theta={:.3f} (K={}), J/K={:.3f}," \
              " exponent={}: N={} (err_nom={:.1f}%, err={:.1f}%), T={} (err_nom={:.1f}%, err={:.1f}%)" \
              .format(dict_info['case'], dict_info['ncases'], dict_params_simul['theta_true'], dict_info['K_true'],
                      dict_params_simul['theta_start'], dict_info['K'],
                      dict_params_simul['buffer_size_activation_factor'],
                      dict_info['exponent'], dict_params_simul['nparticles'], dict_info['error_rel_phi'] * 100,
                      error_rel_phi_real * 100,
                      dict_params_simul['t_sim'], dict_info['error_rel_et'] * 100, error_rel_et_real * 100))
    else:
        print("\n--> CASE {} of {}: theta_true={:.3f} (K_true={}), theta={:.3f} (K={}), J/K={:.3f}," \
              " exponent={}: N={}, T={})" \
              .format(dict_info['case'], dict_info['ncases'], dict_params_simul['theta_true'], dict_info['K_true'],
                      dict_params_simul['theta_start'], dict_info['K'],
                      dict_params_simul['buffer_size_activation_factor'],
                      dict_info['exponent'], dict_params_simul['nparticles'], dict_params_simul['t_sim']))

    # Show execution parameters
    params = dict({
        '1(a)-System-#Servers': simul.getEnv().getNumServers(),
        '1(b)-System-JobClassRates': simul.getEnv().getJobClassRates(),
        '1(c)-System-ServiceRates': simul.getEnv().getServiceRates(),
        '1(d)-System-TrueTheta': dict_params_simul['theta_true'],
        '1(e)-System-TrueK': dict_info['K_true'],
        '2(a)-Learning-Method': dict_info['learning_method'],
        '2(b)-Learning-Method#Particles and % Rel Error Phi': (dict_params_simul['nparticles'], dict_info['error_rel_phi'] * 100),
        '2(c)-Learning-Method#TimeSteps and % Rel Error E(T)': (dict_params_simul['t_sim'], dict_info['error_rel_et'] * 100),
        '2(d)-Learning-LearningMode': simul.dict_learning_params['mode'].name,
        '2(e)-Learning-ThetaStart': dict_params_simul['theta_start'],
        '2(f)-Learning-#Steps': simul.getNumLearningSteps(),
        '2(g)-Learning-SimulationTimePerLearningStep': dict_params_simul['t_sim'],
        '2(h)-Learning-AlphaStart': dict_info['alpha_start'],
        '2(i)-Learning-AdjustAlpha?': dict_info['adjust_alpha'],
        '2(j)-Learning-MinEpisodeToAdjustAlpha': dict_info['min_time_to_update_alpha'],
        '2(k)-Learning-AlphaMin': dict_info['alpha_min'],
    })
    show_exec_params(params)

    _, _, df_learning = simul.run(dict_params_simul,
                                  dict_params_info=dict_params_info,
                                  dict_info=dict_info,
                                  seed=seed, verbose=verbose)

    return df_learning['theta_next'].iloc[-1], df_learning
# ---------------------------- Auxiliary functions ---------------------------#


# Default execution parameters when no arguments are given in the command line
# Example of execution from the command line:
# python simulators.py 50 FV False 1.0 23.0 33.9 0.5 1.0 1.0
print("User arguments: {}".format(sys.argv))
if len(sys.argv) == 1:  # Only the execution file name is contained in sys.argv
    sys.argv += [50]    # t_learn: Number of learning steps
    sys.argv += ["FV"]  # learning_method: estimation method: "FV" or "MC";
                        # when learning_method = "MC", we expect there exists a benchmark file called benchmark_fv.csv
                        # that defines how many events were observed during the equivalent FV learning method
                        # (for fair comparison). If one does not exist, the Monte-Carlo simulation wil be run
                        # no its own and no comparison is expected to be carried out with a previously FV simulation.
    sys.argv += [False] # clipping: whether to use clipping: False or True
    sys.argv += [1.0]   # clipping_value: clipping value when clipping = True
if len(sys.argv) == 5:  # Only the 4 required arguments are given by the user (recall that the first argument is the program name)
    # Note that, if the user didn't pass any arguments, i.e. len(sys.argv) == 1 above, at this stage sys.argv has length 5
    # because the first 4 required parameters were just defined.
    # There is NO check that the user didn't pass the full set of 4 required parameters, i.e. if they pass 2 parameters
    # the parameter parsing process will fail because not all arguments are defined in sys.argv.
    sys.argv += [20.0]  # 23.0]  # theta_true: true theta (only one value is allowed)
    sys.argv += [1.1]   # 39.1]   # theta_start: non-integral initial theta value for the learning process (only one value is allowed)
    sys.argv += [0.5]   # J_factor: fraction J/K to use in the FV learning method
    sys.argv += [4.0]   # error_rel_phi: expected relative error for the estimation of Phi(t,K) in the FV learning method (1.0 means 100%) --> it defines the number of particles to use
    sys.argv += [4.0]   # error_rel_phi: expected relative error for the estimation of E(T) in the FV learning method (1.0 means 100%) --> it defines the number of arrival events to observe in the MC-based simulation to estimate E(T)
    sys.argv += ["nosave"]  # Either "nosave" or anything else for saving the results and log
print("Parsed user arguments: {}".format(sys.argv))
print("")
assert len(sys.argv) == 11, "The number of parsed arguments is 11 ({})".format(len(sys.argv))

# -- Parse user arguments
t_learn = int(sys.argv[1])
learning_method = LearningMethod.FV if sys.argv[2] == "FV" else LearningMethod.MC
clipping = bool(sys.argv[3]) == True
clipping_value = float(sys.argv[4])
theta_true = float(sys.argv[5])
theta_start = float(sys.argv[6])
J_factor = float(sys.argv[7])
error_rel_phi = float(sys.argv[8])
error_rel_et = float(sys.argv[9])
create_log = sys.argv[10] != "nosave"
save_results = sys.argv[10] != "nosave"

print("Execution parameters:")
print("t_learn={}".format(t_learn))
print("learning_method={}".format(learning_method.name))
print("clipping={}".format(clipping))
print("clipping_value={}".format(clipping_value))
print("theta_true={}".format(theta_true))
print("theta_start={}".format(theta_start))
print("J_factor={}".format(J_factor))
print("error_rel_phi={}".format(error_rel_phi))
print("error_rel_et={}".format(error_rel_et))
print("create_log={}".format(create_log))
print("save_results={}".format(save_results))

# Look for memory leaks
# Ref: https://pythonspeed.com/fil/docs/fil/other-tools.html (from the Fil profiler which also seems interesting)
# Doc: https://docs.python.org/3/library/tracemalloc.html
# tracemalloc.start()

start_time_all = timer()

# ---------------------- OUTPUT FILES --------------------#
# create_log = False;    # In case we need to override the parameter received as argument to the script
logsdir = "../../RL-002-QueueBlocking/logs/RL/single-server"
# save_results = True;   # In case we need to override the parameter received as argument to the script
resultsdir = "../../RL-002-QueueBlocking/results/RL/single-server"
# ---------------------- OUTPUT FILES --------------------#

# -- Parameters defining the environment, policies, learners and agent
# Learning parameters for the value function V
gamma = 1.0

# Learning parameters for the policy P
if learning_method == LearningMethod.FV:
    learnerV = LeaFV
    benchmark_file = None
    plot_trajectories = False
    symbol = 'g-'
else:
    # Monte-Carlo learner is the default
    learnerV = LeaMC
    benchmark_file = os.path.join(os.path.abspath(resultsdir), "benchmark_fv.csv")
    if not os.path.exists(benchmark_file):
        warnings.warn("Benchmark file 'benchmark_fv.csv' does not exist. The Monte-Carlo simulation will run without any reference to a previously run Fleming-Viot simulation")
        benchmark_file = None
    plot_trajectories = False
    symbol = 'g-'
fixed_window = False
alpha_start = 10.0  # / t_sim  # Use `/ t_sim` when using update of theta at each simulation step (i.e. LeaPolicyGradient.learn_TR() is called instead of LeaPolicyGradient.learn())
adjust_alpha = False  # True
func_adjust_alpha = np.sqrt
min_time_to_update_alpha = 0  # int(t_learn / 3)
alpha_min = 0.1  # 0.1

dict_params = dict({'environment': {'capacity': np.Inf,
                                    'nservers': 1,  # 3
                                    'job_class_rates': [0.7],  # [0.8, 0.7]
                                    'service_rates': [1.0],  # [1.0, 1.0, 1.0]
                                    'policy_assignment_probabilities': [[1.0]],  # [[0.5, 0.5, 0.0], [0.0, 0.5, 0.5]] )
                                    'reward_func': rewardOnJobRejection_ExponentialCost,
                                    'rewards_accept_by_job_class': None
                                    },
                    'policy': {'parameterized_policy': PolQueueTwoActionsLinearStep,
                               'theta': 1.0  # This value is dummy in the sense that it will be updated below
                               },
                    'learners': {'V': {'learner': learnerV,
                                       'params': {'gamma': 1}
                                       },
                                 'Q': {'learner': None,
                                       'params': {}},
                                 'P': {'learner': LeaPolicyGradient,
                                       'params': {'alpha_start': alpha_start,
                                                  'adjust_alpha': adjust_alpha,
                                                  'func_adjust_alpha': func_adjust_alpha,
                                                  'min_time_to_update_alpha': min_time_to_update_alpha,
                                                  'alpha_min': alpha_min,
                                                  'fixed_window': fixed_window,
                                                  'clipping': clipping,
                                                  'clipping_value': clipping_value,
                                                  }
                                       }
                                 },
                    'agent': {'agent': AgeQueue}
                    })
env_queue, rhos, agent = define_queue_environment_and_agent(dict_params)

# -- Simulation parameters that are common for ALL parameter settings
# t_learn is now defined as input parameter passed to the script
# 2022/01/14: t_learn = 10 times the optimum true theta so that we are supposed to reach that optimum under the REINFORCE_TRUE learning mode with decreasing alpha
# t_learn = 800 #100 #800 #100 #198 - 91 #198 #250 #50
seed = 1313  # 1317 #1717 #1313  #1859 (for learning step 53+91=144) #1769 (for learning step 53, NOT 52 because it took too long) #1717
verbose = False
dict_learning_params = dict({'mode': LearningMode.REINFORCE_TRUE, 't_learn': t_learn})
dict_params_info = dict({'plot': False, 'log': False})

# Simulator object
simul = SimulatorQueue(env_queue, agent, dict_learning_params,
                       log=create_log, save=save_results, logsdir=logsdir, resultsdir=resultsdir, debug=False)

# Open the file to store the results
if save_results:
    # Initialize the output file with the results with the column names
    simul.fh_results.write("case,t_learn,theta_true,theta,theta_next,K,J/K,J,exponent,N,T,err_phi,err_et,seed,E(T),n_cycles,max_time_surv,Pr(K-1),Pr(K),Q_diff(K-1),Q_diff(K),alpha,V,gradV,n_events_mc,n_events_fv,n_trajectories_Q\n")

# Run the simulations, either from parameters defined by a benchmark file or from parameters defined below
if benchmark_file is None:
    # -- Iterate on each set of parameters defined here
    # theta_true_values = np.linspace(start=1.0, stop=20.0, num=20)
    # When defining the theta values we specify the blocking size K and we substract 1. Recall that K = ceiling(theta+1)
    # So, if we want K = 35, we can set theta somewhere between 33+ and 34, so we define e.g. theta = 34.9 - 1
    # Note that the list of theta values can contain more than one value, in which case a simulation will be run for each of them
    theta_true_values = [theta_true]  # [24.0 - 1] #[20.0 - 1]  # [32.0-1, 34.0-1, 36.0-1] #[10.0-1, 15.0-1, 20.0-1, 25.0-1, 30.0-1]  # 39.0
    theta_start_values = [theta_start]  # [34.9 - 1] #[30.0 - 1] #[20.0 - 1, 25.0 - 1]
    assert len(theta_true_values) == len(theta_start_values), \
            "The number of true theta values ({}) and start theta values ({}) should be the same" \
            .format(len(theta_true_values), len(theta_start_values))
    J_factor_values = [0.3]  # [0.2, 0.3, 0.5]  # [0.2, 0.3, 0.5, 0.7]
    NT_exponents = [0]  # [-2, -1, 0, 1]  # Exponents to consider for different N and T values as in exp(exponent)*N0, where N0 is the reference value to achieve a pre-specified relative error
    # Accepted relative errors for the estimation of Phi and of E(T_A)
    # They define respectively the number of particles N and the number of arrival events T to observe in each learning step.
    error_rel_phi = [error_rel_phi]  # [1.0] #0.5
    error_rel_et = [error_rel_et]  # [1.0] #0.5

    # Output variables of the simulation
    case = 0
    ncases = len(theta_true_values) * len(theta_start_values) * len(J_factor_values) * len(NT_exponents)
    theta_opt_values = np.nan * np.ones(
        ncases)  # List of optimum theta values achieved by the learning algorithm for each parameter setting
    for i, theta_true in enumerate(theta_true_values):
        print(
            "\nSimulating with {} learning on a queue environment with optimum theta (one less the deterministic blocking size) = {}".format(
                learning_method.name, theta_true))

        # Set the number of learning steps to double the true theta value
        # Use this ONLY when looking at the MC method and running the learning process on several true theta values
        # to see when the MC method breaks... i.e. when it can no longer learn the optimum theta.
        # The logic behind this choice is that we start at theta = 1.0 and we expect to have a +1 change in
        # theta at every learning step, so we would expect to reach the optimum value after about a number of
        # learning steps equal to the true theta value... so in the end, to give some margin, we allow for as
        # many learning steps as twice the value of true theta parameter.
        # simul.dict_learning_params['t_learn'] = int(theta_true*2)

        for k, theta_start in enumerate(theta_start_values):
            K_true = simul.agent.getAcceptancePolicy().getBufferSizeForDeterministicBlockingFromTheta(theta_true)
            K = simul.agent.getAcceptancePolicy().getBufferSizeForDeterministicBlockingFromTheta(theta_start)
            for j, J_factor in enumerate(J_factor_values):
                Nmin = 10
                Tmin = 10
                NT_values = [compute_nparticles_and_nsteps_for_fv_process(rhos, K, J_factor, error_rel_phi=err1, error_rel_et=err2)
                             for err1, err2 in zip(error_rel_phi, error_rel_et)]
                for idx, (exponent, (N, T)) in enumerate(zip(NT_exponents, NT_values)):
                    # Lower bound for N and T so that we don't have too little particles!
                    N = np.max([Nmin, N])
                    T = np.max([Tmin, T])
                    # Set the parameters for this run
                    case += 1
                    simul.setCase(case)
                    t_sim = T  # This is used just for the title of plots done below (after the loop)
                    dict_params_simul = {
                        'theta_true': theta_true,
                        'theta_start': theta_start,
                        'buffer_size_activation_factor': J_factor,
                        'nparticles': N,
                        't_sim': T
                    }
                    dict_info = {'case': case,
                                 'ncases': ncases,
                                 'learning_method': learning_method.name,
                                 'exponent': exponent,
                                 'rhos': rhos,
                                 'K_true': K_true,
                                 'K': K,
                                 'error_rel_phi': error_rel_phi[idx],
                                 'error_rel_et': error_rel_et[idx],
                                 'alpha_start': alpha_start,
                                 'adjust_alpha': adjust_alpha,
                                 'min_time_to_update_alpha': min_time_to_update_alpha,
                                 'alpha_min': alpha_min
                                 }

                    # Run the simulation process
                    theta_opt_values[case - 1], df_learning = run_simulation_policy_learning(simul,
                                                                                             dict_params_simul,
                                                                                             dict_info,
                                                                                             dict_params_info=dict_params_info,
                                                                                             seed=seed,
                                                                                             verbose=verbose)
else:
    # Read the execution parameters from the benchmark file
    print("Reading benchmark data containing the parameter settings from file\n{}".format(benchmark_file))
    benchmark = pd.read_csv(benchmark_file)
    benchmark_groups = benchmark[benchmark['t_learn'] == 1]
    ncases = len(benchmark_groups)
    theta_true_values = np.nan * np.ones(ncases)
    theta_opt_values = np.nan * np.ones(ncases)  # List of optimum theta values achieved by the learning algorithm for each parameter setting
    idx = -1
    for i in range(benchmark_groups.shape[0]):
        idx += 1
        case = benchmark_groups['case'].iloc[i]
        theta_true = benchmark_groups['theta_true'].iloc[i]
        theta_true_values[idx] = theta_true
        theta_start = benchmark_groups['theta'].iloc[i]
        J_factor = benchmark_groups['J/K'].iloc[i]
        exponent = benchmark_groups['exponent'].iloc[i]
        N = benchmark_groups['N'].iloc[i]
        T = benchmark_groups['T'].iloc[i]
        seed = benchmark_groups['seed'].iloc[i]

        # Get the number of events to run the simulation for (from the benchmark file)
        benchmark_nevents = benchmark[benchmark['case'] == case]
        t_sim = list(benchmark_nevents['n_events_mc'] + benchmark_nevents['n_events_fv'])
        assert len(t_sim) == benchmark_nevents.shape[0], \
            "There are as many values for the number of simulation steps per learning step as the number" \
            " of learning steps read from the benchmark file ({})" \
                .format(len(t_sim), benchmark_nevents.shape[0])
        t_learn = benchmark_nevents['t_learn'].iloc[-1]
        simul.setNumLearningSteps(t_learn)

        K_true = simul.agent.getAcceptancePolicy().getBufferSizeForDeterministicBlockingFromTheta(theta_true)
        K = simul.agent.getAcceptancePolicy().getBufferSizeForDeterministicBlockingFromTheta(theta_start)

        simul.setCase(case)
        dict_params_simul = {
            'theta_true': theta_true,
            'theta_start': theta_start,
            'buffer_size_activation_factor': J_factor,
            'nparticles': 1,
            't_sim': t_sim
        }
        dict_info = {'case': case,
                     'ncases': ncases,
                     'learning_method': learning_method.name,
                     'exponent': exponent,
                     'rhos': rhos,
                     'K_true': K_true,
                     'K': K,
                     'error_rel_phi': 0.0,
                     'error_rel_et': 0.0,
                     'alpha_start': alpha_start,
                     'adjust_alpha': adjust_alpha,
                     'min_time_to_update_alpha': min_time_to_update_alpha,
                     'alpha_min': alpha_min
                     }

        # Run the simulation process
        theta_opt_values[idx], df_learning = run_simulation_policy_learning(simul,
                                                                            dict_params_simul,
                                                                            dict_info,
                                                                            dict_params_info=dict_params_info,
                                                                            params_read_from_benchmark_file=True,
                                                                            seed=seed,
                                                                            verbose=verbose)

print("Optimum theta found by the learning algorithm for each considered parameter setting:\n{}" \
      .format(pd.DataFrame.from_items([('theta_opt', theta_opt_values)])))

# Closes the object (e.g. any log and result files are closed)
simul.close()

if len(theta_true_values) == 1:
    if PLOT_GRADIENT:
        # Save the estimation of G(t) for the last learning step to a file
        # file_results_G = "G.csv"
        # pd.DataFrame({'G': simul.G}).to_csv(file_results_G)

        # -- Plot theta and the gradient of the value function
        SET_YLIM = False

        # Estimated value function
        ax, line_est = plotting.plot_colormap(df_learning['theta'], -df_learning['V'], cmap_name="Blues")

        # True value function
        # Block size for each theta, defined by the fact that K-1 is between theta and theta+1 => K = ceiling(theta+1)
        Ks = [np.int(np.ceil(np.squeeze(t) + 1)) for t in df_learning['theta']]
        # Blocking probability = Pr(K)
        p_stationary = [stationary_distribution_birth_death_process(simul.getEnv().getNumServers(), K, rhos)[1] for K in
                        Ks]
        pblock_K = np.array([p[-1] for p in p_stationary])
        pblock_Km1 = np.array([p[-2] for p in p_stationary])
        # Blocking probability adjusted for different jump rates between K-1 and K (affected by the non-deterministic probability of blocking at K-1)
        pblock_K_adj = np.squeeze(
            [pK * (1 - (K-1-theta)) for K, theta, pK in zip(Ks, df_learning['theta'], pblock_K)])
        pblock_Km1_adj = pblock_Km1  # np.squeeze([pKm1 + pK - pK_adj for pKm1, pK, pK_adj in zip(pblock_Km1, pblock_K, pblock_K_adj)])
        # assert np.allclose(pblock_K + pblock_Km1, pblock_K_adj + pblock_Km1_adj)
        # True value function: expected cost at K which is the buffer size where blocking most likely occurs...
        # (in fact, if theta is say 3.1, the probability of blocking at 4 (= K-1) is small and most blocking
        # will occur at K; if theta is 3.9, the probability of blocking at 4 (= K-1)
        # i.e. we compute at K-1 and NOT at K because we want to compare the true value function
        # with the *estimated* value function when the policy starts blocking at buffer size = theta
        # Vtrue = np.array([rewardOnJobRejection_ExponentialCost(env_queue, (K, None), Actions.REJECT, (K, None)) * pK for K, pK in zip(Ks, pblock_K)])

        # ACTUAL true value function, which takes into account the probability of blocking at K-1 as well, where the policy is non-deterministic (for non-integer theta)
        # The problem with this approach is that the stationary distribution of the chain is NOT the same as with chain
        # where rejection ONLY occurs at s=K... in fact, the transition probabilities to s=K and to s=K-1 when the
        # initial state is s=K-1 are affected by the non-deterministic probability of blocking when s=K-1...
        # Qualitatively, the stationary probability of K would be reduced and the stationary probability of K-1 would be
        # increased by the same amount.
        Vtrue = np.array(
            [rewardOnJobRejection_ExponentialCost(env_queue, (K, None), Actions.REJECT, (K, None)) * pK +
             rewardOnJobRejection_ExponentialCost(env_queue, (K-1, None), Actions.REJECT, (K-1, None)) * (K-1-theta) * pKm1
             for K, theta, pK, pKm1 in zip(Ks, df_learning['theta'], pblock_K_adj, pblock_Km1_adj)])

        # True grad(V)
        # Ref: my hand-written notes in Letter-size block of paper with my notes on the general environment - agent setup
        gradVtrue = [
            -rewardOnJobRejection_ExponentialCost(env_queue, (K-1, None), Actions.REJECT, (K-1, None)) * pKm1 for
            K, pKm1 in zip(Ks, pblock_Km1)]

        ord = np.argsort(Ks)
        # NOTE that we plot the true value function at K-1 (not at K) because K-1 is the value that is closest to theta
        # and we are plotting the *estimated* value function vs. theta (NOT vs. K).
        # line_true, = ax.plot([Ks[o]-1 for o in ord], [-Vtrue[o] for o in ord], 'g.-', linewidth=5, markersize=20)
        line_true, = ax.plot(df_learning['theta'], -Vtrue, 'gx-')  # Use when computing the ACTUAL true Value function V, which also depends on theta!
        ax.set_xlim((0, ax.get_xlim()[1]))
        ax.set_yscale('log')
        # ax.set_ylim((0, 10))
        ax.set_xlabel('theta (for estimated functions) / K-1 for true value function')
        ax.set_ylabel('Value function V (cost)')
        ax.legend([line_est, line_true], ['Estimated V', 'True V'], loc='upper left')
        ax2 = ax.twinx()
        ax2, line_grad = plotting.plot_colormap(df_learning['theta'], -df_learning['gradV'], cmap_name="Reds", ax=ax2)
        line_gradtrue, = ax2.plot([Ks[o] - 1 for o in ord], [-gradVtrue[o] for o in ord], 'k.-', linewidth=3, markersize=12)
        ax2.axhline(0, color="lightgray")
        ax2.set_ylabel('grad(V)')
        if SET_YLIM:
            ax2.set_ylim((-5, 5))  # Note: grad(V) is expected to be -1 or +1...
        ax2.legend([line_grad, line_gradtrue], ['grad(V)', 'True grad(V)'], loc='upper right')
        if is_scalar(t_sim):
            title = "Value function and its gradient as a function of theta and K. " + \
                    "Optimum K = {}, Theta start = {}, t_sim = {:.0f}".format(np.ceil(theta_true+1), theta_start, t_sim)
        else:
            title = "Value function and its gradient as a function of theta and K. " + \
                    "Optimum K = {}, Theta start = {}".format(np.ceil(theta_true+1), theta_start)
        plt.title(title)

        # grad(V) vs. V
        plt.figure()
        plt.plot(-df_learning['V'], -df_learning['gradV'], 'k.')
        ax = plt.gca()
        ax.axhline(0, color="lightgray")
        ax.axvline(0, color="lightgray")
        ax.set_xscale('log')
        # ax.set_xlim((-1, 1))
        if SET_YLIM:
            ax.set_ylim((-1, 1))
        ax.set_xlabel('Value function V (cost)')
        ax.set_ylabel('grad(V)')

    # Plot evolution of theta
    if is_scalar(t_sim):
        title = "Method: {}, Optimum Theta = {}, Theta start = {}, t_sim = {:.0f}, t_learn = {:.0f}, fixed_window={}, clipping={}" \
                    .format(learning_method.name, theta_true, theta_start, t_sim, t_learn, fixed_window, clipping) + \
                (clipping and ", clipping_value={}".format(clipping_value) or "")
    else:
        title = "Method: {}, Optimum Theta = {}, Theta start = {}, t_learn = {:.0f}, fixed_window={}, clipping={}" \
                    .format(learning_method.name, theta_true, theta_start, t_learn, fixed_window, clipping) + \
                (clipping and ", clipping_value={}".format(clipping_value) or "")
    if PLOT_TRAJECTORY and plot_trajectories:
        "In the case of the MC learner, plot the theta-learning trajectory as well as rewards received down the line"
        assert N == 1, "The simulated system has only one particle (N={})".format(N)
        # NOTE: (2021/11/27) I verified that the values of learnerP.getRewards() for the last learning step
        # are the same to those for learnerV.getRewards() (which only stores the values for the LAST learning step)
        plt.figure()
        # Times at which the trajectory of states is recorded
        times = simul.getLearnerP().getTimes()
        times_unique = np.unique(times)
        assert len(times_unique) == len(simul.getLearnerP().getPolicy().getThetas()) - 1, \
                "The number of unique learning times ({}) is equal to the number of theta updates ({})".format(
                len(times_unique), len(simul.getLearnerP().getPolicy().getThetas()) - 1)
        ## Note that we subtract 1 to the number of learning thetas because the first theta stored in the policy object is the initial theta before any update
        plt.plot(np.r_[0.0, times_unique], simul.getLearnerP().getPolicy().getThetas(), 'b.-')
        ## We add 0.0 as the first time to plot because the first theta value stored in the policy is the initial theta with which the simulation started
        ax = plt.gca()
        # Add vertical lines signalling the BEGINNING of each queue simulation
        times_sim_starts = range(0, t_learn * (t_sim + 1), t_sim + 1)
        for t in times_sim_starts:
            ax.axvline(t, color='lightgray', linestyle='dashed')

        # Buffer sizes
        buffer_sizes = [env_queue.getBufferSizeFromState(s) for s in simul.getLearnerP().getStates()]
        ax.plot(times, buffer_sizes, 'g.', markersize=3)
        # Mark the start of each queue simulation
        # DM-2021/11/28: No longer feasible (or easy to do) because the states are recorded twice for the same time step (namely at the first DEATH after a BIRTH event)
        ax.plot(times_sim_starts, [buffer_sizes[t] for t in times_sim_starts], 'gx')
        ax.set_xlabel("time step")
        ax.set_ylabel("theta")
        ax.yaxis.set_major_locator(MaxNLocator(integer=True))

        # Secondary plot showing the rewards received
        ax2 = ax.twinx()
        ax2.plot(times, -np.array(simul.getLearnerP().getRewards()), 'r-', alpha=0.3)
        # Highlight with points the non-zero rewards
        ax2.plot(times, [-r if r != 0.0 else None for r in simul.getLearnerP().getRewards()], 'r.')
        ax2.set_ylabel("Reward")
        ax2.set_yscale('log')
        plt.title(title)
    elif PLOT_TRAJECTORY:
        plt.figure()
        plt.plot(simul.getLearnerP().getPolicy().getThetas(), symbol)
        plt.title(title)
        ax = plt.gca()
        ax.set_xlabel('Learning step')
        ax.set_ylabel('theta')
        ylim = ax.get_ylim()
        ax.set_ylim((0, np.max([ylim[1], K_true])))
        ax.axhline(theta_true, color='black', linestyle='dashed')  # This is the last true theta value considered for the simulations
        ax.yaxis.set_major_locator(MaxNLocator(integer=True))
        ax.set_aspect(1 / ax.get_data_ratio())

end_time_all = timer()
elapsed_time_all = end_time_all - start_time_all
print("\n+++ OVERALL execution time: {:.1f} min, {:.1f} hours".format(elapsed_time_all / 60, elapsed_time_all / 3600))

tracemalloc.stop()

if PLOT_RESULTS_TOGETHER:
    """
    The plots of the FVRL and the respective MC execution are plotted on the same graph
    HOWEVER, THIS IS EXPECTED TO BE RUN MANUALLY AND BY PIECES, AS THE INPUT FILES CONTAINING THE RESULTS TO PLOT
    NEED TO BE CHANGED EVERY TIME WE WANT TO PLOT THE RESULTS.
    """
    # Read the results from files and plot the MC and FV results on the same graph
    resultsdir = "E:/Daniel/Projects/PhD-RL-Toulouse/projects/RL-002-QueueBlocking/results/RL/single-server"

    theta_true = 19
    # IGA results starting at a larger theta value: theta_start = 39, N = 800, t_sim = 800
    N = 800
    t_sim = N
    results_file_fv = os.path.join(os.path.abspath(resultsdir), "SimulatorQueue_20211230_001050.csv")
    results_file_mc = os.path.join(os.path.abspath(resultsdir), "SimulatorQueue_20220101_145647.csv")

    # IGA results starting at a small theta value: theta_start = 1, N = 400, t_sim = 400
    N = 400
    t_sim = N
    results_file_fv = os.path.join(os.path.abspath(resultsdir), "SimulatorQueue_20220102_093954.csv")
    results_file_mc = os.path.join(os.path.abspath(resultsdir), "SimulatorQueue_20220102_173144.csv")

    results_fv = pd.read_csv(results_file_fv)
    results_mc = pd.read_csv(results_file_mc)

    t_learn = results_fv.shape[0]
    n_events_mean = np.mean(results_fv['nevents_mc'] + results_fv['nevents_proba'])
    assert n_events_mean == np.mean(results_mc['nevents_mc'])

    plt.figure()
    plt.plot(results_fv['theta'], 'g.-')
    plt.plot(results_mc['theta'], 'r.-')
    ax = plt.gca()
    ax.set_xlabel('Learning step')
    ax.set_ylabel('theta')
    ax.set_ylim((0, 40))
    ax.axhline(theta_true, color='black', linestyle='dashed')
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    ax.set_aspect(1 / ax.get_data_ratio())
    ax.legend(["Fleming-Viot", "Monte-Carlo", "Optimum theta"])
    plt.title("# particles N = {}, Simulation time for P(T>t) and E(T_A) = {}, # learning steps = {}, Average number of events per learning step = {:.0f}" \
                .format(N, t_sim, t_learn, n_events_mean))

if PLOT_RESULTS_PAPER:
    """
    The plots to show in the paper are generated.
    HOWEVER, THIS IS EXPECTED TO BE RUN MANUALLY AND BY PIECES, AS THE INPUT FILES CONTAINING THE RESULTS TO PLOT
    NEED TO BE CHANGED EVERY TIME WE WANT TO PLOT THE RESULTS.
    """
    # Read the results from files and plot the MC and FV results on the same graph
    resultsdir = "E:/Daniel/Projects/PhD-RL-Toulouse/projects/RL-002-QueueBlocking/results/RL/single-server"

    # -- Alpha adaptive
    # results_file_fv = os.path.join(os.path.abspath(resultsdir), "SimulatorQueue_20220121_031815_FV-J=0.5K-K=20.csv")
    # results_file_mc = os.path.join(os.path.abspath(resultsdir), "SimulatorQueue_20220121_150307_MC-J=0.5K-K=20.csv")
    results_file_fv = os.path.join(os.path.abspath(resultsdir), "SimulatorQueue_20220125_025611_FV-K=20-J=0.5K.csv")
    results_file_mc = os.path.join(os.path.abspath(resultsdir), "SimulatorQueue_20220125_131828_MC-K=20-J=0.5K.csv")
    K_true = 19  # Optimum blocking size, the integer-valued K at which the expected cost is minimum

    results_file_fv = os.path.join(os.path.abspath(resultsdir), "SimulatorQueue_20220124_040745_FV-K=30-J=0.5K.csv")
    results_file_mc = os.path.join(os.path.abspath(resultsdir), "SimulatorQueue_20220125_161121_MC-K=30-J=0.5K.csv")
    K_true = 19  # Optimum blocking size, the integer-valued K at which the expected cost is minimum

    # All exponents, K0 = 10
    # 2022/02/01 --> But this is wrong because of the error about the true theta, which was not updated to the one we set!
    results_file_fv = os.path.join(os.path.abspath(resultsdir),
                                   "SimulatorQueue_20220125_025523_FV-K0=40-K=10-AlphaAdaptive.csv")
    results_file_mc = os.path.join(os.path.abspath(resultsdir), ".csv")
    K_true = 9  # Optimum blocking size, the integer-valued K at which the expected cost is minimum

    # -- Alpha constant
    results_file_fv = os.path.join(os.path.abspath(resultsdir),
                                   "SimulatorQueue_20220125_121657_FV-K0=20-K=30-J=0.5-AlphaConst.csv")
    results_file_mc = os.path.join(os.path.abspath(resultsdir),
                                   "SimulatorQueue_20220125_123300_MC-K0=20-K=30-J=0.5-AlphaConst.csv")
    K_true = 19  # Optimum blocking size, the integer-valued K at which the expected cost is minimum

    # results_file_fv = os.path.join(os.path.abspath(resultsdir), "SimulatorQueue_20220125_122700_FV-K0=10-K=30-J=0.5-AlphaConst.csv")
    # results_file_mc = os.path.join(os.path.abspath(resultsdir), "SimulatorQueue_20220125_124237_MC-K0=10-K=30-J=0.5-AlphaConst.csv")

    results_file_fv = os.path.join(os.path.abspath(resultsdir),
                                   "SimulatorQueue_20220125_180636_FV-K0=30-K=5-J=0.5-AlphaConst.csv")
    results_file_mc = os.path.join(os.path.abspath(resultsdir),
                                   "SimulatorQueue_20220125_181511_MC-K0=30-K=5-J=0.5-AlphaConst.csv")
    K_true = 24  # Optimum blocking size, the integer-valued K at which the expected cost is minimum

    # J/K = 0.5, K0 = 10
    results_file_fv1 = os.path.join(os.path.abspath(resultsdir),
                                    "SimulatorQueue_20220130_105312_FV-K0=30-K=10-J=0.5-E1.5-AlphaConst(B=5).csv")
    results_file_mc1 = os.path.join(os.path.abspath(resultsdir),
                                    "SimulatorQueue_20220130_112523_MC-K0=30-K=10-J=0.5-E1.5-AlphaConst(B=5).csv")
    K_true = 9  # Optimum blocking size, the integer-valued K at which the expected cost is minimum

    # J/K = 0.5, K0 = 20
    results_file_fv1 = os.path.join(os.path.abspath(resultsdir),
                                    "SimulatorQueue_20220125_190830_FV-K0=20-K=30-J=0.5-E1.5-AlphaConst(B=5).csv")
    results_file_mc1 = os.path.join(os.path.abspath(resultsdir),
                                    "SimulatorQueue_20220125_192242_MC-K0=20-K=30-J=0.5-E1.5-AlphaConst(B=5).csv")
    results_file_fv2 = os.path.join(os.path.abspath(resultsdir),
                                    "SimulatorQueue_20220125_193204_FV-K0=20-K=30-J=0.5-E1.0-AlphaConst(B=5).csv")
    results_file_mc2 = os.path.join(os.path.abspath(resultsdir),
                                    "SimulatorQueue_20220125_200859_MC-K0=20-K=30-J=0.5-E1.0-AlphaConst(B=5).csv")
    results_file_fv3 = os.path.join(os.path.abspath(resultsdir), ".csv")
    results_file_mc3 = os.path.join(os.path.abspath(resultsdir), ".csv")
    K_true = 19  # Optimum blocking size, the integer-valued K at which the expected cost is minimum.
    # NOTE: K_true is NOT exactly theta_true + 1 because theta_true defines xref (if I recall correctly)
    # and the minimum of the expected cost function of K is not always xref + 1, although it is close to it.

    # J/K = 0.5, K0 = 25
    # These only simulates for 300 learning steps
    # results_file_fv1 = os.path.join(os.path.abspath(resultsdir), "SimulatorQueue_20220125_233513_FV-K0=25-K=35-J=0.5-E1.5-AlphaConst(B=5).csv")
    # results_file_mc1 = os.path.join(os.path.abspath(resultsdir), "SimulatorQueue_20220126_004040_MC-K0=25-K=35-J=0.5-E1.5-AlphaConst(B=5).csv")
    # results_file_fv2 = os.path.join(os.path.abspath(resultsdir), "SimulatorQueue_20220125_235038_FV-K0=25-K=35-J=0.5-E1.0-AlphaConst(B=5).csv")
    # results_file_mc2 = os.path.join(os.path.abspath(resultsdir), "SimulatorQueue_20220126_032802_MC-K0=25-K=35-J=0.5-E1.0-AlphaConst(B=5).csv")

    results_file_fv1 = os.path.join(os.path.abspath(resultsdir),
                                    "SimulatorQueue_20220126_033710_FV-K0=25-K=35-J=0.5-E1.5-AlphaConst(B=5).csv")
    results_file_mc1 = os.path.join(os.path.abspath(resultsdir),
                                    "SimulatorQueue_20220126_133352_MC-K0=25-K=35-K=0.5-E1.5-AlphaConst(B=5).csv")
    results_file_fv2 = os.path.join(os.path.abspath(resultsdir),
                                    "SimulatorQueue_20220126_033755_FV-K0=25-K=35-J=0.5-E1.0-AlphaConst(B=5).csv")
    results_file_mc2 = os.path.join(os.path.abspath(resultsdir),
                                    "SimulatorQueue_20220126_133409_MC-K0=25-K=35-K=0.5-E1.0-AlphaConst(B=5).csv")
    results_file_fv3 = os.path.join(os.path.abspath(resultsdir),
                                    "SimulatorQueue_20220125_235230_FV-K0=25-K=35-J=0.5-E0.5-AlphaConst(B=5).csv")
    results_file_mc3 = os.path.join(os.path.abspath(resultsdir),
                                    "SimulatorQueue_20220127_022406_MC-K0=25-K=35-K=0.5-E0.5-AlphaConst(B=5).csv")
    K_true = 24  # Optimum blocking size, the integer-valued K at which the expected cost is minimum

    # J/K = 0.3, K0 = 20
    # These only simulates for 300 learning steps
    # results_file_fv1 = os.path.join(os.path.abspath(resultsdir), "SimulatorQueue_20220125_235638_FV-K0=20-K=30-J=0.3-E1.5-AlphaConst(B=5).csv")
    # results_file_mc1 = os.path.join(os.path.abspath(resultsdir), "SimulatorQueue_20220126_004118_MC-K0=20-K=30-J=0.3-E1.5-AlphaConst(B=5).csv")
    # results_file_fv2 = os.path.join(os.path.abspath(resultsdir), "SimulatorQueue_20220125_235608_FV-K0=20-K=30-J=0.3-E1.0-AlphaConst(B=5).csv")
    # results_file_mc2 = os.path.join(os.path.abspath(resultsdir), "SimulatorQueue_20220126_023359_MC-K0=20-K=30-J=0.3-E1.0-AlphaConst(B=5).csv")

    results_file_fv1 = os.path.join(os.path.abspath(resultsdir),
                                    "SimulatorQueue_20220126_135652_FV-K0=20-K=30-J=0.3-E1.5-AlphaConst(B=5).csv")
    results_file_mc1 = os.path.join(os.path.abspath(resultsdir),
                                    "SimulatorQueue_20220126_193306_MC-K0=20-K=30-J=0.3-E1.5-AlphaConst(B=5).csv")
    results_file_fv2 = os.path.join(os.path.abspath(resultsdir),
                                    "SimulatorQueue_20220126_135719_FV-K0=20-K=30-J=0.3-E1.0-AlphaConst(B=5).csv")
    results_file_fv2 = os.path.join(os.path.abspath(resultsdir),
                                    "SimulatorQueue_20220519_165242_FV-K0=20-K=30-J=0.3-E0.5-AlphaConst(B=5)_seed1313.csv")
    results_file_mc2 = os.path.join(os.path.abspath(resultsdir),
                                    "SimulatorQueue_20220126_193252_MC-K0=20-K=30-J=0.3-E1.0-AlphaConst(B=5).csv")
    results_file_mc2 = os.path.join(os.path.abspath(resultsdir),
                                    "SimulatorQueue_20220519_203152_MC-K0=20-K=30-J=0.3-E0.5-AlphaConst(B=5)_seed1313.csv")
    K_true = 19  # Optimum blocking size, the integer-valued K at which the expected cost is minimum

    theta_update_strategy = "normal"

    # -- Alpha constant + clipping
    # results_file_fv = os.path.join(os.path.abspath(resultsdir), "SimulatorQueue_20220125_125902_FV-K0=20-K=30-J=0.5-AlphaConst-Clipping.csv")
    # results_file_mc = os.path.join(os.path.abspath(resultsdir), "SimulatorQueue_20220125_132227_MC-K0=20-K=30-J=0.5-AlphaConst-Clipping.csv")

    # J/K = 0.5, K0 = 20
    results_file_fv1 = os.path.join(os.path.abspath(resultsdir),
                                    "SimulatorQueue_20220125_212158_FV-K0=20-K=30-J=0.5-E1.5-Clipping.csv")
    results_file_mc1 = os.path.join(os.path.abspath(resultsdir),
                                    "SimulatorQueue_20220125_214819_MC-K0=20-K=30-J=0.5-E1.5-Clipping.csv")
    results_file_fv2 = os.path.join(os.path.abspath(resultsdir),
                                    "SimulatorQueue_20220125_212353_FV-K0=20-K=30-J=0.5-E1.0-Clipping.csv")
    results_file_mc2 = os.path.join(os.path.abspath(resultsdir),
                                    "SimulatorQueue_20220125_220710_MC-K0=20-K=30-J=0.5-E1.0-Clipping.csv")
    K_true = 19  # Optimum blocking size, the integer-valued K at which the expected cost is minimum

    # J/K = 0.3, K0 = 20
    results_file_fv1 = os.path.join(os.path.abspath(resultsdir),
                                    "SimulatorQueue_20220126_035241_FV-K0=20-K=30-J=0.3-E1.5-Clipping.csv")
    results_file_mc1 = os.path.join(os.path.abspath(resultsdir),
                                    "SimulatorQueue_20220126_131153_MC-K0=20-K=30-J=0.3-E1.5-Clipping.csv")
    results_file_fv2 = os.path.join(os.path.abspath(resultsdir),
                                    "SimulatorQueue_20220126_035406_FV-K0=20-K=30-J=0.3-E1.0-Clipping.csv")
    results_file_mc2 = os.path.join(os.path.abspath(resultsdir),
                                    "SimulatorQueue_20220126_131215_MC-K0=20-K=30-J=0.3-E1.0-Clipping.csv")
    K_true = 19  # Optimum blocking size, the integer-valued K at which the expected cost is minimum

    # J/K = 0.5, K0 = 25
    results_file_fv1 = os.path.join(os.path.abspath(resultsdir),
                                    "SimulatorQueue_20220126_034444_FV-K0=25-K=35-J=0.5-E1.5-Clipping.csv")
    results_file_mc1 = os.path.join(os.path.abspath(resultsdir),
                                    "SimulatorQueue_20220126_131125_MC-K0=25-K=35-J=0.5-E1.5-Clipping.csv")
    results_file_fv2 = os.path.join(os.path.abspath(resultsdir),
                                    "SimulatorQueue_20220126_034327_FV-K0=25-K=35-J=0.5-E1.0-Clipping.csv")
    results_file_mc2 = os.path.join(os.path.abspath(resultsdir),
                                    "SimulatorQueue_20220126_131057_MC-K0=25-K=35-J=0.5-E1.0-Clipping.csv")
    K_true = 24  # Optimum blocking size, the integer-valued K at which the expected cost is minimum

    theta_update_strategy = "clipping"

    # Read the data
    results_fv = pd.read_csv(results_file_fv)

    results_fv1 = pd.read_csv(results_file_fv1);
    results_fv1['case'] = 1
    results_mc1 = pd.read_csv(results_file_mc1);
    results_mc1['case'] = 1
    results_fv2 = pd.read_csv(results_file_fv2);
    results_fv2['case'] = 2
    results_mc2 = pd.read_csv(results_file_mc2);
    results_mc2['case'] = 2
    results_fv3 = pd.read_csv(results_file_fv3);
    results_fv3['case'] = 3
    results_mc3 = pd.read_csv(results_file_mc3);
    results_mc3['case'] = 3

    results_fv = pd.concat([results_fv1, results_fv2, results_fv3])
    results_mc = pd.concat([results_mc1, results_mc2, results_mc3])

    results_fv = results_fv1
    results_mc = results_mc1
    error = 1.5
    results_fv = results_fv2
    results_mc = results_mc2
    error = 1.0

    # Whether to set specific tick marks for the Y-axis in order to align visually two contiguous plots in the paper
    set_special_axis_ticks = True
    # set_special_axis_ticks = False

    t_learn_max = 200
    if t_learn_max is not None:
        results_fv = results_fv[results_fv['t_learn'] <= 200]
        results_mc = results_mc[results_mc['t_learn'] <= 200]


    # Plotting process starts
    all_cases = aggregation_bygroups(results_fv, ['case', 'theta_true', 'J/K', 'exponent', 'N', 'T'],
                                     ['K', 'n_events_mc', 'n_events_fv'],
                                     stats=['count', 'mean', 'std', 'min', 'max'])
    n_events_by_case = aggregation_bygroups(results_mc, ['case'], ['n_events_mc'])

    J_factor_values = all_cases.index.get_level_values('J/K')
    case_values = all_cases.index.get_level_values('case')
    theta_true_values = all_cases.index.get_level_values('theta_true')
    exponent_values = all_cases.index.get_level_values('exponent')
    N_values = all_cases.index.get_level_values('N')
    T_values = all_cases.index.get_level_values('T')
    # Cases are sorted from larger error to smaller error
    case_descs = ['Larger error', 'Mid error', 'Smaller error']
    colors = ['red', 'blue', 'green']
    colors = ['black', 'black', 'black']
    linestyles = ['dotted', 'dashed', 'solid']
    # linestyles = ['dashed', 'solid', 'solid']
    linestyles = ['solid', 'dashed', 'solid']
    linewidth = 2
    fontsize = 22
    n_subplots = 1
    # Shift of the optimum theta when we define the cost as an increasing function of the blocking size
    rho = 0.7
    b = 3.0
    shift_optimum = np.log(-np.log(rho) / (np.log(b) + np.log(rho))) / np.log(
        b)  # ~ -0.66667 when rho = 0.7 and b = 3.0
    for J_factor in np.unique(J_factor_values):
        cases = case_values[J_factor_values == J_factor]
        ncases = len(cases)

        # IMPORTANT: We assume that the true theta value and the start theta values are ALL the same for all cases
        theta_true = theta_true_values[0] + shift_optimum
        # K_true = int(np.ceil(theta_true + 1))
        theta_start = results_fv['theta'].iloc[0]
        K_start = int(np.ceil(theta_start + 1))

        axes = plt.figure(figsize=(18, 18)).subplots(1, n_subplots, squeeze=False)
        legend = [[], []]
        figfile = os.path.join(os.path.abspath(resultsdir),
                               "RL-single-FVMC-K0={}-start={}-J={}-E{:.1f}-{}.jpg" \
                               .format(K_true, K_start, J_factor, error, theta_update_strategy))
        for idx_case, case in enumerate(cases):
            print("Plotting case {} with idx_case = {}, K_true={}, K_start={}, J={}K".format(case, idx_case, K_true,
                                                                                             K_start, J_factor))
            # The metadata for the title and legend
            case_desc = case_descs[idx_case]
            N = N_values[idx_case]
            T = T_values[idx_case]
            n_events_et = all_cases['n_events_mc']['mean'].iloc[idx_case]
            n_events_fv = all_cases['n_events_fv']['mean'].iloc[idx_case]
            n_events_mean = n_events_by_case.iloc[idx_case]['n_events_mc']['mean']

            # The data to plot
            ind = results_fv['case'] == case
            K_start = int(np.ceil(theta_start + 1))
            K_opt_fv = int(np.round(results_fv['theta'][ind].iloc[-1])) + 1  # Optimum K found by the algorithm = closest integer to last theta + 1
            K_opt_mc = int(np.round(results_mc['theta'][ind].iloc[-1])) + 1  # Optimum K found by the algorithm = closest integer to last theta + 1
            x = results_fv['t_learn'][ind]
            y_fv = results_fv['theta'][ind]
            y_mc = results_mc['theta'][ind]
            axes[0][0].plot(x, y_fv, color='green', linestyle=linestyles[idx_case], linewidth=linewidth)
            axes[0][n_subplots - 1].plot(x, y_mc, color='red', linestyle='dashed', linewidth=linewidth)

            # Errors
            err_phi = results_fv['err_phi'][ind].iloc[0]
            err_et = results_fv['err_et'][ind].iloc[0]

            # legend[0] += ["{}) {}: FVRL (N={}, T={}, error(Phi)={:.0f}%, error(ET)={:.0f}%)" #, avg #events per learning step={})" \
            #                .format(idx_case+1, case_desc, N, T, err_phi*100, err_et*100)] #, n_events_et + n_events_fv)]
            # legend[n_subplots-1] += ["{}) {}: MC (comparable to FVRL case ({})" #, avg #events per learning step={})" \
            #                .format(idx_case+1, case_desc, idx_case+1)] #, n_events_et + n_events_fv)]
            # legend[0] += ["FVRL (N={}, T={}, expected error = {:.0f}%)\n(avg #events per learning step={:.0f})" \
            #                  .format(N, T, error*100, n_events_mean)]
            # legend[n_subplots-1] += ["MC (avg #events per learning step={:.0f})".format(n_events_mean)]
            legend[0] += ["N={}, T={}: expected error = {:.0f}%)\n(avg #events per learning step={:.0f})" \
                              .format(N, T, error * 100, n_events_mean)]

        for idx in range(n_subplots):
            axes[0][idx].set_xlabel('Learning step', fontsize=fontsize)
            axes[0][idx].set_ylabel('theta', fontsize=fontsize)
            for tick in axes[0][idx].xaxis.get_major_ticks():
                tick.label.set_fontsize(fontsize)
            for tick in axes[0][idx].yaxis.get_major_ticks():
                tick.label.set_fontsize(fontsize)
            axes[0][idx].set_ylim((0, np.max([axes[0][idx].get_ylim()[1], K_start, K_true])))
            axes[0][idx].axhline(theta_true, color='gray', linestyle='dashed')
            axes[0][idx].yaxis.set_major_locator(MaxNLocator(integer=True))
            axes[0][idx].set_aspect(1 / axes[0][idx].get_data_ratio())
            if set_special_axis_ticks:
                axes[0][idx].set_yticks(range(0, 36, 5))
            # axes[0][idx].legend(legend[idx] + ["Optimum theta"], fontsize='xx-large', loc='lower right')
            axes[0][idx].text(axes[0][idx].get_xlim()[1] / 1.7, 5,
                              "N={}, T={}\nAvg #events per step = {:.0f}\n\nK* = {}\nEstimated K* (FV) = {}\nEstimated K* (MC) = {}" \
                              .format(N, T, n_events_mean, K_true, K_opt_fv, K_opt_mc),
                              horizontalalignment='center',
                              fontsize=fontsize, bbox={'facecolor': 'none', 'edgecolor': 'black'})
            # plt.title("# particles N = {}, Simulation time for P(T>t) and E(T_A) = {}, # learning steps = {}, Average number of events per learning step = {:.0f}" \
            #          .format(N, 0, 0, 0))

        plt.gcf().subplots_adjust(left=0.15)
        ## To avoid cut off of vertical axis label!!
        ## Ref: https://stackoverflow.com/questions/6774086/why-is-my-xlabel-cut-off-in-my-matplotlib-plot
        plt.savefig(figfile)
