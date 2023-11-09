# -*- coding: utf-8 -*-
"""
Created on Wed Feb  3 21:00:15 2021

@author: Daniel Mastropietro
"""

import runpy
runpy.run_path('../../setup.py')


########
# 2023/10/12: Learn an actor-critic policy using neural networks (with the torch package)
# Learning happens with the ActorCriticNN learner which defines a loss of type `tensor` which can be minimized using the backward() method of torch Tensors
# IT WORKS!
import numpy as np
from  matplotlib import pyplot as plt, cm
from matplotlib.ticker import MaxNLocator
from Python.test.test_optimizers_discretetime import InputLayer, Test_EstPolicy_EnvGridworldsWithObstacles
from Python.lib.agents.learners.policies import LeaActorCriticNN

seed = 1317
test_ac = Test_EstPolicy_EnvGridworldsWithObstacles()
test_ac.setUpClass(shape=(3, 4), nn_input=InputLayer.ONEHOT, nn_hidden_layer_sizes=[8], seed=seed, debug=False)
test_ac.setUp()
print(test_ac.policy_nn.nn_model)

method = "online"
method = "td"
method = "fv"

n_learning_steps = 50
if method == "online":
    # Actor-Critic policy learner with TD(0) as value functions learner
    learner_ac = LeaActorCriticNN(test_ac.env2d, test_ac.agent_nn_td.getPolicy(), test_ac.agent_nn_td.getLearner(), optimizer_learning_rate=0.1, seed=test_ac.seed, debug=True)
    max_time_steps = test_ac.env2d.getNumStates()*100
    n_episodes_per_learning_step = 30
    V_all = np.zeros((n_learning_steps, test_ac.env2d.getNumStates()))
    Q_all = np.zeros((n_learning_steps, test_ac.env2d.getNumStates(), test_ac.env2d.getNumActions()))
    R_all = np.zeros((n_learning_steps,))
    loss_all = np.nan * np.ones((n_learning_steps,))
    for t_learn in range(1, n_learning_steps+1):
        print(f"\n\n*** Running learning step {t_learn}...")
        loss_all[t_learn-1] = learner_ac.learn(n_episodes_per_learning_step, max_time_steps=max_time_steps, prob_include_in_train=1.0)

        V_all[t_learn-1, :] = learner_ac.learner_value_functions.getV().getValues()
        Q_all[t_learn-1, :, :] = learner_ac.learner_value_functions.getQ().getValues().reshape(test_ac.env2d.getNumStates(), test_ac.env2d.getNumActions())
        R_all[t_learn-1] = learner_ac.learner_value_functions.getAverageReward()
        # Could also retrieve the average reward from the Actor-Critic learner (if store_trajectory_history=False in the constructor of the value functions learner)
        #R_all[t_learn-1] = learner_ac.average_reward_over_episodes
else:
    # Actor-Critic policy learner with FV as value functions learner
    learner_ac = LeaActorCriticNN(test_ac.env2d, test_ac.agent_nn_fv.getPolicy(), test_ac.agent_nn_fv.getLearner(), optimizer_learning_rate=0.1, seed=test_ac.seed, debug=True)
    max_time_steps = test_ac.env2d.getNumStates()*100
    n_episodes_per_learning_step = 30
    V_all = np.zeros((n_learning_steps, test_ac.env2d.getNumStates()))
    Q_all = np.zeros((n_learning_steps, test_ac.env2d.getNumStates(), test_ac.env2d.getNumActions()))
    R_all = np.zeros((n_learning_steps,))
    loss_all = np.nan * np.ones((n_learning_steps,))
    for t_learn in range(1, n_learning_steps+1):
        print(f"\n\n*** Running learning step {t_learn}...")

        # Learn the value functions using the FV simulator
        if method == "fv":
            V, Q, state_counts, probas_stationary, expected_reward, expected_absorption_time, n_cycles_absorption_used, n_events_et, n_events_fv = \
                test_ac.sim_fv.run(seed=test_ac.seed, verbose=True, verbose_period=test_ac.agent_nn_fv.getLearner().T // 10)
            #average_reward = test_ac.agent_nn_fv.getLearner().getAverageReward()  # This average reward should not be used because it is inflated by the FV process that visits the states with rewards more often
            average_reward = expected_reward
        else:
            # Check if things work using the TD(0) learner for the value functions
            V, Q, state_counts, _, _, _ = \
                test_ac.sim_td.run(nepisodes=n_episodes_per_learning_step, max_time_steps=max_time_steps, start=test_ac.start_state, seed=test_ac.seed, verbose=True, verbose_period=test_ac.agent_nn_fv.getLearner().T // 10)
            average_reward = test_ac.agent_nn_td.getLearner().getAverageReward()

        V_all[t_learn-1, :] = V
        Q_all[t_learn-1, :, :] = Q.reshape(test_ac.env2d.getNumStates(), test_ac.env2d.getNumActions())
        R_all[t_learn-1] = average_reward
        loss_all[t_learn-1] = learner_ac.learn_from_estimated_value_functions(V, Q, state_counts)

# Make a copy of the measures that we would like to compare (on the same plot)
if method == "online":
    loss_all_online = loss_all.copy()
    R_all_online = R_all.copy()
elif method == "td":
    loss_all_td = loss_all.copy()
    R_all_td = R_all.copy()
elif method == "fv":
    loss_all_fv = loss_all.copy()
    R_all_fv = R_all.copy()

# Plot loss and average reward for the currently analyzed learner
ax_loss = plt.figure().subplots(1,1)
ax_loss.plot(range(1, n_learning_steps+1), loss_all[:n_learning_steps], marker='.', color="red")
ax_loss.set_xlabel("Learning step")
ax_loss.set_ylabel("Loss")
ax_R = ax_loss.twinx()
ax_R.plot(range(1, n_learning_steps+1), R_all[:n_learning_steps], marker='.', color="green")
ax_R.set_ylabel("Average reward")
ax_loss.set_title(f"{method.upper()}\nEvolution of the LOSS (left, red) and Average Reward (right, green) with the learning step")
ax_loss.xaxis.set_major_locator(MaxNLocator(integer=True))

# Plot the value functions for the currently analyzed learner
axes = plt.figure().subplots(test_ac.env2d.shape[0], test_ac.env2d.shape[1])
for i, ax in enumerate(axes.reshape(-1)):
    ax.plot(range(1, n_learning_steps+1), V_all[:n_learning_steps, i], marker='.', color="black")
    ax.plot(range(1, n_learning_steps+1), Q_all[:n_learning_steps, i, :], marker='.')
    ax = plt.gca()
    ax.set_xlabel("Learning step")
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
plt.legend(["V(s)"] + ["Q(s," + str(a) + ")" for a in range(Q_all.shape[2])])
plt.suptitle(f"{method.upper()}\nEvolution of the value functions V(s) and Q(s,a) with the learning step by state")

#-- ALTOGETHER
# Plot all average rewards together (to compare methods)
# Color names are listed here: https://matplotlib.org/stable/gallery/color/named_colors.html
# We normalize the average reward plots so that they converge to 1.0 (easier interpretation of the plot)
max_avg_reward = 0.2    # This value is problem dependent
ax_loss, ax_R = plt.figure().subplots(1,2)
ax_loss.plot(range(1, n_learning_steps+1), loss_all_online[:n_learning_steps], '-.', marker='.', color="darkred")
ax_loss.plot(range(1, n_learning_steps+1), loss_all_td[:n_learning_steps], marker='.', color="red")
ax_loss.plot(range(1, n_learning_steps+1), loss_all_fv[:n_learning_steps], marker='.', color="orangered")
ax_loss.set_xlabel("Learning step")
ax_loss.set_ylabel("Loss")
ax_loss.axhline(0, color="gray")
ax_loss.xaxis.set_major_locator(MaxNLocator(integer=True))
ax_loss.set_title("Evolution of the LOSS with the learning step by learning method")
ax_loss.legend(["TD online", "TD OFFLINE", "FV OFFLINE"])
ax_R.plot(range(1, n_learning_steps+1), R_all_online[:n_learning_steps] / max_avg_reward, '-.', marker='.', color="forestgreen")
ax_R.plot(range(1, n_learning_steps+1), R_all_td[:n_learning_steps] / max_avg_reward, marker='.', color="green")
ax_R.plot(range(1, n_learning_steps+1), R_all_fv[:n_learning_steps] / max_avg_reward, marker='.', color="greenyellow")
ax_R.set_xlabel("Learning step")
ax_R.set_ylabel(f"Average reward (normalized by the MAX average reward = {max_avg_reward})")
ax_R.axhline(1, color="gray")
ax_R.axhline(0, color="gray")
ax_R.set_title("Evolution of the NORMALIZED Average Reward with the learning step by learning method")
ax_R.legend(["TD online", "TD OFFLINE", "FV OFFLINE"])
plt.suptitle("ALL LEARNING METHODS")


# Additional plots
env = test_ac.env2d
gridworld_shape = env.shape
policy = learner_ac.getPolicy()
learner = learner_ac.getValueFunctionsLearner()
print("Final network parameters:")
print(list(policy.getThetaParameter()))

#-- Plots
colormap = cm.get_cmap("rainbow")  # useful colormaps are "jet", "rainbow", seismic"
colornorm = None

# Plot a few weights of the neural network
# TBD

# Policy for each action at each state
axes = plt.figure().subplots(*gridworld_shape)
probs_actions_toplot = np.nan*np.ones((3, 3))
for i in range(axes.shape[0]):
    for j in range(axes.shape[1]):
        state_1d = np.ravel_multi_index((i, j), gridworld_shape)
        print("")
        for action in range(env.getNumActions()):
            print(f"Computing policy Pr(a={action}|s={(i,j)})...", end= " ")
            idx_2d = (0, 1) if action == 0 else (1, 2) if action == 1 else (2, 1) if action == 2 else (1, 0)
            probs_actions_toplot[idx_2d] = policy.getPolicyForAction(action, state_1d)
            print("p = {:.3f}".format(probs_actions_toplot[idx_2d]))
        img = axes[i,j].imshow(probs_actions_toplot, cmap=colormap, vmin=0, vmax=1)
plt.colorbar(img, ax=axes)  # This adds a colorbar to the right of the FIGURE. However, the mapping from colors to values is taken from the last generated image! (which is ok because all images have the same range of values.
                            # Otherwise see answer by user10121139 in https://stackoverflow.com/questions/13784201/how-to-have-one-colorbar-for-all-subplots
plt.suptitle(f"{method.upper()}\nPolicy at each state")

# Distribution of state counts at last learning step run
state_counts = learner.getStateCounts()
state_counts_2d = np.array(state_counts).reshape(*gridworld_shape)
print(state_counts_2d)
print(state_counts_2d / np.sum(state_counts_2d.reshape(-1)))
ax = plt.figure().subplots(1,1)
img = ax.imshow(state_counts_2d, cmap=colormap, norm=colornorm)
plt.colorbar(img)

# Let's look at the trajectories of the learner (it works when constructing the learner with store_history_over_all_episodes=True)
print(len(learner.getStates()))
print([len(trajectory) for trajectory in learner.getStates()])

raise KeyboardInterrupt


########
# 2023/03/08: Test the package optparse to parse arguments when calling a script from the command prompt, specially
# its capabilities of parsing an argument that should be interpreted as a list.
# Goal: Run several replications of the same simulation (using different seeds)
# Ref:
# https://stackoverflow.com/questions/15753701/how-can-i-pass-a-list-as-a-command-line-argument-with-argparse
# https://docs.python.org/3.7/library/optparse.html#option-callbacks

import optparse

# ---------------------------- Auxiliary functions ---------------------------#
def convert_str_to_list_of_floats(option, opt, value, parser):
    print(f"option: {type(option)}, {dir(option)}")
    print(f"opt: {opt}")
    print(f"value: {value}")
    print(f"parser: {parser}")
    print(f"parser.values: {parser.values}")
    if isinstance(value, str):
        setattr(parser.values, option.dest, [float(s) for s in value.split(",")])

def parse_input_parameters(argv):
    # Written for uugot.it project in Apr-2021
    # Parse input parameters
    # Ref: https://docs.python.org/3.7/library/optparse.html
    # Main steps:
    # 1) The option parser is initialized with optparse.OptionParser(), where we can specify the usage= and version=,
    # as e.g. `optparse.OptionParser(usage="%prog [-v] [-p]", version="%prog 1.0")`
    # 2) New options to parse are added with parser.add_option(), where the metavar= argument (e.g. `metavar="FILE"`)
    # is used to indicate the option expects a value to be specified (e.g. `--filename="file.txt"` as opposed to `--verbose`, which expects no value).
    # We can also define:
    #    a) the default value of the option (although this is more clearly done with parser.set_defaults().
    #    b) the action to take with the option value read with the action= argument, e.g. "store_true", "store_false",
    #       which are actually needed for FLAG options that do NOT require any option value (e.g. -v for verbose, etc.),
    #       and ***whose default value (i.e. when the flag is not given) is specified by the default= parameter***.
    #       The default action is "store" which is used for options accepting a value as in `--file="file.txt".
    #       --> NOTE that the action can be "callback" meaning that a callback function with the signature callback(option, opt, value, parser)
    #       is called to parse the argument value. In this case, if the value of the argument needs to be updated
    #       (e.g. a string converted to a list) we need to:
    #       - define the name of the argument to set with the `dest=` option of the parser.add_option() method.
    #       - set the value of the argument in the callback by calling `setattr(parser.values, option.dest, <value>)`.
    #       Ref: https://docs.python.org/3.7/library/optparse.html#option-callbacks
    #    b) the type of the option value expected with the type= argument (e.g. type="int"), which defaults to "string".
    #    c) the store destination with the dest= argument defining the attribute name of the `options` object
    #       created when running parser.parse_args() (see next item) where the option value is stored.
    #       See more details about the default value of dest= below.
    # 3) Options are parsed with parser.parse_args() into a tuple (options, args), where `options` is an object
    # that contains all the name-value pair options and `args` is an object containing the positional parameters
    # that come after all other options have been passed (e.g. `-v --file="file.txt" arg1 arg2`).
    # 4) Every option read is stored as an attribute of the `options` object created by parser.parse_args()
    # whose name is the value specified by the dest= parameter of the parser.add_option() method, or its
    # (intelligent) default if none is specified (e.g. the option '--model_pos' is stored as options.model_pos by default)
    usage = "usage: %prog [options]"
    parser = optparse.OptionParser(usage="%prog [--method] [--t_learn] [--replications] [--theta_ref] [--theta_start] [-N] [-T] [--J_factor] [-d] [-v]")
    parser.add_option("--method",
                      metavar="METHOD",
                      help="Learning method [default: %default]")
    parser.add_option("--t_learn",
                      type="int",
                      metavar="# Learning steps",
                      help="Number of learning steps [default: %default]")
    parser.add_option("--replications",
                      type="int",
                      metavar="# Replications", default=1,
                      help="Number of replications to run [default: %default]")
    parser.add_option("--theta_start", dest="theta_start",
                      type="str",
                      action="callback",
                      callback=convert_str_to_list_of_floats,
                      metavar="Initial theta",
                      help="Initial theta parameter for the learning process [default: %default]")
    parser.add_option("-N",
                      type="int",
                      metavar="# particles", default=1,
                      help="Number of Fleming-Viot particles [default: %default]")
    parser.add_option("-T",
                      type="int",
                      metavar="# arrival events", default=100,
                      help="Number of arrival events to observe before ending the simulation [default: %default]")
    parser.add_option("-d", "--debug", dest="debug", default=False,
                      action="store_true",
                      help="debug mode")
    parser.add_option("-v", "--verbose", dest="verbose", default=False,
                      action="store_true",
                      help="verbose: show relevant messages in the log")

    parser.set_defaults(method="MC",
                        t_learn=10,
                        theta_start=0.1)

    (options, args) = parser.parse_args(argv)

    print("Parsed command line options: " + repr(options))

    # options: dictionary with name-value pairs
    # args: argument values (which do not require an argument name
    return options, args
# ---------------------------- Auxiliary functions ---------------------------#

# Parse input parameters
options, args = parse_input_parameters(sys.argv[1:])
print(f"options: {options}")
print(f"args: {args}")

raise KeyboardInterrupt


########
# 2021/02/03: Test a generator of value combinations.
# Goal: Generate all possible states of a multidimensional Markov chain.
import numpy as np
from math import factorial
from time import time

from Python.lib.utils.computing import all_combos_with_sum, comb

def prob(n, const, rho):
    return np.prod( [(1- r)*r**nr for r, nr in zip(rho, n)] ) / const

C = 20
R = 3
rho = [0.5]*R
#rho = [0.2, 0.875, 0.833]

time_start = time()
const = 0
ncases_total = 0
prod = [0]*(C+1)   # Array to store the contributions to the normalizing for each 1 <= c <= C
for c in range(C+1):
    print("Computing normalizing constant for R={}, c={}...".format(R, c), end=" ")
    ncases = comb(c+R-1,c)
    combos = all_combos_with_sum(R, c)
    count = 0
    while True:
        try:
            v = next(combos)
            #print(v, end=" ")
            assert len(v) == len(rho), "The length of v and rho coincide ({}, {})".format(len(v), len(rho))
            prod[c] += np.prod( [(1- r)*r**nr for r, nr in zip(rho, v)] )
            count += 1
        except StopIteration as e:
            #print("END!")
            break
    combos.close()
    const += prod[c]
    print("--> generated combinations: {}".format(count))
    #print("prod: {}".format(prod))
    #print("const: {}".format(const))
    assert count == ncases
    ncases_total += ncases
assert const <= 1, "The normalizing constant is <= 1"
assert abs(sum(prod)/const - 1.0) < 1E-6

# Blocking probability
pblock = prod[C] / const
time_end = time()

print("\nExecution time: {} sec".format(time_end - time_start))
print("Total number of cases: {}".format(ncases_total))
print("Normalizing constant for rho={}: {}".format(rho, const))
print("Blocking probability: Pr(C)={:.5f}%".format(pblock*100))
