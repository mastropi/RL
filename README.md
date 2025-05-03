<!--
About using Latex in Markodown: https://www.fabriziomusacchio.com/blog/2021-08-10-How_to_use_LaTeX_in_Markdown/

They suggest using the following piece of HTML code in the Markdown document, but it doesn't actually work...
<script
  src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS-MML_HTMLorMML"
  type="text/javascript">
</--script>
-->


# Overview
This repository contains implementation code to reproduce the results presented in the following papers (listed from most to least recent):

1. QUESTA journal ("Queueing Systems: Theory and Applications", under review): "Fast-exploring reinforcement learning with applications to stochastic networks", D. Mastropietro, U. Ayesta, M. Jonckheere, S. Majewski, 2025
   (see Appendix (A1) for details about simulation and estimation setups relevant for this paper).
2. EWRL-2024 workshop: "Sample-efficient reinforcement learning for environments with rare high-reward states", D. Mastropietro, U. Ayesta, M. Jonckheere, 17th European Workshop on Reinforcement Learning (EWRL), 2024, https://ut3-toulouseinp.hal.science/hal-04917977.
3. EWRL-2022 workshop: "Boosting reinforcement learning with sparse and rare rewards using Fleming-Viot particle systems”, D. Mastropietro, S. Majewski, U. Ayesta, M. Jonckheere, 15th European Workshop on Reinforcement Learning (EWRL), https://ewrl.wordpress.com/wp-content/uploads/2022/09/rlfleming-ewrl-2022-cameraready.pdf. 

The above papers present a reinforcement learning methodology that leverages Fleming-Viot particle systems (FV) to speed up the discovery of very rarely occurring states
that may potentially carry very large rewards affecting the objective function to optimize.
Fleming-Viot particle systems are used, not only to _discover_ rarely occurring states faster than Monte-Carlo exploration,
but also to consistently _estimate_ their probability of occurrence.

The Fleming-Viot methodology is particularly relevant when the probability of occurrence of those states is very small under all policies,
and where other techniques such as importance sampling are not feasible. For instance, in the `M/M/1` queue one-dimensional example presented in paper #1 above,
where only threshold-type admission control policies are considered, it is not possible to define an alternate policy that increases
the probability of occurrence of the blocking state `K` (the reject threshold) because that would entail _reducing_ the value of $K$ to say `K'`.
However, under such alternate policy that rejects an incoming job when the queue size is `K' < K` at the time of arrival,
all the system states that are between `K'` and `K` will never be observed, and this violates the conditions under which importance sampling is applicable.

# Relevant execution scripts
1. `run_FV.py`: Runs the Fleming-Viot methodology on an `M/M/1` queue system,
i.e. a single-server queue with exponential inter-arrival times and exponential service times.
The script can be used to evaluate the impact of deterministic threshold-type admission control policies that reject an incoming job when the queue length is `x = K` at the time of the job arrival.
In particular, it can be used to:
   1. Estimate the blocking probability at `K`, either using Fleming-Viot or Monte-Carlo.
   2. Estimate the long-run expected cost due to rejecting incoming jobs at `K`, either using Fleming-Viot or Monte-Carlo. 
   The script can be used to reproduce the violin plots presented in paper #1 (QUESTA) for the `M/M/1` queue system that compare the Fleming-Viot estimation procedure with the Monte-Carlo estimation procedure
   on different FV hyperparameters, such as the number `N` of particles, the number `T` of arrival events, and the size `J` of the absorption set `A`, that is at the core of the Fleming-Viot methodology.

   For the execution details, see the comments at the top part of the `__main__` section, which also includes an example of execution from the command line.
   (A better way of accessing the execution details is expected to come in the future, following the mechanism used for the other scripts below.)
2. `run_FV_LossNetwork.py`: Runs the Fleming-Viot methodology on an `M/M/I/R` loss network,
i.e. a server system with no queues that is able to serve `I` different types of jobs --arriving with exponential inter-arrival times-- using `R` servers that serve jobs in exponential service times.
The script can be used to evaluate the impact of deterministic threshold-type admission control policies acting on the system.
In particular, it can be used to:
   1. Estimate the blocking probability, either using Fleming-Viot or Monte-Carlo.
   2. Estimate the long-run expected cost due to rejecting incoming jobs, either using Fleming-Viot or Monte-Carlo. 

   The script can be used to reproduce the violin plots presented in paper #1 (QUESTA) for the Loss Network system, that compare the Fleming-Viot estimation procedure with the Monte-Carlo estimation procedure
on different FV hyperparameters, such as the number `N` of particles, the number `T` of arrival events, and the size of the absorption set `A`, that is at the core of the Fleming-Viot methodology.

    For the execution details, see the documentation of the script by running: `python run_FV_LossNetwork.py --help`. 
3. `run_FVRL.py`: Runs the Fleming-Viot Reinforcement Learning algorithm to estimate optimum blocking sizes on either the single-server `M/M/1` queue system or on the `M/M/I/R` loss network.
For the execution details, see the documentation of the script by running: `python run_FVRL.py --help`.
4. Use `plot_FV.py` and `plot_FVRL.py` to generate plots of the results obtained by running respectively `run_FV.py / run_FV_LossNetwork.py` and `run_FVRL.py`.



# Appendix A
This appendix presents simulation setups that impact the results presented in the papers mentioned in the **Overview** section.

## A1) Simulation setup for the results presented in paper #1 (QUESTA)
In all simulations run (both for Fleming-Viot and Monte-Carlo estimation methods), we stick to the simulation estimation mechanisms described in Appendix A and
use a burn-in period of 10 system transitions before assuming the system entered the stationary regime and starting to collect the necessary data to compute estimators.

In addition, the FV estimator of the stationary probability, $\hat{p}_{FV}^\pi(x)$
is computed only when at least 5 return cycles (under the assumed stationarity) are observed to the absorption set $\mathcal{A}$ by the excursion of the underlying Markov chain.
This regular excursion is used to estimate the expected return cycle time appearing in the denominator of $\hat{p}_{FV}^\pi(x)$.
If a smaller number of return cycles are observed, the Fleming-Viot simulation is _not_ run and the stationary probability estimator $\hat{p}_{FV}^\pi(x)$ is set to $0$,
except in the estimation problem of the $M/M/1$ system where it is left undefined, thus reducing the number of successful FV estimations of the stationary probability
contributing to the violin plots evaluating the estimation properties.
