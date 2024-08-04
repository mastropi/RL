# -*- coding: utf-8 -*-
"""
Created on 22 Mar 2021

@author: Daniel Mastropietro
@description: Definition of policy for queues in terms of acceptance of new job of a given class
@reference: For the policy learned via a neural network (typically PolNN), I used Francisco Robledo's GitHub project,
https://github.com/frrobledo/LPQL_NN, in particular the file main/tools_nn.py where he defines a Q neural network
and then trains it.
"""


if __name__ == "__main__":
    # Needed to run tests (see end of program)
    import runpy
    runpy.run_path('../../../../setup.py')

import copy
import numpy as np

from gym.envs.toy_text.discrete import categorical_sample, DiscreteEnv

from torch import nn, tensor

import Python.lib.queues as queues
from Python.lib.agents.policies import AcceptPolicyType, GenericParameterizedPolicyTwoActions

from Python.lib.environments.queues import Actions, BufferType, GenericEnvQueueWithJobClasses, EnvQueueSingleBufferWithJobClasses, rewardOnJobClassAcceptance
from Python.lib.environments import EnvironmentDiscrete

from Python.lib.estimators import nn_models

from Python.lib.utils.basic import is_scalar


class QueueParameterizedPolicyTwoActions(GenericParameterizedPolicyTwoActions):
    """
    Superclass that defines common constructor and methods for some parameterized policies with two possible actions and a scalar parameter

    Arguments:
    env_queue: GenericEnvQueueWithJobClasses
        Queue environment on which the parameterized policy is applied.
        It should satisfy the conditions described in the documentation for the superclass GenericParameterizedPolicyTwoActions.

    theta: float
        Initial value for the parameter of the parameterized policy applied on the queue environment.

    accept_policy_type: AcceptPolicyType
        Type of acceptance policy as defined by the AcceptPolicyType Enum. Examples are: THRESHOLD policy and TRUNK_RESERVATION policy.
    """
    def __init__(self, env_queue: GenericEnvQueueWithJobClasses, theta: float, accept_policy_type: AcceptPolicyType=AcceptPolicyType.THRESHOLD):
        super().__init__(env_queue, theta)

        if self.getEnv().getCapacity() < np.Inf: # When capacity is Inf, theta is also allowed to be Inf
            assert theta < self.getEnv().getCapacity(), "The value of theta is smaller than the queue capacity"

        self.accept_policy_type = accept_policy_type

    def get_x_value_for_policy(self, state):
        """
        Computes the real value x to be used when evaluating the policy or its gradient

        state: Queue-environment dependent
            State of the environment at which the policy gradient is evaluated.
            It should contain at least the information about the queue state itself.

            Depending on the buffer type of the queue environment, the process of "getting" the SCALAR value x
            against which the scalar-parameterized policy is evaluated is as follows:
            1) for BufferType.SINGLE:
                - the QUEUE state is assumed to be a TUPLE.
                - the value of x is computed by calling getBufferSizeFromState(state).
            2) for BufferType.NOBUFFER:
                - the QUEUE state is assumed to be a SCALAR, which corresponds to the job occupancy of a given job class in the system.
                - the value of x is computed by calling getQueueStateFromState(state).

        Return: int
        The buffer size for single-buffer queue systems or the job class occupancy for queue systems with no buffer (e.g. loss networks).
        """
        if self.getEnv().getBufferType() == BufferType.SINGLE:
            # The queue environment has a single buffer whose value decides whether blocking occurs or may occur (i.e. probabilistic blocking)
            # => the buffer size is computed from the potentially multidimensional queue state
            buffer_size_or_jobclass_occupancy = self.getEnv().getBufferSizeFromState(state)
        elif self.getEnv().getBufferType() == BufferType.NOBUFFER:
            # The policy is assumed to apply separately for each job class
            # => the input `state` parameter is assumed to contain the job occupancy of the arriving job class as its "queue state" component
            # (i.e. its queue state component is assumed to be a SCALAR), which is retrieved using the getQueueStateFromState() method of the environment class
            queue_state = self.getEnv().getQueueStateFromState(state)
            buffer_size_or_jobclass_occupancy = queue_state
        else:
            raise ValueError("The buffer type of the environment is not a valid one for this parameterized policy ({}): {}" \
                             .format(type(self), env_queue.getBufferType().name))

        assert is_scalar(buffer_size_or_jobclass_occupancy)
        return buffer_size_or_jobclass_occupancy

    def getAcceptPolicyType(self):
        return self.accept_policy_type


class PolQueueTwoActionsLogit(QueueParameterizedPolicyTwoActions):
    """
    Randomized policy on a queue with two possible actions: accept or reject an incoming job.
    The policy is defined as a logit function around parameter theta where the probability of any of the action
    is 0.5.

    Specifically, the probability of acceptance is defined as 1 / (1 + exp( beta*(s - theta) ))
    where beta is a parameter passed to the constructor that controls how steep the transition from probability 1
    to 0 is.

    Arguments:
    env_queue: environment
        Environment representing a queue system.

    theta: float
        Initial value for the reference parameter where the policy is equal to 0.5.

    beta: positive float
        Parameter of the logit that defines the steepness of the transition from probability 1 to 0.
    """

    def __init__(self, env_queue: GenericEnvQueueWithJobClasses, theta: float, beta :float):
        super().__init__(env_queue, theta)
        self.beta = beta

    def getBeta(self):
        return self.beta

    def getGradient(self, action, state):
        """
        Returns the policy gradient at the given action when the environment is at the given state

        action: Actions
            Accept or Reject action at which the policy gradient is evaluated.
            It is assumed that the value of the action is 1 for accept and 0 for reject.

        state: Environment dependent
            State of the environment at which the policy gradient is evaluated.
            For more information about its expected type, see the documentation for the
            `QueueParameterizedPolicyTwoActions.get_x_value_for_policy()` method.

        Return: float
        Gradient of the policy for the given action given the environment's state.
        """
        sign = (action.value - 0.5) * 2

        if self.getPolicyForAction(action, state) != 0.0:
            # The following expression is only well defined when the probability for the action is not 0.0
            # But if it is 0.0, the gradient is actually 0.0 because the value of the policy does not change
            # when theta increases infinitesimally.
            gradient = sign * self.beta * (1 - self.getPolicyForAction(action, state))
        else:
            gradient = 0.0

        return gradient

    def getPolicyForAction(self, action, state):
        """
        Returns the value of the policy for the given action when the environment is at the given state

        action: Actions
            Accept or Reject action at which the policy is evaluated.

        state: Queue-environment dependent
            State of the environment at which the policy gradient is evaluated.
            For more information about its expected type, see the documentation for the
            `QueueParameterizedPolicyTwoActions.get_x_value_for_policy()` method.

        Return: float in [0, 1]
        Value of the policy (probability) for the given action at the given environment's state.
        """
        buffer_size_or_jobclass_occupancy = super().get_x_value_for_policy(state)

        if self.env.getBufferSizeFromState(state) >= self.env.getCapacity():
            # Reject when the queue environment is already operating at its capacity
            policy_accept = 0.0
        else:
            policy_accept = 1 / (1 + np.exp( self.beta * (buffer_size_or_jobclass_occupancy - self.getThetaParameter()) ))

        if action == Actions.ACCEPT:
            policy_value_for_action = policy_accept
        elif action == Actions.REJECT:
            policy_value_for_action = 1 - policy_accept
        else:
            raise ValueError(str(__class__) + ": An invalid action was given: {}".format(action))

        return policy_value_for_action

    def getDeterministicBlockingValue(self):
        return self.getDeterministicBlockingValueFromTheta(self.getThetaParameter())

    def getDeterministicBlockingValueFromTheta(self, theta):
        "What should we return in the case of a logit function for the parameterized policy? Perhaps choose the smallest K whose acceptance policy is close enough to 0.0 probability?"
        assert is_scalar(theta), "Parameter theta must be scalar: {}".format(theta)
        raise NotImplementedError


class PolQueueTwoActionsLinearStep(QueueParameterizedPolicyTwoActions):
    """
    Randomized policy on a queue with two possible actions: accept or reject an incoming job.
    The policy is defined as a stepwise linear function that transitions from deterministic acceptance to deterministic rejection.
    The location of the linear transition from acceptance to rejection happens at only one buffer size and is defined
    by the threshold parameter `theta`.

    Specifically, the probability of acceptance is defined as:
    - 1.0 if s <= theta
    - 0.0 if s >= theta + 1
    - theta - s + 1.0, o.w.
    where s is the current buffer size of the queue.

    Note that if theta is integer, the policy is deterministic for all buffer sizes.

    Ref: Massaro et al. (2019) "Optimal Trunk-Reservation by Policy Learning", pag. 3, from where we adapt
    the parameterized policy defined for multiple-class jobs.

    Arguments:
    env_queue: environment
        Environment representing a queue system.

    theta: float
        Initial value for the threshold parameter of the policy.
    """

    def __init__(self, env_queue: GenericEnvQueueWithJobClasses, theta: float):
        super().__init__(env_queue, theta)

    def getGradient(self, action, state):
        """
        Returns the policy gradient (with respect to the theta parameter)
        at the given action when the environment is at the given state

        action: Actions
            Accept or Reject action at which the policy gradient is evaluated.

        state: Queue-environment dependent
            State of the environment at which the policy gradient is evaluated.
            For more information about its expected type, see the documentation for the
            `QueueParameterizedPolicyTwoActions.get_x_value_for_policy()` method.

        Return: float
        Gradient of the policy for the given action given the environment's state.
        """
        buffer_size_or_jobclass_occupancy = super().get_x_value_for_policy(state)

        is_buffer_size_or_jobclass_occupancy_in_linear_piece = float( self.getThetaParameter() < buffer_size_or_jobclass_occupancy < self.getThetaParameter() + 1 )
        # Recall that the gradient is w.r.t. theta, NOT w.r.t. to the buffer size
        # which is how we normally plot the policy and against which the derivative has the opposite sign
        # compared to the sign of the derivative w.r.t. theta.
        if action == Actions.ACCEPT:
            slope = is_buffer_size_or_jobclass_occupancy_in_linear_piece
        elif action == Actions.REJECT:
            slope = -is_buffer_size_or_jobclass_occupancy_in_linear_piece
        else:
            raise ValueError(str(__class__) + ": An invalid action was given: {}".format(action))

        return slope

    def getPolicyForAction(self, action, state):
        """
        Returns the value of the policy for the given action when the queue environment is at the given state,
        i.e. it returns the *probability* that the given action is taken when the system is at the given state.

        action: Actions
            Accept or Reject action at which the policy is evaluated.

        state: Environment dependent
            State of the environment at which the policy is evaluated.
            For more information about its expected type, see the documentation for the
            `QueueParameterizedPolicyTwoActions.get_x_value_for_policy()` method.

        Return: float in [0, 1]
        Value of the policy (probability) for the given action at the given environment's state.
        """
        buffer_size_or_jobclass_occupancy = super().get_x_value_for_policy(state)

        if self.env.getBufferSizeFromState(state) >= self.env.getCapacity():
            # Reject when the queue environment is already operating at its capacity
            policy_accept = 0.0
        else:
            if buffer_size_or_jobclass_occupancy <= self.getThetaParameter():
                policy_accept = 1.0
            elif buffer_size_or_jobclass_occupancy >= self.getThetaParameter() + 1:
                policy_accept = 0.0
            else:
                # theta < s < theta + 1
                # => The acceptance policy is linear in s = 1 - rejection_probability(s, theta) = 1 - (s - theta)
                # Note that we lower bound theta to -1 because o.w. the acceptance policy would be negative (since the minimum value that s can take is 0)
                policy_accept = 1 - (buffer_size_or_jobclass_occupancy - max(-1, self.getThetaParameter()))
                assert 0 <= policy_accept <= 1, "The acceptance policy in the linear part is between 0 and 1 ({})".format(policy_accept)

        if action == Actions.ACCEPT:
            policy_value_for_action = policy_accept
        elif action == Actions.REJECT:
            policy_value_for_action = 1 - policy_accept
        else:
            raise ValueError(str(__class__) + ": An invalid action was given: {}".format(action))

        return policy_value_for_action

    def getDeterministicBlockingValue(self, func=np.ceil):
        "Returns the K associated to the theta parameter stored in the object as described by method getDeterministicBlockingValueFromTheta()"
        return self.getDeterministicBlockingValueFromTheta(self.getThetaParameter(), func=func)

    def getDeterministicBlockingValueFromTheta(self, theta, func=np.ceil):
        "Returns K, the first integer greater than theta + 1 when func=np.ceil (the default) or the closest integer to theta + 1 when func=np.round"
        assert is_scalar(theta), "Parameter theta must be scalar: {}".format(theta)
        return int( func( theta + 1 ) )


class PolQueueTwoActionsLinearStepOnJobClasses(GenericParameterizedPolicyTwoActions):
    """
    Randomized trunk-reservation policy on a queue with two possible actions: accept or reject an incoming job.
    The policy is defined as a stepwise linear function that transitions from deterministic acceptance to deterministic rejection.
    The location of the linear transition from acceptance to rejection happens at only one job class and is defined
    by the threshold parameter `theta`.

    Job classes can be thought of as job priorities.

    Specifically, the probability of acceptance is defined as:
    - 1.0 if job_class <= theta[k]
    - 0.0 if job_class >= theta[k] + 1
    - theta[k] - job_class + 1.0, o.w.
    where k is the current buffer size of the queue, which is limited to its capacity.

    Note that if theta[k] is integer, the policy is deterministic for all job classes.
    
    Ref:
    Massaro et al. (2019) "Optimal Trunk-Reservation by Policy Learning", pag. 3
    Keith W. Ross (1995) "Multiservice Loss Models for Broadband Communication Systems", Ch. 4: Admission Policies (for the trunk-reservation policy)

    Arguments:
    env_queue: environment
        Environment representing a queue system that accepts multiple job classes.

    theta: list
        Threshold parameter. Its length is equal to the capacity of the queue system.
    """

    def __init__(self, env_queue :GenericEnvQueueWithJobClasses, theta: list):
        super().__init__(env_queue, theta)

        # We now define the linear step policy
        # NOTE: if retrieving the policy becomes somehow inefficient for choosing the action
        # (because we are storing the policy for ALL the states)
        # we may want to AVOID the definition of the policy since, in order to determine the action to take
        # we just need to know the job class falling in between the two different deterministic selections of an action.
        K = self.getEnv().getCapacity()
        J = self.getEnv().getNumJobClasses()

        # Check the theta parameter given by the user
        if len(self.getThetaParameter()) != self.getEnv().getCapacity() + 1:
            raise ValueError("The number of theta parameters given ({}) must be equal to the capacity of the queue plus 1 ({})".format(len(theta), env_queue.getCapacity() + 1))
        if not all([0 <= t <= J-1 for t in self.getThetaParameter()]):
            # Note that the minimum accepted value for each theta dimension is -1 and not 0 because the job class numbers range from 0 to J-1
            # A value of -1 for theta implies that NONE of the job classes is accepted, not even job class 0.
            raise ValueError("All the values in the theta parameter of the policy must be between -1 and the number of classes minus 1 ({}): {}".format(J-1, self.getThetaParameter()))

        # Initialize acceptance policy to 0.0 which applies to job classes j >= theta[k] + 1
        # Note that
        self.policy_accept = np.zeros((K + 1, J), dtype=float)

        # Define acceptance policy with non-zero probability
        # for buffer sizes < K and job classes j <= theta[k]
        for k in range(K+1):
            max_jobclass_with_accept_proba_1 = int( np.min([np.floor(theta[k])  , J-1]) )
            min_jobclass_with_accept_proba_0 = int( np.min([np.floor(theta[k])+1, J-1]) )
            #print("k={} of K={}: max_j_proba_1={}, min_j_proba_0={}".format(k, K, max_jobclass_with_accept_proba_1, min_jobclass_with_accept_proba_0))
            assert max_jobclass_with_accept_proba_1 == min_jobclass_with_accept_proba_0 - 1 if max_jobclass_with_accept_proba_1 < J-1 \
                else max_jobclass_with_accept_proba_1 == min_jobclass_with_accept_proba_0
            for j in range(max_jobclass_with_accept_proba_1 + 1):
                # Policy of accept = 1.0 for j <= theta[k]
                #print("1: j={}".format(j))
                self.policy_accept[k, j] = 1.0
            for j in range(max_jobclass_with_accept_proba_1 + 1, min_jobclass_with_accept_proba_0 + 1):
                # Policy of accept = theta[k] - j + 1 for theta[k] < j < theta[k] + 1
                #print("2: j={}".format(j))
                self.policy_accept[k, j] = theta[k] - j + 1
                assert 0 <= self.policy_accept[k, j] <= 1, "The acceptance policy in the linear part is between 0 and 1 ({})".format(self.policy_accept[k, j])

            print("Acceptance policy as a function of the job class for buffer size k = {} (theta[k]={:.1f}): {}".format(k, theta[k], self.policy_accept[k]))

    def getGradient(self, action, state):
        """
        Returns the policy gradient at the given action when the environment is at the given state

        action: Actions
            Accept or Reject action at which the policy gradient is evaluated.

        state: tuple
            State of the environment at which the policy gradient is evaluated.
            It should be a duple containing the queue's buffer size and the job class of the latest arriving job.

        Return: float
        Gradient of the policy for the given action given the environment's state.
        """
        buffer_size, job_class = state

        is_job_class_in_linear_piece = self.getThetaParameter()[buffer_size] < job_class < self.getThetaParameter()[buffer_size] + 1
        if action == Actions.ACCEPT:
            slope = is_job_class_in_linear_piece
        elif action == Actions.REJECT:
            slope = -is_job_class_in_linear_piece
        else:
            raise ValueError(str(__class__) + ": An invalid action was given: {}".format(action))

        return slope

    def getPolicyForAction(self, action, state):
        """
        Returns the value of the policy for the given action when the environment is at the given state

        action: Actions
            Accept or Reject action at which the policy is evaluated.

        state: tuple
            State of the environment at which the policy is evaluated.
            It should be a duple containing the queue's buffer size and the job class of the latest arriving job.

        Return: float in [0, 1]
        Value of the policy (probability) for the given action at the given environment's state.
        """
        buffer_size, job_class = state

        if action == Actions.ACCEPT:
            policy_value_for_action = self.policy_accept[buffer_size, job_class]
        elif action == Actions.REJECT:
            policy_value_for_action = 1 - self.policy_accept[buffer_size, job_class]
        else:
            raise ValueError(str(__class__) + ": An invalid action was given: {}".format(action))

        return policy_value_for_action

    def getAcceptancePolicy(self):
        return self.policy_accept


class PolNN:
    """
    Parameterized policy that is the output of a neural network model fitted with PyTorch
    which is assumed to output one preference for each possible action in the environment.

    env: EnvironmentDiscrete
        Discrete-state/action environment on which the agent acts.

    nn_model: Neural network inheriting from torch.nn.Module
        Neural network model typically defined in nn_models.py.
        The network should have as input either one neuron representing the state or as many neurons as states (i.e. dummy or one-hot coding of states)
        and as many output neurons as the number of possible actions over all possible states in the environment.
        Note: The parameters of the model can be retrieved with nn_model.parameters(), which is a generator.
    """
    def __init__(self, env: EnvironmentDiscrete, nn_model: nn.Module):
        self.env = env
        self.nn_model = nn_model

        if self.nn_model.getNumInputs() != 1 and self.nn_model.getNumInputs() != self.env.getNumStates():
            raise ValueError(f"The number of input values to the neural network model must be either 1 or as many as the number of states in the environment ({self.env.getNumStates()}): {self.nn_model.getNumInputs()}")
        if self.nn_model.getNumOutputs() != self.env.getNumActions():
            raise ValueError("The environment where the model is to be applied and the model itself are incompatible in the number of actions they deal with."
                             f"\n# environment actions = {self.env.getNumActions()}"
                             f"\n# model actions = {self.nn_model.getNumOutputs()}")

        if self.nn_model.getNumHiddenLayers() == 0:
            # Set `requires_grad=False` for neural network models with no hidden layer
            # because that means that we are using the NN model simply as an implementation of action preferences via the softmax function,
            # i.e. no backpropagation will be carried out on the network, instead the policy will be set proportional to exp(advantage-function).
            # This is important, o.w. the neural network parameters will not be able to be set without triggering the error message
            # "RuntimeError: a view of a leaf Variable that requires grad is being used in an in-place operation".
            self.nn_model.getOutputLayer().bias.requires_grad = False
            self.nn_model.getOutputLayer().weight.requires_grad = False

    def reset(self, initial_values=None, seed=None):
        "Resets the policy to the random walk or to the given initial values, optionally using a seed for the random initialization of the neural network weights"
        self.init_policy(values=initial_values, seed=seed)

    def init_policy(self, values=None, eps=1E-2, seed=None):
        """
        Initializes the parameters of the neural network so that the output policy is either almost constant over all actions for all input states
        or is almost equal to the `values` given (which are indexed by all possible actions), also for all input states.

        In the particular case of a neural network model with NO hidden layer, the model is actually assumed to be just an implementation
        of action preferences via the softmax function applied when retrieving the policy values with the get_policy_values() method,
        i.e. it is assumed that NO backpropagation will be performed on the neural network to update its parameters.
        Instead, parameters are always SET by calling either the set_policy_values() method or the set_model_params() method.
        Therefore, the policy values can be (easily) set differently for each state (o.w. in the general neural network model with hidden layers,
        it is difficult to define a parameter set that gives the desired output as a function of the input state), in which case the `values` argument
        should be a 2D array-like object of size #states x #actions.
        But, as in the full neural network case with hidden layers, it can also be a 1D array-like object giving the policy values to use for ALL states.
        In this case of the neural network being used just as an implementation of action preferences, NO noise is added to the parameter values
        when calling this init_policy() method because they are not needed, as no backpropagation will be run on the neural network, therefore the problem
        of not having the model parameters updated because of all weights being equal to 0, described in the next paragraphs, does not happen.

        In the regular neural network model with hidden layers, parameters (weights and biases) are initialized using a zero-mean normal distribution
        with a small standard deviation (compared to 1) given by `eps`, except for the biases of the neurons in the output layer which are initialized
        so that the policy is almost equal to the given `values` or to a constant value for all actions, if no `values` are given
        (we write "almost" because there is noise in the output policy induced by the almost zero values
        to which the other weights and biases are initialized to, but this is perfectly ok for most purposes).

        We need to initialize parameters randomly because, if all parameters were initialized to 0, their values would never change during the learning process(!)
        as the gradient would be always 0 due to the chain rule (easy to prove).

        IMPORTANT: for reproducibility, the torch seed should be set prior to calling this method, for instance through the torch.manual_seed() method.

        Arguments:
        values: (opt) array-like
            Values defining the policy, i.e. conceptually the probability of each possible action for each possible state in the `self.env` environment.
            The admitted values depend on whether the neural network has a hidden layer or not:
            - if the NN has hidden layers, the values can be any real number because they are converted to probabilities using the softmax function by
            the self.get_policy_values() method.
            - if the NN does NOT have hidden layers, the values should be NON-NEGATIVE because they are transformed by the log() function
            in order to set the NN weight values, which are then converted back to probabilities using the softmax function when the self.get_policy_values()
            method is called.
            In terms of shape, it can be either:
            - a 1D list or array-like of values indexed by the possible actions to which the policy should be initialized for all states or
            - a 2D list or array-like of values indexed by state (across rows) and action (across columns) of the policy to be set for each state.
            The 2D option is ONLY valid when there is no hidden layer in the neural network.
            default: None, in which case the policy is initialized to (almost) a uniform policy

        eps: (opt) positive float
            Small value defining the standard deviation of the normal distribution used to define the weights and biases of all layers except for
            the biases of the neurons in the output layer.
            default: 1E-2

        seed: (opt) int
            Seed to use in the random initialization of the neural network weights as described above.
            Only used when the neural network has at least ONE hidden layer. Otherwise, the neural network used to model the policy
            is assumed to be used simply as an implementation of action preferences, meaning that NO backpropagation is performed to learn its parameters,
            but simply to set the policy values using the softmax function on the weights connecting the input neurons to the output neurons
            (whose biases are set to zero).
            default: None
        """
        # Inspired by the weights_init() function in this code:
        # https://github.com/pytorch/examples/blob/main/dcgan/main.py
        # Note also the use of Module.apply() method which calls a function on each submodule of a module (e.g. of a neural network).

        if seed is not None:
            nn.init.torch.manual_seed(seed)

        if self.nn_model.getNumHiddenLayers() == 0:
            # The neural network is assumed to be used simply as an implementation of action preferences
            # (e.g. as in a Natural Policy Gradient context, where the policy is proportional to the exponential of the advantage function)
            # which is materialized by using one-hot encoding of the environment states as input layer, and connecting each input neuron directly to each output neuron.
            # => Set the weights of the input-to-output layer (for EACH input state) as `log(values)` (so that they become `values` when applying the softmax function)
            # and set the biases of the output layer to zero (because no biases should appear in the softmax function in this case).
            assert self.nn_model.getNumInputs() == self.env.getNumStates(), f"The number of neurons in the input layer ({self.nn_model.getNumInputs()} " \
                                                                            f"must be equal to the number of environment states ({self.env.getNumStates()}) " \
                                                                            "when the neural network does not have any hidden layer, because the network is assumed to be simply " \
                                                                            "a transmitter of action preferences from the input (states) to the output (actions)."
            # Bias
            self.nn_model.getOutputLayer().bias = nn.Parameter(tensor(np.zeros(self.nn_model.getNumOutputs())), requires_grad=False)
            # Weights
            if values is None:
                nn.init.zeros_(self.nn_model.getOutputLayer().weight)
                # Set the requires_grad option to False to the weights so that we can update them freely as needed because we are NOT using the neural network for backpropagation
                # when there is no hidden layer.
                self.nn_model.getOutputLayer().weight.requires_grad = False
            else:
                # Initial weight values are given by the user, so we set them here as described in the above comment "Set the weights..."
                # The weights connecting the input to the output layer are set to the `values` array arranged as a column repeated on all columns of the `.weight` 2D array
                # which is of size <# output neurons> x <# input neurons>.
                # Note that we use the broadcast method (i.e. multiplying the `values` array by an array of 1's (`matrix_of_ones`) instead of np.repeat() because broadcasting seems to be faster
                # than np.repeat() as mentioned here: https://stackoverflow.com/questions/65795393/create-a-numpy-array-with-all-rows-identical)

                # Check that the given values are between 0 and 1
                values = np.array(values)
                if np.min(values) < 0:
                    raise ValueError(f"The values in parameter `values` must be non-negative: {values}")
                # We first compute the log(values) because the policy is computed as the softmax function applied to the preferences (i.e. softmax(h) ~ exp(h), where h=preference),
                # and therefore the preferences (which here are the weights) are defined as log(values)
                # Note that we use the maximum() function to avoid log(0) in case any value in `values` contains zeros.
                log_values = np.log(np.maximum(1E-9, np.array(values)))
                if np.array(values).ndim == 1:
                    # The `values` array is 1D, in which case we assume that the same policy should be applied to all states
                    if len(values) != self.env.getNumActions():
                        raise ValueError(f"The number of policy values to set ({len(values)}) must match the number of possible actions ({self.env.getNumActions()})")
                    matrix_of_ones = np.ones((self.nn_model.getNumOutputs(), self.nn_model.getNumInputs()))
                    weights = np.array(log_values, ndmin=2).T * matrix_of_ones
                    self.nn_model.getOutputLayer().weight = nn.Parameter(tensor(weights), requires_grad=False)  # We are NOT going to compute the gradient because we don't do backpropagation in neural networks with no hidden layer (see assumption above)
                else:
                    # The `values` array is 2D, in which case the policy may change with the state
                    if len(np.array(values).shape) != 2:
                        raise ValueError(f"The policy values to set must be 2D: {np.array(values).shape}\n{values}")
                    if len(values) != self.env.getNumStates():
                        raise ValueError(f"The first dimension of the policy values to set ({len(values)}) must match the number of states in the environment ({self.env.getNumStates()})")
                    for s in range(self.env.getNumStates()):
                        weights = log_values[s]
                        self.nn_model.getOutputLayer().weight[:, s] = nn.Parameter(tensor(weights), requires_grad=False)
        else:
            # The neural network is assumed to be used as an actual neural network model (it has hidden layers)
            # => The initial policy is set approximately by setting small random weights around 0 and
            # the bias of the output layer that are required to get the policy values in the output layer.
            for i, component in enumerate(self.nn_model.modules()):
                # Use the following if we want to filter out specific module types. But the problem is that this is not an exhaustive list!
                # So, I think using `try` is better (as done below)
                #if component.__class__.__name__ not in [self.nn_model.__class__.__name__, "ModuleList"]:
                try:
                    # Only actual layers containing weights are capable of being initialized
                    # Initializing all parameters to 0 prevents their update during learning as they are always stuck at 0!
                    #nn.init.zeros_(component.weight)
                    #nn.init.zeros_(component.bias)
                    nn.init.normal_(component.weight, 0, eps)
                    nn.init.normal_(component.bias, 0, eps)
                except:
                    pass

            # Set the bias of the neurons in the output layer (which is the last module referenced in the above loop)
            # which are the ones that define the policy.
            if values is None:
                # Initialize the biases to the same value (1) so that the policy is almost constant for all possible actions
                nn.init.ones_(component.bias)
            else:
                # Initialize the biases so that the policy is almost equal to the given values (indexed by the possible actions)
                # The initialization is based on the fact that the probability of each action in the output layer is given by the softmax function
                # applied on the weights and biases reaching each output neuron. Since weights are approximately 0, the probabilities are approximately equal to
                # the softmax function applied on the biases.
                # The initialization performed below (as bias = log(p)) gives biases whose exponential sum is equal to 1
                # (i.e. the denominator of the softmax function sums up to 1, i.e. sum(exp(beta)) = 1).
                if len(values) != len(component.bias):
                    raise ValueError(f"The length of parameter `values` ({len(values)}) must be the same as the number of neurons in the output layer ({len(component.bias)})")
                component.bias = nn.Parameter(tensor(np.log( np.maximum(1E-9, np.array(values)) )))

    def set_policy_values(self, values=None, eps=1E-2, seed=None):
        self.init_policy(values, eps, seed)

    def set_model_params(self, param_values):
        """
        Sets the model parameters to the given values if the model is of type `nn_models.NNBackprop`

        See the documentation of NNBackprop.set_params() for details of the structure of parameter `param_values`,
        """
        if not isinstance(self.nn_model, nn_models.NNBackprop):
            raise NotImplementedError(f"The model stored in the policy must be of type `nn_models.NNBackprop` in order to set its parameter values: {type(self.nn_model)}")
        self.nn_model.set_params(param_values)

    def choose_action(self, state):
        """
        Choose an action given the state

        state: int
            Environment state at which the action should be chosen.

        Return: int
        Index of the action chosen by the policy.
        """
        proba_actions = self.getPolicyForState(state)

        # Choose the action
        if isinstance(self.env, DiscreteEnv):
            # Use the random number generator defined in the environment (using the function defined in gym.envs.discrete.toy_text.discrete)
            # to choose an action, so that we use the same generator that has been used elsewhere.
            action = categorical_sample(proba_actions, self.env.np_random)
        else:
            # Use np.random.choice to choose an action
            action = np.random.choice(range(self.env.getNumActions()), p=proba_actions)

        return action

    def get_policy_values(self):
        "Returns the policy values in a 2D array indexed by each state and each action of the environment"
        policy = np.nan * np.ones((self.env.getNumStates(), self.env.getNumActions()))
        for s in self.env.getAllStates():
            for a in range(self.env.getNumActions()):
                policy[s][a] = self.getPolicyForAction(a, s)

        return policy

    def getPolicyForState(self, state):
        "Returns the probability of choosing each possible action for the given state"
        # Compute the action preferences output by the neural network (each output node represents a possible action)
        if self.nn_model.getNumInputs() == 1:
            exp_preferences = np.exp(self.nn_model([state]).tolist())  # We use tolist() to avoid the error message "Numpy not found"
        elif self.nn_model.getNumInputs() == self.env.getNumStates():
            input = np.zeros(self.env.getNumStates(), dtype=int)
            input[state] = 1
            exp_preferences = np.exp(self.nn_model(input).tolist())
        else:
            raise ValueError(f"The number of inputs in the neural network ({self.nn_model.getNumInputs()}) cannot be handled by the policy learner. It must be either 1 or as many as the number of states in the environment.")

        # Compute the action probabilities from the action preferences using the soft-max function
        proba_actions = exp_preferences / np.sum( exp_preferences )
            ## Note: we could compute the probability of the different actions using the torch function `F.softmax(self.nn_model([state]), dim=0).tolist()`
            ## but astonishingly this gives probabilities that do not sum up to 1!!!
            ## (not exactly at least, they sum up to e.g. 0.999999723, but this makes the call np.random.choice() fail)
        #print(f"State: {state}, Prob. Actions = {proba_actions}")

        return proba_actions

    def getPolicyForAction(self, action, state):
        """
        Returns the value of the policy for the given action when the environment is at the given state

        action: int
            Valid value of the environment's action space

        state: int
            Valid value for the environment's state space

        Return: float in [0, 1]
        Value of the policy (probability) for the given action at the given environment's state.
        """
        proba_actions = self.getPolicyForState(state)
        policy_value_for_action = proba_actions[action]

        return policy_value_for_action

    def getThetaParameter(self):
        "Returns the parameters of the neural network"
        return self.nn_model.parameters()

    def copy(self, keep_same_environment=True):
        """
        Creates a copy of the object by optionally keeping the same reference of the environment on which it acts (default).

        In general we would like to keep the same environment in the copy (i.e. the environment in the copy should reference the same environment
        in the original policy) for REPRODUCIBILITY purposes.
        In fact, actions taken by the policy are chosen using the RANDOM NUMBER GENERATOR stored in the environment attribute of the policy object, self.env.
        So, if we create a policy passing an environment on which random numbers will be generated to simulate the agent's interaction with the environment
        using such policy, and we set the seed of the environment at the beginning of the simulation, we need to make sure that the random number generator
        defined by that initial seeding step is the same random number generator used by the self.choose_action() method defined in this policy object
        to generate an action, namely the random number generator stored in the self.env attribute of this policy object (typically called `np_random`).
        If the environment attribute of this policy object is DIFFERENT from the environment whose seed was set at the beginning of the simulation,
        (which typically is the environment passed to a Simulator object, typically the same environment that is used to instantiate this policy object
        when defining a policy), the agent's interaction with the environment will NOT be reproducible, even if we set the environment's seed
        at the beginning of the simulation.

        Note: A use case of copying a policy is to compare the values of policies learned by different agents at the end of a policy learning process,
        such as an Actor-Critic policy learning algorithm.
        """
        policy_copy = copy.deepcopy(self)

        if keep_same_environment:
            # VERY IMPORTANT TO KEEP THE SAME ENVIRONMENT IN THE COPY OF THE POLICY FOR REPRODUCIBILITY!! (see explanation in method's documentation above)
            policy_copy.env = self.env

        return policy_copy


if __name__ == "__main__":
    #------------------------- Unit tests -----------------------------#
    #--- Test #1: Single-server, single-class job: stepwise linear policy on buffer size
    print("\n***********\nTest #1: Tests on Stepwise-linear policy on a single server with single threshold parameter theta")
    job_class_rates = [0.7]
    job_rates_by_server = job_class_rates     # Since this is a single server system on a SINGLE class, the job arriving rate to the server is the same as the job class rate
    service_rates = [1]
    nservers = 1
    capacity = np.Inf
    rewards_accept_by_job_class = [0]
    queue = queues.QueueMM(job_rates_by_server, service_rates, nservers, capacity)
    env_queue = EnvQueueSingleBufferWithJobClasses(queue, job_class_rates, rewardOnJobClassAcceptance, rewards_accept_by_job_class)

    #-- Thresholds
    # theta <= 0: No new job is accepted
    # theta very close to 0
    theta = -0.3
    print("\ntheta: {}".format(theta))
    policy = PolQueueTwoActionsLinearStep(env_queue, theta)

    buffer_size = 0
    env_queue.setState((buffer_size, None))
    state = env_queue.getState()
    print("\tTesting buffer_size = {}... Policy Accept = {}".format(buffer_size, policy.getPolicyForAction(Actions.ACCEPT, state)))
    assert policy.getPolicyForAction(Actions.ACCEPT, state) == 0.7
    assert policy.getPolicyForAction(Actions.REJECT, state) == 1 - policy.getPolicyForAction(Actions.ACCEPT, state)

    buffer_size = 1
    env_queue.setState((buffer_size, None))
    state = env_queue.getState()
    print("\tTesting buffer_size = {}... Policy Accept = {}".format(buffer_size, policy.getPolicyForAction(Actions.ACCEPT, state)))
    assert policy.getPolicyForAction(Actions.ACCEPT, state) == 0.0
    assert policy.getPolicyForAction(Actions.REJECT, state) == 1 - policy.getPolicyForAction(Actions.ACCEPT, state)

    # theta a little farther away than 0
    # The policy for all buffer sizes should be equal to 0
    theta = -1.3
    print("\ntheta: {}".format(theta))
    policy = PolQueueTwoActionsLinearStep(env_queue, theta)
    for buffer_size in range(3):
        env_queue.setState((buffer_size, None))
        state = env_queue.getState()
        print("\tTesting buffer_size = {}... Policy Accept = {}".format(buffer_size, policy.getPolicyForAction(Actions.ACCEPT, state)))
        assert policy.getPolicyForAction(Actions.ACCEPT, state) == 0.0
        assert policy.getPolicyForAction(Actions.REJECT, state) == 1 - policy.getPolicyForAction(Actions.ACCEPT, state)

    # theta > 0, non-integral
    theta = 5.3
    print("\ntheta: {}".format(theta))
    policy = PolQueueTwoActionsLinearStep(env_queue, theta)

    buffer_size_small = 3
    env_queue.setState((buffer_size_small, None))
    state = env_queue.getState()
    print("\tTesting buffer_size smaller than theta: {}... Policy Accept = {}".format(buffer_size_small, policy.getPolicyForAction(Actions.ACCEPT, state)))
    assert policy.getPolicyForAction(Actions.ACCEPT, state) == 1.0
    assert policy.getPolicyForAction(Actions.REJECT, state) == 1 - policy.getPolicyForAction(Actions.ACCEPT, state)

    buffer_size_large = 8
    env_queue.setState((buffer_size_large, None))
    state = env_queue.getState()
    print("\tTesting buffer_size = {}, MUCH larger than theta... Policy Accept = {}".format(buffer_size_large, policy.getPolicyForAction(Actions.ACCEPT, state)))
    assert policy.getPolicyForAction(Actions.ACCEPT, state) == 0.0
    assert policy.getPolicyForAction(Actions.REJECT, state) == 1 - policy.getPolicyForAction(Actions.ACCEPT, state)

    buffer_size_border_blocked = int(theta+2)
    env_queue.setState((buffer_size_border_blocked, None))
    state = env_queue.getState()
    print("\tTesting buffer_size = {}, near theta with sure blocking... Policy Accept = {}".format(buffer_size_border_blocked, policy.getPolicyForAction(Actions.ACCEPT, state)))
    assert policy.getPolicyForAction(Actions.ACCEPT, state) == 0.0
    assert policy.getPolicyForAction(Actions.REJECT, state) == 1 - policy.getPolicyForAction(Actions.ACCEPT, state)

    buffer_size_border_random = int(theta+1)
    env_queue.setState((buffer_size_border_random, None))
    state = env_queue.getState()
    print("\tTesting buffer_size = {}, near theta with randomized blocking... Policy Accept = {}".format(buffer_size_border_random, policy.getPolicyForAction(Actions.ACCEPT, state)))
    assert np.isclose(policy.getPolicyForAction(Actions.ACCEPT, state), 0.3)
    assert policy.getPolicyForAction(Actions.REJECT, state) == 1 - policy.getPolicyForAction(Actions.ACCEPT, state)

    #--- Test #2: Single-server, multi-class job: stepwise linear policy
    print("\n***********\nTest #2: Tests on Stepwise-linear policy on a single server with one threshold parameter theta for each job class")
    job_class_rates = [0.7, 0.3, 0.4, 0.9]  # Note: This rates are currently NOT used in the test. The list just defines the number of possible job classes
    job_rates_by_server = [0.7]             # This value is not used at all, it's just needed to create the queue system below
    service_rates = [1]                     # This value is not used in the test but needed to create the queue system below
    nservers = 1
    capacity = 6
    rewards_accept_by_job_class = [1, 0.8, 0.3, 0.2]
    queue = queues.QueueMM(job_rates_by_server, service_rates, nservers, capacity)
    env_queue = EnvQueueSingleBufferWithJobClasses(queue, job_class_rates, rewardOnJobClassAcceptance, rewards_accept_by_job_class)

    # Thresholds as a function of the buffer size (length = capacity+1)
    # (the threshold values are given from top (higher buffer size) to bottom (lower buffer size) and then reversed)
    # Note the following extreme cases for testing purposes:
    # - the threshold for the largest job class is set to be negative
    # - the threshold for job class 0 is set way larger than the queue's capacity
    theta = list( reversed( [-2.0, 1.0, 1.3, 1.3, 2.8, 2.9, 5.0] ))
    print("\ntheta: {}".format(theta))
    policy = PolQueueTwoActionsLinearStepOnJobClasses(env_queue, theta)

    # 1.1: Highest priority job is always accepted
    # regardless of the buffer size (unless the buffer is full)
    job_class = 0
    env_queue.setState((buffer_size, job_class))
    state = env_queue.getState()
    assert policy.getPolicyForAction(Actions.ACCEPT, state) == 1.0
    assert policy.getPolicyForAction(Actions.REJECT, state) == 1 - policy.getPolicyForAction(Actions.ACCEPT, state)

    # 1.2: In this setup, jobs with lowest priority are accepted with non-zero probability lower than 1
    # regardless of the buffer size
    job_class = 2
    env_queue.setState((3, job_class))
    state = env_queue.getState()
    print("Prob(action=1/s={},theta={:.1f}) for a job class {} (falling between theta and theta+1): {}" \
          .format(env_queue.getJobClass(), env_queue.getBufferSize(), policy.theta[job_class], policy.getPolicyForAction(Actions.ACCEPT, state)))
    assert 0 < policy.getPolicyForAction(Actions.ACCEPT, state) and policy.getPolicyForAction(Actions.ACCEPT, state) < 1.0
    assert policy.getPolicyForAction(Actions.REJECT, state) == 1 - policy.getPolicyForAction(Actions.ACCEPT, state)
    assert np.abs(policy.getPolicyForAction(Actions.ACCEPT, state) - 0.30) < 1E-6

    # 1.3: Any job is rejected when the buffer is full
    env_queue.setState((env_queue.queue.getCapacity(), 0))
    state = env_queue.getState()
    assert policy.getPolicyForAction(Actions.ACCEPT, state) == 0.0
