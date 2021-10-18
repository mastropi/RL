# -*- coding: utf-8 -*-
"""
Created on Thu Jun  6 19:39:44 2020

@author: Daniel Mastropietro
@description: Definition of classes defining queues.
"""

from enum import Enum, unique

import warnings
import numpy as np
import copy

if __name__ == "__main__":
    # Needed to run tests (see end of program)
    import runpy

    runpy.run_path('../../../setup.py')

    from Python.lib.utils.basic import is_scalar
else:
    from utils.basic import is_scalar


@unique
class Event(Enum):
    # The values of each element representing normal events (e.g. birth, death)
    # should be a valid INDEX for a list storing event types, starting at 0
    BIRTH = 0
    DEATH = 1
    RESIZE = 9
    RESET = -1
    END = 99

class GenericQueue:
    """
    Class that holds information that is present in all queues
    
    Arguments:
    capacity: positive int or np.Inf
        Capacity of the buffer that collects all the arriving jobs before being assigned to a server.
        The capacity can be infinite.
    
    nservers: (opt) positive int
        Number of servers to process jobs in the system.
        default: 1

    size: (opt) non-negative int, list or array
        Size of the queue of each server.
        default: 0

    origin: (opt) float
        Time value that should be used as time origin.
        All communication of event times with the object (set or get) should be interpreted as
        times relative to this time origin.
        Internally the event time is stored as an absolute value.
        For example an apply_event(t) method would set the internal event time of the queue as `t + origin`.
        default: 0.0

    log: (opt) bool
        Whether to activate logging messages about operations run on the object.
        default: False
    """

    def __init__(self, capacity: int or np.Inf, nservers: int=1, size: int=0, origin: float=0.0, log: bool=False):
        self.K = capacity           # Capacity of the buffer that receives the jobs before assigning them to a server
        self.c = nservers

        self.setServerSizes(size)
        self.size_initial = copy.deepcopy(self.size)
        self.origin = origin
        self.LOG = log

        self.reset()

    def reset(self, size=None):
        """
        Resets the queue to optionally a given size.
        If the size is not given, it resets the queue to the original size with which it was first constructed.
        """
        if size is None:
            self.size = self.size_initial
        else:
            self.size = size
        self.last_change = np.zeros(self.c, dtype=int)

    def resize(self, size):
        if len(size) != self.getNServers():
            warnings.warn("The number of sizes to set ({}) and the number of servers ({}) differ." \
                          "\nNo resize done." \
                          .format(len(size), self.getNServers()))
            return
        if not (0 <= np.sum(size) <= self.getCapacity()):
            warnings.warn("Invalid size values given (0 <= sum(size) <= {}) (size={})." \
                          "\nNo resize done." \
                          .format(self.getCapacity(), size))
            return

        previous_size = self.size
        self.size = np.array(size)

        # Store how the size changed from the size prior to the resize just performed
        self.last_change = self.size - previous_size

    def add(self, server=0):
        if not self.isFull():
            self.size[server] += 1
            self.last_change[server] = +1
        else:
            self.last_change[server] = 0
            if self.LOG:
                print("Queue is full (size={}): no job added".format(self.size))

    def remove(self, server=0):
        if not self.isEmpty(server):
            self.size[server] -= 1
            self.last_change[server] = -1
        else:
            self.last_change[server] = 0
            if self.LOG:
                print("Queue is empty (size={}): no job removed".format(self.size))

    def isEmpty(self, server=0):
        "Whether the server has no jobs to be served or being served"
        return self.size[server] == 0

    def isFull(self):
        "Whether the buffer capacity (one buffer for ALL servers) has been reached"
        return np.sum(self.size) == self.K

    # GETTERS
    def getNServers(self):
        return self.c

    def getCapacity(self):
        return self.K

    def getOrigin(self):
        return self.origin

    def getServerSize(self, server=0):
        "Returns the size of the given server"
        return self.size[server]

    def getServerSizes(self):
        """
        Returns a COPY of the `size` attribute which is an array.
        A copy is returned so that changes in the object's attribute doesn't change
        the value of the variable on which the returned value is assigned.
        """
        return copy.deepcopy(self.size)

    def getLastChange(self, server=None):
        "Returns the last change in the queue size of each server"
        if server is None:
            return self.last_change
        else:
            return self.last_change[server]

    def getBufferSize(self):
        return np.sum(self.size)

    # SETTERS
    def setOrigin(self, time_origin):
        self.origin = time_origin

    def setServerSizes(self, size):
        "Sets the size of the queue in EACH server"
        # Convert the size to an array (and to int if given as float)
        if is_scalar(size):
            size = np.repeat(int(size), self.c)
        self.size = np.array(size)


class QueueMM(GenericQueue):

    def __init__(self, rates_birth, rates_death, nservers: int, capacity: int, origin: float=0.0):
        super().__init__(capacity, nservers=nservers, size=0, origin=origin)
        is_rates_birth_scalar = not isinstance(rates_birth, (list, tuple, np.ndarray))
        is_rates_death_scalar = not isinstance(rates_death, (list, tuple, np.ndarray))
        if  not is_rates_birth_scalar and len(rates_birth) != nservers or \
            not is_rates_death_scalar and len(rates_death) != nservers:
            raise ValueError("Both birth and death rates must be either just one value " \
                             "({} birth rates given, {} death rates given) or be as many as the number of servers ({})." \
                             "\nThe process stops." \
                             .format(len(rates_birth), len(rates_death), nservers))
        if  is_rates_birth_scalar and rates_birth < 0 or \
            is_rates_death_scalar and rates_death < 0 or \
            not is_rates_birth_scalar and any(np.array(rates_birth) < 0) or \
            not is_rates_death_scalar and any(np.array(rates_death) < 0):
            raise ValueError("Both birth and death rates must be non-negative (birth rates: {}, death rates: {})." \
                             "\nThe process stops." \
                             .format(rates_birth, rates_death))
        # The rates are stored in a 2D array:
        # - Dim 1 indexes the possible servers
        # - Dim 2 indexes the possible events
        self.rates = np.repeat([[0.0]*2], nservers, axis=0)  # 2 is the number of possible events: birth and death
        if is_rates_birth_scalar:
            rates_birth = np.repeat(rates_birth, nservers)
        if is_rates_death_scalar:
            rates_death = np.repeat(rates_death, nservers)
        for server in range(nservers):
            self.rates[server][Event.BIRTH.value] = rates_birth[server]
            self.rates[server][Event.DEATH.value] = rates_death[server]

        self.reset()

    def reset(self, size=None):
        # We need to call the SUPER reset() method because the `self.reset()` call in the super class
        # calls THIS reset() method, NOT the SUPER method!!
        super().reset(size=size)
        # Type and time of the last event by server
        # They are stored as ARRAYS instead of lists in order to facilitate operations on them during simulations.
        self.types_last_event = np.repeat(Event.RESET, self.getNServers())
        self.times_last_event = self.origin + np.zeros(self.getNServers(), dtype=float)

    def apply_events(self, events: list, event_times: list):
        """
        Applies a set of events to all servers in the system
        and returns the change in queue size in each server.
        
        Arguments:
        events: list of Event
            Types of the events to apply. Either Event.BIRTH or Event.DEATH.
        
        event_times: list of positive float
            Times of the events to apply.
        
        Return: array
        Array containing the change in queue size on each server in the system after the events are applied.
        """
        if not isinstance(events, (list, tuple, np.ndarray)):
            warnings.warn("The events argument must be a list, tuple, or array: {}".format(events))
        if not isinstance(event_times, (list, tuple, np.ndarray)):
            warnings.warn("The event_times argument must be a list, tuple, or array: {}".format(event_times))

        size_changes = np.zeros(self.getNServers(), dtype=int)
        for server in range(self.getNServers()):
            size_changes[server] = self.apply_event(events[server], event_times[server], server=server)

        return size_changes

    def apply_event(self, event: Event, event_time: float, server=0):
        """
        Applies to the given server the event of the given type taking place at the given time.
        
        Arguments:
        event: Event
            Type of the event to apply. Either Event.BIRTH or Event.DEATH.
        
        event_time: positive float
            Time of the event to apply.
        
        server: (opt) non-negative int
            Index indicating the server on which the event is applied.
            default: 0

        Return: int
            Change in queue size on the given server after the event has been applied.
        """
        # Note: we cannot check the instance of event because it is of type Enum or similar but not of type Event 
#        if not isinstance(event, Event):
#            warnings.warn("The event to apply is not of type `Event` ({}).\nNo event applied".format(type(Event)))
#            return 0
        if event_time < 0:
            warnings.warn("The event time is negative.\nNo event applied")
            return 0

        size_prev = self.size[server]
        if event == Event.BIRTH:
            super().add(server)
        elif event == Event.DEATH:
            super().remove(server)
        self.types_last_event[server] = event
        self.times_last_event[server] = event_time + self.origin

        return self.size[server] - size_prev

    def generate_next_event(self):
        """ (2021/10/18)
        Generates the next event of the system, i.e. the time, the event type, and the server where the event takes place

        NOTE: This method uses the Markovian property (or equivalently the memory-less property of the exponential
        distribution) to generate the events!
        That is, it computes the TOTAL event rate of the whole queue system
        (i.e. the sum of the job arrival rate + service rate over ALL servers) and generates
        a value drawn from an exponential distribution with that rate
        (this is in fact the distribution of the min( {T(1), T(2), ..., T(K)} ), where T(k) are ALL the possible
        event times in the system).
        It then chooses the event to apply with probability proportional to the event rate of each possible event.

        Return: tuple
        Tuple with two elements:
        - the time of the generated event (positive float)
        - the type of the generated event (of Enum class Event)
        - the server at which the generated event takes place (int)
        """

        #-- Generate the next event time
        # Note that we only consider the event rates of the events that are possible
        # i.e. birth if buffer size < K and death if the server has at least one job in its queue

        # Boolean indices defining the self.rates that can be used to pick the next event
        is_valid_rate = np.c_[
                                [True]*self.getNServers() if self.getBufferSize() < self.getCapacity() else [False]*self.getNServers(),
                                [True if self.getServerSize(s) > 0 else False for s, d in enumerate(self.getDeathRates())]
                                ]
        # Set the invalid rates to NaN so that they are not picked by the algorithm as valid events
        valid_rates = copy.deepcopy(self.rates)
        valid_rates[~is_valid_rate] = np.nan
        valid_rates_flat = valid_rates.flatten()

        event_rate = np.nansum(valid_rates)
        event_time = self._generate_event_times_at_rate(event_rate, size=1)

        #-- Probability of selection of each valid event in each server and indices of the flatten self.rates to choose from
        probs = valid_rates[is_valid_rate].flatten() / event_rate
        indices_to_choose_from = [idx for idx, r in enumerate(valid_rates_flat) if not np.isnan(r)]

        #-- Assign the event type and server on which the event takes place as described in the method's documentation
        # Linear index indicating server and event type (recall that dim1 in self.rates is server, dim2 is event type)
        idx_event = np.random.choice(indices_to_choose_from, size=1, p=probs)  # Note: np.flatten() covers every row first (i.e. C style by default)
        # Extract the server and event type from the linear index just chosen
        # Below, `2` is the number of possible event types: birth and death
        event_server = int( idx_event / 2 )
        event_type = Event( np.mod(idx_event, 2) )

        return event_time, event_type, event_server

    def _generate_event_times_at_rate(self, rate: float, size=1):
        "Generates event times at the given rate, as many as the `size` parameter"
        if rate < 0:
            warnings.warn("The event rate must be non-negative ({}). np.nan is returned.".format(rate))
            return np.nan

        # Note: the parameter of the exponential (`scale`) is 1/lambda, i.e. the inverse of the event rate
        if size == 1:
            # Return a scalar (it would be an array if we use parameter `size=1` in np.random.exponential()!!)
            return np.random.exponential(1/rate)
        else:
            # Return an array
            return np.random.exponential(1/rate, size=size)

    def generate_event_time(self, event, server=0):
        """
        Generates an event time of the given event type for the given server
        
        Arguments:
        events: Event
            Event type to generate for the given server.
        
        Return: non-negative float
            Generated event time.
        """
        return self._generate_event_times_at_rate( self.rates[server][event.value] )

    def generate_event_times(self, events):
        """
        Generates relative (to the previous event) event times for the given event types
        defined for every server in the queue.
        
        Arguments:
        events: Event or list of Event
            Event type to generate or list of event types to generate for each server,
            i.e. a list with the same length as the number of servers in the queue.
        
        Return: array
            Array with the generated event times
        """
        event_times = np.zeros(self.getNServers(), dtype=float)
        if isinstance(events, Event):
            # Convert the events variable to a list with all the same event type
            events = [events]*self.getNServers()
        for server in range(self.getNServers()):
            event_times[server] = self._generate_event_times_at_rate( self.rates[server][events[server].value] )
        return event_times

    def generate_birth_time(self, server=0):
        return self._generate_event_times_at_rate( self.rates[server][Event.BIRTH.value] )

    def generate_death_time(self, server=0):
        return self._generate_event_times_at_rate( self.rates[server][Event.DEATH.value] )

    def resize(self, size: np.ndarray, t: float):
        """"
        Sets the queue size of every server in the queue to the given values
        assigning the given times 't' as the times of last event for the respective server.
        The type of each last event is set to Event.RESIZE.

        Arguments:
        size: list or array of non-negative int
            Sizes to set to each server queue.

        t: non-negative float
            Time at which the resize takes place.
        """
        if t < 0:
            warnings.warn("Invalid time given (t >= 0) (size={}, t={})." \
                          "\nNo resize done." \
                          .format(size, t))
            return

        super().resize(size)
        self.types_last_event = np.repeat(Event.RESIZE, self.getNServers())
        self.times_last_event = np.repeat(t + self.origin, self.getNServers())

    def getTypesLastEvents(self):
        return copy.deepcopy(self.types_last_event)

    def getTimesLastEvents(self):
        return copy.deepcopy(self.times_last_event - self.origin)

    def getTypeLastEvent(self, server=0):
        return self.types_last_event[server]

    def getTimeLastEvent(self, server=0):
        return self.times_last_event[server] - self.origin

    def getMostRecentEventInfo(self):
        """
        Returns the array of last change in size experienced by the system,
        together with the server, event type, and event time associated to the latest event
        taking place in the system (i.e. the event --among all the latest events taking place
        in the servers-- with the largest time).
        
        Note that we return the last change for ALL servers, because this is needed
        for processing purposes (in e.g. estimators.py) when the last change was due to
        a RESIZE, because in such case, the size in the server selected here as the server
        having the event with largest time (which is any of them because a RESIZE happens
        for all servers at once, i.e. at the same time), may NOT have changed at all,
        although all the other servers may have changed their sizes...;
        and this situation will not be caught if we only returned the last change in size
        of the server with the largest time identified by this function.
        """
        server_with_largest_time = np.argmax( self.times_last_event )
        return  self.last_change, \
                server_with_largest_time, \
                self.types_last_event[server_with_largest_time], \
                self.times_last_event[server_with_largest_time] - self.origin

    def getMostRecentEventTime(self):
        "Returns the time of the latest event"
        return np.max(self.times_last_event) - self.origin

    def getBirthRate(self, server=0):
        return self.rates[server][Event.BIRTH.value]

    def getDeathRate(self, server=0):
        return self.rates[server][Event.DEATH.value]

    def getRatesForServer(self, server=0):
        "Returns a deep copy of the birth and death rates of the given server"
        return copy.deepcopy(self.rates[server])

    def getRates(self):
        """
        Returns the birth and death rates for all servers

        Return: numpy array
        The numpy array has dimension #servers x 2, where the first column corresponds to the BIRTH event
        and the second column corresponds to the DEATH event.
        """
        return copy.deepcopy(self.rates)

    def getBirthRates(self):
        "Returns the birth rates of the servers as a numpy array"
        return self.rates[:, Event.BIRTH.value]
        # (2021/10/18) Earlier version, when the rates were returned as a list --> CAN DELETE WHEN NEW VERSION IS TESTED
        #rates = []
        #for server in range(self.getNServers()):
        #    rates += [self.rates[server][Event.DEATH.value]]
        #return rates

    def getDeathRates(self):
        "Returns the death rates of the servers as a numpy array"
        return self.rates[:, Event.DEATH.value]
        # (2021/10/18) Earlier version, when the rates were returned as a list --> CAN DELETE WHEN NEW VERSION IS TESTED
        #rates = []
        #for server in range(self.getNServers()):
        #    rates += [self.rates[server][Event.DEATH.value]]
        #return rates

    def setBirthRates(self, rates):
        if len(rates) != self.getNServers():
            raise ValueError("The number of rates given ({}) must coincide with the number of servers ({})".format(len(rates), self.nservers))
        self.rates[:, Event.BIRTH.value] = rates
        # (2021/10/18) Earlier version that doesn't exploit the fact that self.rates is an array --> CAN DELETE WHEN NEW VERSION IS TESTED
        #for server in range(self.getNServers()):
        #    self.rates[server][Event.BIRTH.value] = rates[server]

    def setDeathRates(self, rates):
        if len(rates) != self.getNServers():
            raise ValueError("The number of rates given ({}) must coincide with the number of servers ({})".format(len(rates), self.nservers))
        self.rates[:, Event.DEATH.value] = rates
        # (2021/10/18) Earlier version that doesn't exploit the fact that self.rates is an array --> CAN DELETE WHEN NEW VERSION IS TESTED
        #for server in range(self.getNServers()):
        #    self.rates[server][Event.DEATH.value] = rates[server]


if __name__ == "__main__":
    # ------------------------- Unit tests -----------------------------#
    # --- Test #1: Generate events using the Markovian approach
    capacity = 6
    nservers = 3
    queue = QueueMM([0.7, 0.4, 0.8], [1.2, 0.9, 1.4], nservers, capacity)

    # Test 1A) Number of generated events per type and server on a non-border situation
    print("\nRunning test #1A: #events generated on an M/M/c/K queue on a non-border situation...")
    start_state = [1, 1, 1]
    queue.reset(start_state)
    N = 1000
    # Expected proportions of events by server and type of event
    p = ( queue.getRates().flatten() / np.sum(queue.getRates().flatten()) ).reshape(nservers, 2)
    assert np.isclose(np.sum(p), 1.0)

    # Run a number of event generation steps on the same start state
    nevents_by_server_and_type = np.zeros((queue.getNServers(), 2))
    for i in range(N):
        time, event, server = queue.generate_next_event()
        queue.reset(start_state)
        nevents_by_server_and_type[server, event.value] += 1
    # Observed proportions of events by server and type of event
    phat = 1.0 * nevents_by_server_and_type / N
    se_phat = np.sqrt(p * (1 - p) / N)
    print("EXPECTED proportions of events by server and type (server x event-types):\n{}".format(p))
    print("Observed proportions of events by server and type on N={} generated events (server x event-types):\n{}".format(N, phat))
    print("Standard Errors on N={} generated events (server x event-types):\n{}".format(N, se_phat))
    assert np.allclose(phat, p, atol=3*se_phat)   # true probabilities should be contained in +/- 3 SE(phat) from phat

    # Test 1B) Number of generated events per type and server on a border situation (some servers with no jobs)
    print("\nRunning test #1B: #events generated on an M/M/c/K queue on a border situation (some servers with no jobs)...")
    start_state = [0, 0, 1]
    queue.reset(start_state)
    N = 100
    # Expected proportions of events by server and type of event
    # Note that the servers with 0 jobs in the queue have zero probability of being selected
    p = ( queue.getRates().flatten() / np.sum(queue.getRates().flatten()[[0, 2, 4, 5]]) ).reshape(nservers, 2)
    p[0, 1] = 0.0
    p[1, 1] = 0.0
    assert np.isclose(np.sum(p), 1.0)

    # Run a number of event generation steps on the same start state
    nevents_by_server_and_type = np.zeros((queue.getNServers(), 2))
    for i in range(N):
        time, event, server = queue.generate_next_event()
        queue.reset(start_state)
        nevents_by_server_and_type[server, event.value] += 1
    # Observed proportions of events by server and type of event
    phat = 1.0 * nevents_by_server_and_type / N
    se_phat = np.sqrt(p * (1 - p) / N)
    print("EXPECTED proportions of events by server and type (server x event-types):\n{}".format(p))
    print("Observed proportions of events by server and type on N={} generated events (server x event-types):\n{}".format(N, phat))
    print("Standard Errors on N={} generated events (server x event-types):\n{}".format(N, se_phat))
    assert np.allclose(phat, p, atol=3*se_phat)   # true probabilities should be contained in +/- 3 SE(phat) from phat

    # Test 1C) Number of generated events per type and server on a border situation (some servers with no jobs and buffer size at full capacity)
    print("\nRunning test #1C: #events generated on an M/M/c/K queue on a border situation (some servers with no jobs and buffer size at full capacity)...")
    start_state = [2, capacity-2, 0]
    queue.reset(start_state)
    N = 100
    # Expected proportions of events by server and type of event
    # Note that the servers with 0 jobs in the queue have zero probability of being selected as well as all are birth
    # events, since the buffer is at full capacity.
    p = ( queue.getRates().flatten() / np.sum(queue.getRates().flatten()[[1, 3]]) ).reshape(nservers, 2)
    p[0, 0] = 0.0
    p[1, 0] = 0.0
    p[2, 0] = 0.0
    p[2, 1] = 0.0
    assert np.isclose(np.sum(p), 1.0)

    # Run a number of event generation steps on the same start state
    nevents_by_server_and_type = np.zeros((queue.getNServers(), 2))
    for i in range(N):
        time, event, server = queue.generate_next_event()
        queue.reset(start_state)
        nevents_by_server_and_type[server, event.value] += 1
    # Observed proportions of events by server and type of event
    phat = 1.0 * nevents_by_server_and_type / N
    se_phat = np.sqrt(p * (1 - p) / N)
    print("EXPECTED proportions of events by server and type (server x event-types):\n{}".format(p))
    print("Observed proportions of events by server and type on N={} generated events (server x event-types):\n{}".format(N, phat))
    print("Standard Errors on N={} generated events (server x event-types):\n{}".format(N, se_phat))
    assert np.allclose(phat, p, atol=3*se_phat)   # true probabilities should be contained in +/- 3 SE(phat) from phat
