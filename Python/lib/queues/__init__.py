# -*- coding: utf-8 -*-
"""
Created on Thu Jun  6 19:39:44 2020

@author: Daniel Mastropietro
@description: Definition of classes defining queues.
"""

from enum import Enum, unique

import sys
import warnings
import numpy as np
import copy

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
    capacity: positive int
        Capacity of the buffer that collects all the arriving jobs before being assigned to a server.
    
    nservers: (opt) positive int
        Number of servers to process jobs in the system.
        default: 1

    size: (opt) non-negative int, list or array
        Size of the queue of each server.
        default: 0

    log: (opt) bool
        Whether to activate logging messages about operations run on the object.
        default: False
    """

    def __init__(self, capacity, nservers=1, size=0, log=False):
        self.K = capacity           # Capacity of the buffer that receives the jobs before assigning them to a server
        self.c = nservers

        self.setServerSizes(size)
        self.size_initial = copy.deepcopy(self.size)
        self.LOG = log

        self.reset()

    def reset(self):
        self.size = self.size_initial
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
    def setServerSizes(self, size):
        "Sets the size of the queue in EACH server"
        # Convert the size to an array (and to int if given as float)
        if isinstance(size, (int, np.int32, np.int64, float, np.float32, np.float64)):
            size = np.repeat(int(size), self.c)
        self.size = np.array(size)


class QueueMM(GenericQueue):

    def __init__(self, rates_birth, rates_death, nservers :int, capacity :int):
        super().__init__(capacity, nservers=nservers, size=0)
        is_rates_birth_scalar = not isinstance(rates_birth, (list, tuple, np.ndarray))
        is_rates_death_scalar = not isinstance(rates_death, (list, tuple, np.ndarray))
        if  not is_rates_birth_scalar and len(rates_birth) != nservers or \
            not is_rates_death_scalar and len(rates_death) != nservers:
            raise ValueError("Both birth and death rates must be either just one value " \
                             "({} birth rates given, {} death rates given) or be as many as the number of servers ({})." \
                             "\nThe process stops." \
                             .format(len(rates_birth), len(rates_death), nservers))
            sys.exit(-1)
        if  is_rates_birth_scalar and rates_birth < 0 or \
            is_rates_death_scalar and rates_death < 0 or \
            not is_rates_birth_scalar and any(np.array(rates_birth) < 0) or \
            not is_rates_death_scalar and any(np.array(rates_death) < 0):
            raise ValueError("Both birth and death rates must be non-negative (birth rates: {}, death rates: {})." \
                             "\nThe process stops." \
                             .format(rates_birth, rates_death))
            sys.exit(-1)
        # The rates are stored in a 2D array (generated by np.repeat())
        self.rates = np.repeat([[0.0]*2], nservers, axis=0)  # 2 is the number of rates: birth and death
        if is_rates_birth_scalar:
            rates_birth = np.repeat(rates_birth, nservers)
        if is_rates_death_scalar:
            rates_death = np.repeat(rates_death, nservers)
        for server in range(nservers):
            self.rates[server][Event.BIRTH.value] = rates_birth[server]
            self.rates[server][Event.DEATH.value] = rates_death[server]

        self.reset()

    def reset(self):
        # We need to call the SUPER reset() method because the `self.reset()` call in the super class
        # calls THIS reset() method, NOT the SUPER method!!
        super().reset()
        # Time and type of the last event in EACH server
        # They are stored as ARRAYS instead of lists in order to facilitate operations on them during simulations.
        self.time_last_event = np.zeros(self.getNServers(), dtype=float)
        self.type_last_event = np.repeat(Event.RESET, self.getNServers())

    def apply_events(self, events :list, event_times :list):
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

    def apply_event(self, event :Event, event_time :float, server=0):
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
        self.type_last_event[server] = event
        self.time_last_event[server] = event_time

        return self.size[server] - size_prev

    def _generate_event_times_at_rate(self, rate :float, size=1):
        # Note: the parameter of the exponential (`scale`) is 1/lambda, i.e. the inverse of the event rate
        if rate < 0:
            warnings.warn("The event rate must be non-negative ({}). np.nan is returned.".format(rate))
            return np.nan
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
        Generates event times for the given event types defined for every server in the queue
        
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

    def resize(self, size :np.ndarray, t :float):
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
        self.time_last_event = np.repeat(t, self.getNServers())
        self.type_last_event = np.repeat(Event.RESIZE, self.getNServers())

    def getTypesLastEvents(self):
        return copy.deepcopy(self.type_last_event)

    def getTimesLastEvents(self):
        return copy.deepcopy(self.time_last_event)

    def getTypeLastEvent(self, server=0):
        return self.type_last_event[server]

    def getTimeLastEvent(self, server=0):
        return self.time_last_event[server]

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
        server_with_largest_time = np.argmax( self.time_last_event )
        return  self.last_change, \
                server_with_largest_time, \
                self.type_last_event[server_with_largest_time], \
                self.time_last_event[server_with_largest_time]

    def getMostRecentEventTime(self):
        "Returns the time of the latest event"
        return np.max(self.time_last_event)
    
    def getBirthRate(self, server=0):
        return self.rates[server][Event.BIRTH.value]

    def getDeathRate(self, server=0):
        return self.rates[server][Event.DEATH.value]

    def getRates(self, server=0):
        return copy.deepcopy(self.rates[server])

    def getBirthRates(self):
        rates = []
        for server in range(self.getNServers()):
            rates += [self.rates[server][Event.BIRTH.value]]
        return rates

    def getDeathRates(self):
        rates = []
        for server in range(self.getNServers()):
            rates += [self.rates[server][Event.DEATH.value]]
        return rates

    def setBirthRates(self, rates):
        if len(rates) != self.getNServers():
            raise ValueError("The number of rates given ({}) must coincide with the number of servers ({})".format(len(rates), self.nservers))
        for server in range(self.getNServers()):
            self.rates[server][Event.BIRTH.value] = rates[server]

    def setDeathRates(self, rates):
        if len(rates) != self.getNServers():
            raise ValueError("The number of rates given ({}) must coincide with the number of servers ({})".format(len(rates), self.nservers))
        for server in range(self.getNServers()):
            self.rates[server][Event.DEATH.value] = rates[server]
