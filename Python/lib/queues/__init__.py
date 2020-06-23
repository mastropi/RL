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
from bokeh.core.properties import Size

@unique
class Event(Enum):
    # The values of each element representing normal events (e.g. birth, death)
    # should be a valid INDEX for a list storing event types, starting at 0
    BIRTH = 0
    DEATH = 1
    RESIZE = 9
    RESET = 99

class GenericQueue:
    "Class that holds information that is present in all queues"

    def __init__(self, capacity, nservers=1, size=0, log=False):
        self.K = capacity
        self.c = nservers
        self.size_initial = size
        self.size = size
        self.LOG = log

        self.reset()

    def reset(self):
        self.size = self.size_initial

    def add(self):
        if not self.isFull():
            self.size += 1
        else:
            if self.LOG:
                print("Queue is full (size={}): no job added".format(self.size))
 
    def remove(self):
        if not self.isEmpty():
            self.size -= 1
        else:
            if self.LOG:
                print("Queue is empty (size={}): no job removed".format(self.size))

    def isEmpty(self):
        return self.size == 0

    def isFull(self):
        return self.size == self.K

    # GETTERS
    def getNServers(self):
        return self.c

    def getCapacity(self):
        return self.K


class QueueMM(GenericQueue):

    def __init__(self, rate_birth, rate_death, nservers, capacity):
        super().__init__(capacity, nservers=nservers, size=0)
        if rate_birth < 0 or rate_death < 0:
            raise ValueError("Both birth and death rates must be non-negative ({}, {})." \
                             "\nThe process stops." \
                             .format(rate_birth, rate_death))
            sys.exit(-1)
        self.rates = [0]* 2
        self.rates[Event.BIRTH.value] = rate_birth
        self.rates[Event.DEATH.value] = rate_death
        
        self.reset()

    def reset(self):
        self.time_last_event = 0.0
        self.type_last_event = Event.RESET

    def apply_event(self, event, event_time):
        """
        Applies the event of the given type taking place at the given time
        and returns the change in queue size, which can be either -1, 0 or +1.
        
        Arguments:
        event: Event
            Type of the event to apply. Either Event.BIRTH or Event.DEATH.
        
        event_time: positive float
            Time of the event to apply.
        """
        assert isinstance(event, Event), "The event is of type `Event`"

        size_prev = self.size
        if event == Event.BIRTH:
            super().add()
        elif event == Event.DEATH:
            super().remove()
        self.type_last_event = event
        self.time_last_event = event_time

        return self.size - size_prev

    def generate_event_times(self, rate, size):
        # Note: the parameter of the exponential (`scale`) is 1/lambda, i.e. the inverse of the event rate
        if rate < 0:
            warnings.warn("The event rate must be non-negative ({}). np.nan is returned.".format(rate))
            return np.nan
        if size == 1:
            # Return a scalar (it would be an array if we use parameter `size=1`!!)
            return np.random.exponential(1/rate)
        else:
            # Return an array
            return np.random.exponential(1/rate, size=size)

    def generate_birth_time(self):
        return self.generate_event_times(self.rates[Event.BIRTH.value], size=1)

    def generate_death_time(self):
        return self.generate_event_times(self.rates[Event.DEATH.value], size=1)

    def resize(self, size, t):
        """"
        Sets the size of the queue to the given value and assigns the given time 't' as the time of the last event.
        The type of the last event is set to Event.RESIZE
        """
        if not (0 <= size <= self.getCapacity() and t >= 0):
            warnings.warn("Invalid size or time given (0 <= size <= {}; t >= 0) (size={}, t={})." \
                          "\nNo resize done." \
                          .format(self.getCapacity(), size, t))
        else:
            self.size = size
            self.time_last_event = t
            self.type_last_event = Event.RESIZE

    def getTypeLastEvent(self):
        return self.type_last_event

    def getTimeLastEvent(self):
        return self.time_last_event

    def getBirthRate(self):
        return self.rates[Event.BIRTH.value]

    def getDeathRate(self):
        return self.rates[Event.DEATH.value]
