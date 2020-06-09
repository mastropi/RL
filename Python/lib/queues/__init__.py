# -*- coding: utf-8 -*-
"""
Created on Thu Jun  6 19:39:44 2020

@author: Daniel Mastropietro
@description: Definition of classes defining queues.
"""

import numpy as np

BIRTH = 0
DEATH = 1

class GenericQueue:
    "Class that holds information that is present in all queues"

    def __init__(self, capacity, nservers=1, size=0):
        self.K = capacity
        self.c = nservers
        self.size_initial = size
        self.size = size

        self.reset()

    def reset(self):
        self.size = self.size_initial

    def add(self):
        if not self.isFull():
            self.size += 1
        else:
            print("Queue is full (size={}): no job added".format(self.size))
 
    def remove(self):
        if not self.isEmpty():
            self.size -= 1
        else:
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
        self.rates = [0]* 2
        self.rates[BIRTH] = rate_birth
        self.rates[DEATH] = rate_death
        
        self.time_last_event = 0.0
        self.type_last_event = None

    def reset(self):
        self.time_last_event = 0.0
        self.type_last_event = None

    def apply_event(self, event_type, event_time):
        """
        Applies the event of the given type taking place at the given time
        and returns the change in queue size, which can be either -1, 0 or +1."""
        size_prev = self.size
        if event_type == BIRTH:
            super().add()
        elif event_type == DEATH:
            super().remove()
        self.type_last_event = event_type
        self.time_last_event = event_time

        return self.size - size_prev

    def generate_event_times(self, rate, size):
        if size == 1:
            # Return a scalar (it would be an array if we use parameter `size=1`!!)
            return np.random.exponential(rate)
        else:
            # Return an array
            return np.random.exponential(rate, size=size)

    def generate_next_birth_time(self):
        return self.generate_event_times(self.rates[BIRTH], size=1)

    def generate_next_death_time(self):
        return self.generate_event_times(self.rates[DEATH], size=1)

    def getTypeLastEvent(self):
        return self.type_last_event

    def getTimeLastEvent(self):
        return self.time_last_event

    def getBirthRate(self):
        return self.rates[BIRTH]

    def getDeathRate(self):
        return self.rates[DEATH]
