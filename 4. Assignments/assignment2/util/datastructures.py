import random 
from time import sleep
import re
from collections import namedtuple

import numpy as np
import matplotlib.pyplot as plt

from util import set_up_GBG, plot_pos, plot_Q, plot_average_r, clear_plot

Transition = namedtuple(typename='Transition', field_names=['source', 'event', 'target'])

Place = namedtuple('Place', ['label', 'marking'])

Arc = namedtuple('Arc', ['source', 'target', 'weight'])

Edge = namedtuple('Edge', ['source', 'target', 'label'])

DiGraph = namedtuple(typename='DiGraph', field_names=['nodes', 'init', 'edges'])


class Automaton(object):

    def __init__(self, states, init, events, trans, marked=None, forbidden=None):
        """
        This is the constructor of the automaton.

        At creation, the automaton gets the following attributes assigned:
        :param states: A set of states
        :param init: The initial state
        :param events: A set of events
        :param trans: A set of transitions
        :param marked: (Optional) A set of marked states
        :param forbidden: (Optional) A set of forbidden states
        """
        self.states = states
        self.init = init
        self.events = events
        self.trans = trans
        self.marked = marked if marked else set()
        self.forbidden = forbidden if forbidden else set()

    def __str__(self):
        """Prints the automaton in a pretty way."""
        return 'states: \n\t{}\n' \
               'init: \n\t{}\n' \
               'events: \n\t{}\n' \
               'transitions: \n\t{}\n' \
               'marked: \n\t{}\n' \
               'forbidden: \n\t{}\n'.format(
                   self.states, self.init, self.events,
                   '\n\t'.join([str(t) for t in self.trans]), self.marked, self.forbidden)

    def __setattr__(self, name, value):
        """Validates and protects the attributes of the automaton"""
        if name in ('states', 'events'):
            value = frozenset(self._validate_set(value))
        elif name == 'init':
            value = self._validate_init(value)
        elif name == 'trans':
            value = frozenset(self._validate_transitions(value))
        elif name in ('marked', 'forbidden'):
            value = frozenset(self._validate_subset(value))
        super(Automaton, self).__setattr__(name, value)

    def __getattribute__(self, name):
        """Returns a regular set of the accessed attribute"""
        if name in ('states', 'events', 'trans', 'marked', 'forbidden'):
            return set(super(Automaton, self).__getattribute__(name))
        else:
            return super(Automaton, self).__getattribute__(name)

    def __eq__(self, other):
        """Checks if two Automata are the same"""
        if type(other) is type(self):
            return self.__dict__ == other.__dict__
        return False

    @staticmethod
    def _validate_set(states):
        """Checks that states is a set and the states in it are strings or integers"""
        assert isinstance(states, set)
        for state in states:
            assert isinstance(state, str) or isinstance(
                state, int), 'A state must be either of type string or integer!'
        return states

    def _validate_subset(self, subset):
        """Validates the set and checks whether the states in the subset are part of the state set"""
        subset = self._validate_set(subset)
        assert subset.issubset(
            self.states), 'Marked and forbidden states must be subsets of all states!'
        return subset

    def _validate_init(self, state):
        """Checks whether the state is part of the state set"""
        assert isinstance(state, str) or isinstance(
            state, int), 'The initial state must be of type string or integer!'
        assert state in self.states, 'The initial state must be member of states!'
        return state

    def _validate_transitions(self, transitions):
        """Checks that all transition elements are part in the respective sets (states, events)"""
        assert isinstance(transitions, set)
        for transition in transitions:
            assert isinstance(transition, Transition)
            assert transition.source in self.states
            assert transition.event in self.events
            assert transition.target in self.states
        return transitions


class PetriNet(object):

    def __init__(self, places, transitions, arcs):
        """
        This is the constructor of the PetriNet.

        At creation, the PetriNet gets the following attributes assigned:
        :param places: A set of Places
        :param transitions: A set of Transitions
        :param arcs: A set of Arcs
        """
        assert isinstance(places, list)
        self.places = places
        assert isinstance(arcs, set)
        self.arcs = arcs
        assert isinstance(transitions, set)
        self.transitions = transitions
        
        self.P = np.array([p.label for p in places], str)
        self.init_marking = np.array([p.marking for p in places], int)
        self.T = np.array(list(transitions))
        self.A_minus = np.zeros((len(places), len(transitions)), int)
        self.A_plus = np.zeros((len(places), len(transitions)), int)
        # populate transition matrices
        for a in arcs:
            if a.source in self.P:
                n, = np.where(self.P == a.source)
                m, = np.where(self.T == a.target)
                self.A_minus[n, m] = a.weight
            else:
                n, = np.where(self.P == a.target)
                m, = np.where(self.T == a.source)
                self.A_plus[n, m] = a.weight

    def make_reachability_graph(self):
        """Computes the reachability graph of the PetriNet"""        
        new_markings = [self.init_marking]
        nodes = {array_str(self.init_marking)}
        edges = set()

        while new_markings:
            m = new_markings.pop()
            m = np.reshape(m, [-1, 1])
            for i, t in enumerate(self.T):
                s = np.zeros([len(self.T), 1], int)
                s[i] = 1
                if np.all(m >= np.matmul(self.A_minus, s)):
                    m_plus = m + np.matmul(self.A_plus - self.A_minus, s)
                    m_plus_str = array_str(m_plus)
                    edges.add(Edge(array_str(m), m_plus_str, t))
                    if m_plus_str not in nodes:
                        nodes.add(m_plus_str)
                        new_markings.append(m_plus)
        return DiGraph(nodes, array_str(self.init_marking), edges)
    

def array_str(a):
    """Casts numpy array to string and removes superfluous whitespaces"""
    return re.sub(" +", " ", str(a.flatten())).replace('[ ', '[')


class WindyGothenburg:

    def __init__(self, name, w=12, h=12, obstacles=set(), water=set(), pripps_reward=None, test=False):
        """The constructor of the grid world WindyGothenburg"""
        self.name = name
        self.w = w
        self.h = h
        self.states = {(i, j) for i in range(self.w + 1) for j in range(self.h + 1)} - obstacles
        self.actions = {'north', 'east', 'south', 'west'}
        
        self._jarntorget = (0, self.h)
        self._home = (self.w, self.h)
        self._gota_alv = water
        if pripps_reward is None:
            self._chalmers = None
            self._pripps_reward = 0
        else:
            self._chalmers = (int(self.w/4), int(self.h/6))
            assert 0.001 <= pripps_reward <= 10
            self._pripps_reward = pripps_reward
        self._current_state = self._jarntorget    
        
        # for rendering
        self._test = test
        if not self._test:
            self.fig, self.gw, self.sp = set_up_GBG(self.w, self.h, self._jarntorget, 
                                                    self._chalmers, self._home,
                                                    obstacles, self._gota_alv)
            self.plot_elems = ()
            self.last_pos = None
        
    def _get_next_state(self, state, action):
        """The transition function"""
        i, j = state
        i = i + 1 if random.random() < 0.05 else i # wind blows you to the east
        j = j + 1 if random.random() < 0.1 else j # wind blows you to the north
        if action == 'north':
            j += 1
        elif action == 'east':
            i += 1
        elif action == 'south':
            j -= 1
        elif action == 'west':
            i -= 1
        next_state = (i, j) if (i, j) in self.states else state
        return next_state

    def step(self, action):
        """
        Advances the simulation by one step
        
        :param action: the control action to apply
        """
        assert action in self.actions
        
        next_state = self._get_next_state(self._current_state, action)
        self._current_state = next_state
        
        if next_state == self._home:
            reward = 100
            done = True
        elif next_state in self._gota_alv:
            reward = -10
            done = True
        elif next_state == self._chalmers:
            reward = random.choice([0, self._pripps_reward])
            done = False
        else:
            reward = 0
            done = False
            
        return next_state, reward, done
    
    def reset(self):
        """Must be called if done == True"""
        self._current_state = self._jarntorget
        return self._current_state
    
    def render(self, Q=None, average_reward=None, avg_r_smoothed=None, episode=None):
        """Renders the environment"""
        assert not self._test, 'Disable test mode for rendering'
        if self.last_pos:
            self.last_pos.remove()
        if self.plot_elems:
            clear_plot(*self.plot_elems)
        self.last_pos = plot_pos(self.gw, self._current_state)
        if Q:
            # Plots the contours of max_u Q(*|x) for each x
            # Plots also the direction of argmax_u Q(*|x) for each x
            self.plot_elems = plot_Q(self.gw, Q, self.w, self.h) 
        if average_reward is not None:
            # Tracking learning over episodes
            assert episode is not None, 'Provide the number of the current episode.'
            plot_average_r(self.sp, average_reward, avg_r_smoothed, episode)
        self.fig.canvas.draw()
        display(self.fig)
        sleep(0.001) # So you can enjoy the pretty plot
    
    def close(self):
        """Closes the visualization"""
        if not self._test:
            plt.close(self.fig)