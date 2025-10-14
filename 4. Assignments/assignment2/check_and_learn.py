# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.17.3
#   kernelspec:
#     display_name: .venv
#     language: python
#     name: python3
# ---

# %% [markdown] deletable=false editable=false nbgrader={"cell_type": "markdown", "checksum": "042c693d37df41bc35d6a7c91def6e85", "grade": false, "grade_id": "cell-6ec5aa4507a8f918", "locked": true, "schema_version": 3, "solution": false}
# <center>
#
# # Logic, Learning, and Decision
#
# ## Home Assignment 2
#
# ### Model Checking with $\mu%$-Calculus & Controlling through Q-Learning
# </center>
#
# - - -

# %% [markdown] deletable=false editable=false nbgrader={"cell_type": "markdown", "checksum": "0475f99dec2e5397c3eed304242f0108", "grade": false, "grade_id": "cell-d299ce342e54a4d8", "locked": true, "schema_version": 3, "solution": false}
# # Introduction
#
# The first objective of this home assignment is to deepen the understanding of temporal logic specifications and model checking algorithms based on $\mu$-calculus. You will achieve that through implementing a fixed-point algorithm for a particular _CTL*_ specification and test it out on a booking problem of variable size. 
#
# The second objective is to obtain a basic understanding of a central Reinforcement Learning algorithm called *Q-learning*.
#
# This home assignment is performed in *two member groups*. Write all your answers into this notebook and **submit only this notebook (.ipynb) containing your team's own original work on Canvas.**

# %% [markdown] deletable=false editable=false nbgrader={"cell_type": "markdown", "checksum": "ac0abd938ff82f6c25852940e9da18c3", "grade": false, "grade_id": "cell-8dfe6514b5b0e564", "locked": true, "schema_version": 3, "solution": false}
# ## Comments and Recommendations
# As always:
# * The following resources will be of great help to you for this assignment:
#   * Lecture Notes
#   * [Python docs](https://docs.python.org/3/)
#   * [Google](https://www.google.com)
# * This assignment is written for Python 3.5 or later!
# * We will test your code with additional edge cases. So convince yourself that everything is correct before you submit.
# * This assignment makes use of the Python packages [numpy](https://docs.scipy.org/doc/numpy/) and [matplotlib](https://matplotlib.org/index.html). Make sure to have it installed.

# %% deletable=false editable=false nbgrader={"cell_type": "code", "checksum": "fc04b29c5d48f53f509114009f833954", "grade": false, "grade_id": "cell-d0ce19473bf6c277", "locked": true, "schema_version": 3, "solution": false}
try:
    import numpy as np
except ImportError:
    print("You need to install numpy! Open a command prompt and run 'pip install numpy'")

try:
    import matplotlib.pyplot as plt
except ImportError:
    print("You need to install matplotlib! Open a command prompt and run 'pip install matplotlib'")

import platform
if platform.system() == 'Windows':
    try:
        import msvc_runtime
    except ImportError:
        print("You may need to install msvc-runtime for multiprocessing! " + 
        "If you have issues with the function `eval_hyperparam_grid`, open " + 
        "a command prompt and run 'pip install msvc-runtime'. Then restart the kernel.")

# %% [markdown] deletable=false editable=false nbgrader={"cell_type": "markdown", "checksum": "d11fe46a5b8dc42a36e20f63dabc85b7", "grade": false, "grade_id": "cell-c55d42dab3dd9445", "locked": true, "schema_version": 3, "solution": false}
# ---

# %% [markdown] deletable=false editable=false nbgrader={"cell_type": "markdown", "checksum": "65d5efc21901317a346d6893f17cc7ca", "grade": false, "grade_id": "cell-950323411a5504b7", "locked": true, "schema_version": 3, "solution": false, "task": false}
# # Part 1: Model Checking with $\mu-$Calculus
#
# Imagine you are employed by a company, but you are quite unhappy recently, because the system you are working with (for instance, a manufacturing cell, but could be anything else) exhibits some strange behavior and just freezes randomly. So, you decide to model the system (i.e. manufacturing cell, etc) and analyze it with the methods from your favorite course at university. Hence, you come up with the petri net $P_1$:
# ![petri net](fig/petri_net.png)
# Two parallel processes require two resources $R_1$ and $R_2$ for their operations. This is a classic booking problem. Although appearing simple, this system may exhibit undesirable behavior. For the majority of Part 1 of the assignment we will work with this system. 

# %% deletable=false editable=false nbgrader={"cell_type": "code", "checksum": "a220242a51351383ac07ca4ffe618065", "grade": false, "grade_id": "cell-cf7667e2af990fc3", "locked": true, "schema_version": 3, "solution": false, "task": false}
from util import PetriNet
from util import Place, Arc
from util import plot_petrinet


def make_petrinet(p11_tokens, p21_tokens, R1_tokens, R2_tokens):
    return PetriNet(places=[Place('p11', p11_tokens), Place('p12', 0),
                            Place('p21', p21_tokens), Place('p22', 0),
                            Place('R1', R1_tokens), Place('R2', R2_tokens)],
                    transitions={'a1', 'b1', 'a2', 'b2'},
                    arcs={Arc('p11', 'a1', 1),
                          Arc('a1', 'p12', 1),
                          Arc('p12', 'b1', 1),
                          Arc('b1', 'p11', 1),
                          Arc('R1', 'a1', 1),
                          Arc('R1', 'b2', 1),
                          Arc('b1', 'R1', 1),
                          Arc('b2', 'R1', 1),
                          Arc('R2', 'a2', 1),
                          Arc('R2', 'b1', 1),
                          Arc('b1', 'R2', 1),
                          Arc('b2', 'R2', 1),
                          Arc('p21', 'a2', 1),
                          Arc('a2', 'p22', 1),
                          Arc('p22', 'b2', 1),
                          Arc('b2', 'p21', 1)})


P_1 = make_petrinet(p11_tokens=3, p21_tokens=2, R1_tokens=2, R2_tokens=1)

# %% [markdown] deletable=false editable=false nbgrader={"cell_type": "markdown", "checksum": "b44b5200f463855f586716e75ca24f5c", "grade": false, "grade_id": "cell-2e491eff08188b4c", "locked": true, "schema_version": 3, "solution": false, "task": false}
# We have also implemented a function for you in the PetriNet class that generates the corresponding reachability graph.

# %% deletable=false editable=false nbgrader={"cell_type": "code", "checksum": "e5460c1ef947f9e4a89fa20a40c1471f", "grade": false, "grade_id": "cell-bfdddd0d867fad17", "locked": true, "schema_version": 3, "solution": false, "task": false}
from util import plot_digraph

print('P_1:')
plot_digraph(P_1.make_reachability_graph(), 'fig/P_1_reach_graph')


# %% [markdown] deletable=false editable=false nbgrader={"cell_type": "markdown", "checksum": "970cd86c0c8c32f64c53ca29beb166c1", "grade": false, "grade_id": "cell-f89f8e548997b4a9", "locked": true, "schema_version": 3, "solution": false, "task": false}
# To be able to use $\mu$-calculus to check the Petri net model, we need to transform it into a transition system, though. 

# %% [markdown] deletable=false editable=false nbgrader={"cell_type": "markdown", "checksum": "ed9f9ab2054716282c9a2595bf43d707", "grade": false, "grade_id": "cell-9b3b3c7c1016cf98", "locked": true, "schema_version": 3, "solution": false, "task": false}
# ---

# %% [markdown] deletable=false editable=false nbgrader={"cell_type": "markdown", "checksum": "a48f4a8a210ad9fbbb47d022b22424bd", "grade": false, "grade_id": "cell-3dfb2dc445d04d3b", "locked": true, "schema_version": 3, "solution": false, "task": false}
# ## Transition Systems
#
# In this assignment, we will work with a more general model for discrete event systems called __Transition System__. A transition system $G$ is defined by a 6-tuple $G = \langle X, \Sigma, T, I, AP, \lambda \rangle$ where $X$ is a set of states, $\Sigma$ is a finite set of events, $T \subseteq X \times T \times X$ is a transition relation, where a transition $t = (x, a, x') \in T$, includes the source state $x$, the event label $a$, and the target state $x'$, $I \subseteq X$ is a set of possible initial states, $AP$ is a set of atomic propositions, and $\lambda: X \mapsto 2^{AP}$ is a state labeling function. A transition system  $G$ without the state labels, where $AP$ and $\lambda$ are excluded from $G$ is obviously an automaton without marked and forbidden states.
#
# In order to implement a data structure corresponding to a transition system, we introduce a new class of _State_ objects.

# %% deletable=false editable=false nbgrader={"cell_type": "code", "checksum": "4905c5e7328664c5fcc02b90c3449ce7", "grade": false, "grade_id": "cell-380888be34de0178", "locked": true, "schema_version": 3, "solution": false, "task": false}
class State(object):
    
    def __init__(self, name, labels=None):
        """
        Constructor  of the state.
        
        :param name: String. Default atomic proposition of the state
        :param labels: Set of atomic propositions that a true in the state
        """
        self.name = name
        assert labels is None or type(labels) is set
        self.labels = {name} if not labels else labels | {name}
    
    def __str__(self):
        """Prints the state in a pretty way."""
        return 'name: {} & ' \
               'labels: {}'.format(self.name, self.labels)
        
    def is_satisfied(self, atomic_proposition):
        """Checks whether the atomic proposition is statisfied in the state."""
        return atomic_proposition in self.labels


# %% [markdown] deletable=false editable=false nbgrader={"cell_type": "markdown", "checksum": "d5c55fdaa6db01b133b10b6778f0d3db", "grade": false, "grade_id": "cell-9d2b45a323388413", "locked": true, "schema_version": 3, "solution": false, "task": false}
# That allows us to define the _TransitionSystem_ class:

# %% deletable=false editable=false nbgrader={"cell_type": "code", "checksum": "6ffa219bd6b989a023788e222eeacd93", "grade": false, "grade_id": "cell-244d255993c355a5", "locked": true, "schema_version": 3, "solution": false, "task": false}
from util import Transition


class TransitionSystem(object):

    def __init__(self, states, init, events, trans):
        """
        This is the constructor of the transition system.

        At creation, the automaton gets the following attributes assigned:
        :param states: A set of States
        :param init: A set of initial States
        :param events: A set of events
        :param trans: A set of transitions
        """
        assert isinstance(states, set)
        self.states = states
        assert isinstance(init, set)
        self.init = init
        assert isinstance(events, set)
        self.events = events
        assert isinstance(trans, set)
        self.trans = trans

    def __str__(self):
        """Prints the transition system in a pretty way."""
        states_str = '{\n\t' + ',\n\t'.join(
            [str(s) for s in self.states]) + '\n\t}'
        init_str = '{\n\t' + ', '.join([str(s.name) for s in self.init]) + '\n\t}'
        trans_str = '\n\t'.join(
            ['{} --{}--> {},'.format(t.source.name, t.event, t.target.name) for t in self.trans])
        trans_str = '{\n\t' + trans_str + '\n\t}'
        return 'states: \n\t{}\n' \
               'init: \n\t{}\n' \
               'events: \n\t{}\n' \
               'transitions: \n\t{}\n'.format(
                   states_str, init_str, self.events, trans_str)

    def __eq__(self, other):
        """Checks if two transition systems are the same"""
        if type(other) is type(self):
            return self.__dict__ == other.__dict__
        return False


# %% [markdown] deletable=false editable=false nbgrader={"cell_type": "markdown", "checksum": "0feffb62f1271e6c008f8daab84e22ef", "grade": false, "grade_id": "cell-74ff7f77c329d6a9", "locked": true, "schema_version": 3, "solution": false, "task": false}
# Now, we can translate our Petri net $P_1$ of the booking problem that we are working with into a *TransitionSystem* via its reachability graph.

# %% deletable=false editable=false nbgrader={"cell_type": "code", "checksum": "83a67aac459273877ae2da708e35577b", "grade": false, "grade_id": "cell-61168b606b0b8afd", "locked": true, "schema_version": 3, "solution": false, "task": false}
from util import plot_transitionsystem


def make_transition_system(petri_net):
    """Transforms a PetriNet to a TransitionSystem."""
    G = petri_net.make_reachability_graph()
    states = {State(n) for n in G.nodes}
    
    def get_state(node):
        for s in states:
            if s.name == node:
                return s
    
    init = {get_state(G.init)}
    events = petri_net.transitions
    trans = {Transition(get_state(e.source), e.label, get_state(e.target)) for e in G.edges}
    return TransitionSystem(states, init, events, trans)


T_1 = make_transition_system(P_1)
print('T_1:')
plot_transitionsystem(T_1, 'fig/P_1_transition_system')

# %% [markdown] deletable=false editable=false nbgrader={"cell_type": "markdown", "checksum": "3368b206e875792bff56831d018ef7ed", "grade": false, "grade_id": "cell-64e841c2f47c6614", "locked": true, "schema_version": 3, "solution": false, "task": false}
# ---

# %% [markdown] deletable=false editable=false nbgrader={"cell_type": "markdown", "checksum": "a5e6cdc2d39a43dee1d9f4c79a46a2df", "grade": false, "grade_id": "cell-a9f3f4ebf3699dc4", "locked": true, "schema_version": 3, "solution": false, "task": false}
# ## Task 1.1 
#
# ### ___CTL*___-Specification to $\mu$-Calculus *[2p]*
#
# In this assignment, you will implement the $\mu$-calculus algorithms for the booking problem $P_i$ shown above. Specifically, you will check its TransitionSystem model $T_i$ for the following temporal logic specification given in ___CTL*___:
#
# <center>
#     $\varphi_i = \forall \square \exists \lozenge m_i$,
# </center>
# with $m_i$ indicating a specific initial marking of $P_i$ (e.g. $m_1 = [3 0 2 0 2 1]$). 
#
# * In simple words, what does this specification $\varphi_1 = \forall \square \exists \lozenge m_1$ say and why does it specify a desired behavior? Write your answer below. ***[1p]***

# %% [markdown] deletable=false nbgrader={"cell_type": "markdown", "checksum": "2e1e7a9738011d7117bc4079b1645aaf", "grade": true, "grade_id": "spec_in_words", "locked": false, "points": 1, "schema_version": 3, "solution": true, "task": false}
# It means that you will/can forever, always, take a path which eventually leads you to a marking. Means that no matter how many times or what path you take you can always reach a desired state m_i. There are no blocking states.
#

# %% [markdown] deletable=false editable=false nbgrader={"cell_type": "markdown", "checksum": "6dc13cf29f0ef2f951efe970bcafa750", "grade": false, "grade_id": "cell-a1922ae5d4d411ae", "locked": true, "schema_version": 3, "solution": false, "task": false}
# Now, 
# * transform the ___CTL*___ formula $\varphi_1$ to $\mu$-calculus set expressions. Show also intermediate results. ***[1p]***
# * *Note*: [This webpage](http://detexify.kirelabs.org/classify.html) may be useful for finding the right LaTex symbols.
# * *Hint:* The double square brackets $[\![p]\!]$ can be done with [ \ ! [ p ] \ ! ] within the math environment. The LaTeX environments of the [amsmath package](https://tex.stackexchange.com/questions/3782/how-can-i-split-an-equation-over-two-or-more-lines) are also supported (equation, multiline and split).

# %% [markdown] deletable=false nbgrader={"cell_type": "markdown", "checksum": "336f5fbe458b8980c23c131a33b3996a", "grade": true, "grade_id": "cell-97e5d76cea7a732d", "locked": false, "points": 1, "schema_version": 3, "solution": true, "task": false}
#

# %% [markdown] deletable=false editable=false nbgrader={"cell_type": "markdown", "checksum": "4c6e305cc0c1027046d34d4512df2d4d", "grade": false, "grade_id": "cell-2b90678f9da35277", "locked": true, "schema_version": 3, "solution": false, "task": false}
# ---

# %% [markdown] deletable=false editable=false nbgrader={"cell_type": "markdown", "checksum": "277cfaaa94ce71952ff2d1966b4586b3", "grade": false, "grade_id": "cell-eba6f74362201fa1", "locked": true, "schema_version": 3, "solution": false, "task": false}
# ## Task 1.2 
# ### $\text{Pre}^\exists$ Operator *[1p]*
#
# $\mu$-calculus includes next modality functions $f \in \mathcal{F}$, namely $f = \exists \bigcirc$ and $f = \forall \bigcirc$. You will implement $f = \exists \bigcirc$ as __predecessor set operation__ in this section of the assignment. We will need this operator to check our model of the booking problem. 
#
# This set operator is defined as
#
# <center>
#    $\text{Pre}^\exists(Y) = \{x \mid (\exists a \in \Sigma(x))\delta(x, a) \subseteq Y \}$, 
# </center>
# where $x \in X$ and $Y \in 2^X$.
#
# Now,
# * implement the $\text{Pre}^\exists$ set operator in the `pre_exists` function.
# * _Hint:_ The inbuilt Python function [`any`](https://docs.python.org/3/library/functions.html#any) might be useful here.

# %% deletable=false editable=false nbgrader={"cell_type": "code", "checksum": "cb54d8925972a1800195efaa8b95f277", "grade": false, "grade_id": "cell-f35fd41037c8f6f6", "locked": true, "schema_version": 3, "solution": false, "task": false}
# You might want to use one of these...
from util import filter_trans_by_source, filter_trans_by_events, filter_trans_by_target
from util import extract_elems_from_trans, flip_trans


# %% deletable=false nbgrader={"cell_type": "code", "checksum": "375c18b212193399f738ceeda6780e56", "grade": false, "grade_id": "pre_exists", "locked": false, "schema_version": 3, "solution": true, "task": false}
def pre_exists(Y, ts):
    """
    Returns the new set of states for which the exists next modality is true.
    
    :param Y: Set of States
    :param ts: TransitionSystem
    """
    # YOUR CODE HERE
    raise NotImplementedError()
    return Y


# %%
# space for your own tests

# %% deletable=false editable=false nbgrader={"cell_type": "code", "checksum": "6301576348c56abda380a784853cde6d", "grade": true, "grade_id": "pre_exists_tests", "locked": true, "points": 1, "schema_version": 3, "solution": false, "task": false}
s1 = State(1)
s2 = State(2)

T0 = TransitionSystem({s1}, {s1}, set(), set())
assert pre_exists(set(), T0) == set()
assert pre_exists({s1}, T0) == set()

T1 = TransitionSystem({s1}, {s1}, {'a'}, {Transition(s1, 'a', s1)})
assert pre_exists(set(), T1) == set()
assert pre_exists({s1}, T1) == {s1}

T2 = TransitionSystem({s1, s2}, {s1}, {'a'}, {Transition(s1, 'a', s2)})
assert pre_exists(set(), T2) == set()
assert pre_exists({s1}, T2) == set()
assert pre_exists({s2}, T2) == {s1}
assert pre_exists({s1, s2}, T2) == {s1}


# %% [markdown] deletable=false editable=false nbgrader={"cell_type": "markdown", "checksum": "7029d24fbcf22b1c3183c8f882e54d7c", "grade": false, "grade_id": "cell-667a34de277b8c78", "locked": true, "schema_version": 3, "solution": false, "task": false}
# This $\text{Pre}^\forall$ set operator for $f = \forall \bigcirc$ is defined as
#
# <center>
#    $\text{Pre}^\forall(Y) = \{x \mid (\forall a \in \Sigma(x))\delta(x, a) \subseteq Y \}$, 
# </center>
# where $x \in X$ and $Y \in 2^X$.
#
# An implementation is given below.

# %% deletable=false editable=false nbgrader={"cell_type": "code", "checksum": "8214459668726adce417dd5a7d0ac938", "grade": false, "grade_id": "cell-32bdbb3bef5deab4", "locked": true, "schema_version": 3, "solution": false, "task": false}
def pre_forall(Y, ts):
    """
    Returns the new set of states for which the forall next modality is true.
    
    :param Y: Set of States
    :param ts: TransitionSystem
    """
    def all_t_into_Y_from(source):
        out_transitions = filter_trans_by_source(ts.trans, {source})
        return all({t.target in Y for t in out_transitions})
    
    Y = {x for x in ts.states if all_t_into_Y_from(x)}
    return Y


# %% [markdown] deletable=false editable=false nbgrader={"cell_type": "markdown", "checksum": "0b30e2e9fe3c06d3c0d0f87f53a34ac3", "grade": false, "grade_id": "cell-2e046444f492bb00", "locked": true, "schema_version": 3, "solution": false, "task": false}
# ---

# %% [markdown] deletable=false editable=false nbgrader={"cell_type": "markdown", "checksum": "517be1e2f510664109649fcb081ae93e", "grade": false, "grade_id": "cell-963815443f9f10c6", "locked": true, "schema_version": 3, "solution": false, "task": false}
# ## Task 1.3
#
# ### Model Checking through Fixed-Point Iteration *[2p]*
#
# After having reformulated $\varphi_i = \forall \square \exists \lozenge m_i$ as $\mu$-calculus set expressions, and having obtained an implementation of both $\text{Pre}^\exists$ and $\text{Pre}^\forall$, we can start on the actual model checking algorithm. 
#
# * Implement the function `is_always_eventually_satisfied` that takes as inputs an atomic proposition (e.g. $m_i$) and a TransitionSystem. It then checks whether the TransitionSystem satisfies $\varphi_i$. ***[2p]***
# * _Hint:_ You need to implement both a least fixed-point iteration $\mu Z$ and a greatest fixed-point iteration $\nu Y$.

# %% deletable=false nbgrader={"cell_type": "code", "checksum": "4800761135b89b61bd058c4aa4544eb9", "grade": false, "grade_id": "check", "locked": false, "schema_version": 3, "solution": true, "task": false}
def is_always_eventually_satisfied(m_i, ts):
    """
    Checks if a TransitionSystem always eventually satisfies an atomic proposition.
    
    :param m_i: String/integer. Atomic proposition
    :param ts: TransitionSystem to check
    """
    satisfied = False
    # YOUR CODE HERE
    raise NotImplementedError()
    return satisfied


m_1 = '[3 0 2 0 2 1]'
print('T_1 {} satisfy phi_1!'.format('does' if is_always_eventually_satisfied(m_1, T_1) else 'does NOT'))

# %%
# space for your own tests

# %% deletable=false editable=false nbgrader={"cell_type": "code", "checksum": "b9283cb938111a6b393839cb3d3cc2bc", "grade": true, "grade_id": "check_tests", "locked": true, "points": 2, "schema_version": 3, "solution": false, "task": false}
s1 = State(1, {'AP'})
s2 = State(2)

T0 = TransitionSystem({s1}, {s1}, set(), set())
assert is_always_eventually_satisfied(1, T0) == True
assert is_always_eventually_satisfied('AP', T0) == True

T1 = TransitionSystem({s1}, {s1}, {'a'}, {Transition(s1, 'a', s1)})
assert is_always_eventually_satisfied(1, T1) == True

T2 = TransitionSystem({s1, s2}, {s1}, {'a'}, {Transition(s1, 'a', s2)})
assert is_always_eventually_satisfied(1, T2) == False
assert is_always_eventually_satisfied(2, T2) == True

T3 = TransitionSystem({s1, s2}, {s1}, {'a'}, {Transition(s1, 'a', s1), Transition(s1, 'a', s2)})
assert is_always_eventually_satisfied(1, T3) == False
assert is_always_eventually_satisfied(2, T3) == True

T4 = TransitionSystem({s1, s2}, {s1}, {'a'}, {Transition(s1, 'a', s1), 
                                              Transition(s1, 'a', s2), 
                                              Transition(s2, 'a', s2)})
assert is_always_eventually_satisfied(1, T4) == False
assert is_always_eventually_satisfied(2, T4) == True

s3 = State(3)
s4 = State(4, {3})
T5 = TransitionSystem({s1, s2, s3, s4}, {s1, s2}, {'a', 'b'}, 
                      {Transition(s1, 'a', s3),
                       Transition(s1, 'b', s4),
                       Transition(s2, 'a', s4)})
assert is_always_eventually_satisfied("AP", T5) == False
assert is_always_eventually_satisfied(2, T5) == False
assert is_always_eventually_satisfied(3, T5) == True
assert is_always_eventually_satisfied(4, T5) == False

T6 = TransitionSystem({s1, s2, s3, s4}, {s1}, {'a', 'b'}, 
                      {Transition(s1, 'a', s2),
                       Transition(s1, 'b', s3),
                       Transition(s2, 'a', s1),
                       Transition(s2, 'b', s4),
                       Transition(s3, 'a', s1),
                       Transition(s3, 'b', s4),
                       Transition(s4, 'a', s2)})
assert is_always_eventually_satisfied(1, T6) == True
assert is_always_eventually_satisfied(2, T6) == True
assert is_always_eventually_satisfied(3, T6) == True
assert is_always_eventually_satisfied(4, T6) == True

# %% [markdown] deletable=false editable=false nbgrader={"cell_type": "markdown", "checksum": "48ef0e33e01d6f744e47f4cfcec3bb8b", "grade": false, "grade_id": "cell-eb911511d14629a0", "locked": true, "schema_version": 3, "solution": false, "task": false}
# ---

# %% [markdown] deletable=false editable=false nbgrader={"cell_type": "markdown", "checksum": "1fb4352a433820fb31b3fe7dec05190e", "grade": false, "grade_id": "cell-8ed6180918fb0dfc", "locked": true, "schema_version": 3, "solution": false, "task": false}
# ## Task 1.4
#
# ### Solving the Booking Problem *[1p]*
#
# It seems that there is an issue with the booking of the resources in the system (the Petri net from earlier) at some point. And that even pertains when we scale the system.

# %%
# e.g. multiplying all tokens by factor 7
m_2 = '[21 0 14 0 14 7]'
T_2 = make_transition_system(make_petrinet(p11_tokens=21, p21_tokens=14, R1_tokens=14, R2_tokens=7))
print('T_2 {} satisfy phi_2!'.format('does' if is_always_eventually_satisfied(m_2, T_2) else 'does NOT'))

"""
!!! PLEASE COMMENT OUT THE LINE BELOW BEFORE SUBMISSION !!!
"""
# plot_transitionsystem(T_2, 'fig/T_2')

# %% [markdown] deletable=false editable=false nbgrader={"cell_type": "markdown", "checksum": "ea7846c7117d82e067edf1e8a5a00220", "grade": false, "grade_id": "cell-81c92aab8a1c7cf6", "locked": true, "schema_version": 3, "solution": false, "task": false}
# Given that observation:
#
# * Change $P_1$ so that the booking problem is resolved. Elements of the original structure of the Petri net (places and arcs) must not be removed in that. ***[1p]***
# * *Hint:* Based on the model you could figure out what the problem is, and start to experiment with the model to find a way to prevent the problem from occurring. 

# %% deletable=false nbgrader={"cell_type": "code", "checksum": "5055c76d0df8ba5fc43fd4198f9fe358", "grade": false, "grade_id": "fix", "locked": false, "schema_version": 3, "solution": true, "task": false}
from util import array_str


def make_fixed_petrinet():
    """Makes a new, altered, issue-free version of P_1"""
    # YOUR CODE HERE
    raise NotImplementedError()
    return P


P_3 = make_fixed_petrinet()
T_3 = make_transition_system(P_3)
print('T_3 {} satisfy phi_3!'.format('does' if is_always_eventually_satisfied(array_str(P_3.init_marking), T_3)
                                     else 'does NOT'))
print('P_3:')
plot_petrinet(P_3, 'fig/P_3')

# %% [markdown] deletable=false editable=false nbgrader={"cell_type": "markdown", "checksum": "1f9e8e59a7b7a4c89c48e338f9b14818", "grade": false, "grade_id": "cell-394dfb346b1edbd1", "locked": true, "schema_version": 3, "solution": false, "task": false}
# Finally, you found a configuration of your model for which everything is fine. You get super excited and burst into your boss' office, and exclaim: _"I know how to fix our problem with the system! Give me a raise!"_ And your boss gets super excited and asks you how to solves her greatest nightmare. Obviously, you cannot say now: _"We need to get a big can of paint and draw some circles and arrows on the floor!"_ (except maybe if you know some ancient magic ritual against booking problems). No, you need to translate your model changes into your boss' world, i.e. do we need to change the process, buy new equipment, re-program the controllers, etc. Also, you cannot decide that one of the resources is no longer needed for processing or remove one of the two processes altogether. In other words, the places, transitions and arcs of the original petri net $P_1$ must remain. You can add places and so on, or adjust the number of tokens as long  as the original structure is part of the new model and you can motivate your changes in words that are understandable to your manager, who knows nothing about discrete event systems.
#
# * Describe briefly in words how you could implement your proposed changes in the real system. ***[1p]***

# %% [markdown] deletable=false nbgrader={"cell_type": "markdown", "checksum": "c767c7ccb37598cd4f0e949a517b137f", "grade": true, "grade_id": "implementation", "locked": false, "points": 1, "schema_version": 3, "solution": true, "task": false}
# YOUR ANSWER HERE

# %% [markdown] deletable=false editable=false nbgrader={"cell_type": "markdown", "checksum": "b50e4d222d7c457b732eec2e2deac242", "grade": false, "grade_id": "cell-04fd41381b92af64", "locked": true, "schema_version": 3, "solution": false, "task": false}
# ## Bonus Task (optional)
#
# * Can you derive a general rule to avoid the booking problem for the given Petri net structure of $P_1$, that is a booking problem with two parallel processes? If yes, explain it with a few words. ***[+1p]***

# %% [markdown] deletable=false nbgrader={"cell_type": "markdown", "checksum": "0f520f0417d6081a6d1b0e2148040e25", "grade": true, "grade_id": "bonus", "locked": false, "points": 0, "schema_version": 3, "solution": true, "task": false}
# YOUR ANSWER HERE

# %% [markdown] deletable=false editable=false nbgrader={"cell_type": "markdown", "checksum": "9e018bb429e8652ff4f741fc2a0f7d64", "grade": false, "grade_id": "cell-3aebb8bbe5d39250", "locked": true, "schema_version": 3, "solution": false, "task": false}
# ---

# %% [markdown] deletable=false editable=false nbgrader={"cell_type": "markdown", "checksum": "357c9e99c2bd2e951ab8d38105c60e65", "grade": false, "grade_id": "cell-141c000a5ea6f1d4", "locked": true, "schema_version": 3, "solution": false}
# # Part 2: Reinforcement Learning
#
# After having had a tremendously successful day at work, you call your friends to share the story of how you single-handedly solved your company's booking issue using $\mu$-calculus. All agree that this needs to be celebrated accordingly, and so you meet up at Andra Långgatan. Your likely future promotion is cause for a long and jolly evening, and when you finally decide to go home, it has become quite dark already. To make matters even worse, a strong wind is blowing from south/south-west, and you seem to have forgotten the way to your home in Gamlestaden (for whatever reason - blame it on the long day thinking about temporal logic if you'd like). You conclude that your best option would be to start walking in some direction in the hope of finding your home eventually. But careful! If you walk too close alongside the Göta Älv, the strong wind might blow you into the river. In addition to the unpleasant experience, the river will take you back to Järntorget and you need to start your journey home all over again. And if that wasn't bad enough yet, you might get convinced to join some _"late-night studying"_ if you pass by J. A. Pripps at Chalmers.   
#
# In this part of the assignment, you will implement a Reinforcement Learning algorithm called Q-learning. Reinforcement Learning uses data sampled from the plant (or the environment in RL terms) to derive an optimal controller - just right for finding your way back home.
#
# Let us look at the environment first:
#
# ![WindyGothenburg](fig/windy_gothenburg.png)
#
# The available actions in this environment are: {'north', 'east', 'south', 'west'}. In addition to the action you take, the wind will blow you with a probability of 5% to the east and with a probability of 10% to the north. To start with, we simplify the environment by removing the river and J.A. Pripps. These two features will be added again later on. You can find the full code of the environment in `util/datastructures.py`. Here, we will import it and configure it now:

# %% deletable=false editable=false nbgrader={"cell_type": "code", "checksum": "daa2682e7ed386af60891e5a2ffe8e28", "grade": false, "grade_id": "cell-6dfafa26e37f7224", "locked": true, "schema_version": 3, "solution": false, "task": false}
from util import WindyGothenburg

BASIC_ENV = {'name': 'Basic Env', 
             'w': 12, 'h': 12,                    # width and height of the grid
             'obstacles': {(2,9), (3,9), (4,9),
                          (8,9), (9,9), (10,9)},  # Coordinates of obstacles on the grid 
             'water': set(),                      # Coordinates of the Göta Älv (not included in this env)
             'pripps_reward': None}               # Reward for passing by J.A. Pripps (not included in this env)

# %% [markdown] deletable=false editable=false nbgrader={"cell_type": "markdown", "checksum": "39d94ae988c449c2460e80a21326e41f", "grade": false, "grade_id": "cell-7b0a3e55845aa420", "locked": true, "schema_version": 3, "solution": false}
# The RL algorithm that you will implement is as follows:
#
# **Algorithm 1.** Q-learning$(\alpha, \epsilon, \gamma)$
#
# >Initialize $Q(x,a)$ arbitrarily
# >
# >**for all** episodes **do**
# >
# >>Initialize $x$
# >>
# >>**for all** steps of episode  **do**
# >>>
# >>>Choose $a$ in $x$ using policy derived from $Q$ (e.g. $\epsilon$-greedy)
# >>>
# >>>Take action $u$, observe $r$, $x'$
# >>>
# >>>$Q(x,a) = Q(x,a) + \alpha \left[r + \gamma \max_{a'} Q(x', a') - Q(x,a) \right]$
# >>>
# >>>$x = x'$
# >>>
# >>**end for**
# >
# >**end for**
# >
# **return** $\pi(a) = argmax_a Q(x, a)$
#
# We have already implemented parts of this algorithm for you. Make sure to read carefully all the functions below. One of those functions creates the Q-table by using nested Python [dictionaries](https://docs.python.org/3/library/stdtypes.html#mapping-types-dict). 

# %% deletable=false editable=false nbgrader={"cell_type": "code", "checksum": "8bfb7bde012adc3a6381d8afd065c29a", "grade": false, "grade_id": "cell-2939871e3c171dbc", "locked": true, "schema_version": 3, "solution": false}
import random

def initialize_Q(states, actions, scaling_factor=0.1):
    """
    Initializes the Q-table as a dictionary of dictionaries.
    
    A particular Q-value can be retrieved by calling Q[x][a].
    All actions and their associated values in a state x can 
    be retrieved through Q[x].
    Q-values are initialized to a small random value to encourage 
    exploration and to facilitate learning.
    
    :param states: iterable set of states
    :param actions: iterable set of actions
    """
    return {x: {a: random.random() * scaling_factor for a in actions} for x in states}

def argmax_Q(Q, state):
    """Computes the argmax of Q in a particular state."""
    max_q = float("-inf")
    argmax_q = []
    for a, q in Q[state].items():
        if q == max_q:
            argmax_q.append(a)
        if q > max_q:
            max_q = q
            argmax_q = [a]
    return random.choice(argmax_q)


# %% [markdown] deletable=false editable=false nbgrader={"cell_type": "markdown", "checksum": "18c4c6f332d2055358790adc6a1f2439", "grade": false, "grade_id": "cell-05ba1591136d06b7", "locked": true, "schema_version": 3, "solution": false}
# ---

# %% [markdown] deletable=false editable=false nbgrader={"cell_type": "markdown", "checksum": "5a49a7f6e0ea81bfecbabfe091bbe0ff", "grade": false, "grade_id": "cell-cb04ccc104be9b68", "locked": true, "schema_version": 3, "solution": false}
# ## Task 2.1
#
# ### Finding Home through Q-learning *[1p]*
#
# As a first task, implement a function that chooses with probability $1-\epsilon$ the action with the highest $Q$-value in a given state (i.e. it chooses greedily), and with probability $\epsilon$ a random action, where $0 < \epsilon < 1$. This is a popular exploration strategy in RL that ensures that all states are visited theoretically infinitively often.
#
# * Implement the $\epsilon$-greedy choice in code. 
# * *Hint*: You might want to use the Python function [random.random()](https://docs.python.org/3/library/random.html#random.random) and [random.choice()](https://docs.python.org/3/library/random.html#random.choice).
# * *Hint*: You might want to read the documentation of [dictionaries](https://docs.python.org/3/library/stdtypes.html#mapping-types-dict) again.

# %% deletable=false nbgrader={"cell_type": "code", "checksum": "9b9e8dca67734386b89411013321337b", "grade": false, "grade_id": "epsilon_greedy", "locked": false, "schema_version": 3, "solution": true}
def choose_epsilon_greedily(Q, x, epsilon):
    """
    Chooses random action with probability epsilon, else argmax_a(Q(*|x))
    
    :param Q: Q-table as dict of dicts
    :param x: state
    :param epsilon: float
    """
    # YOUR CODE HERE
    rng = random.random()
    
    if rng > epsilon:
        #choose greedyly
        action = argmax_Q(Q,x)
    else:
        #choose non greedy\
        action = random.choice(list(Q[x]))
    return action


# %% [markdown] deletable=false editable=false nbgrader={"cell_type": "markdown", "checksum": "2c8692c6043aa1eea122293cd4de9fd8", "grade": false, "grade_id": "cell-15e9519922bab68d", "locked": true, "schema_version": 3, "solution": false}
# Next, we need to implement a function that decides how fast our RL algorithm will be learning. Generally, the learning rate $\alpha_k$ must satisfy the conditions $\sum_{k=0}^\infty \alpha_k^2 < \infty$ and $\sum_{k=0}^\infty \alpha_k = \infty$, to be able to guarantee that the estimates of Q converge to the optimal Q-function.
#
# * Implement a function that computes $\alpha$ given the state-action visitation count $k$. 
# * *Hint*: Check the lecture notes for further information.
#
# A correct implementation of both functions is needed to complete Task 2.1. 

# %% deletable=false nbgrader={"cell_type": "code", "checksum": "cbf308a8fd8df8db8391c965960b15d2", "grade": false, "grade_id": "alpha", "locked": false, "schema_version": 3, "solution": true}
def get_alpha(x, a, stateaction_visitation_counts, A, B):
    """
    Returns a value of the learning rate.
    
    A particular state-action visitation count can be 
    retrieved by calling stateaction_visitation_count[x][a].
    :param x: state
    :param a: action
    :param stateaction_visitation_counts: dictionary of dictonaries
    :param A: integer parameter of the learning rate
    :param B: integer parameter of the learning rate
    """
    # YOUR CODE HEre 
    # antar B = 2A
    N=stateaction_visitation_counts[x][a]
    alpha = A / (N + B) # formel från s.5 i  LLD Lecture Notes Reinforcement Learning.pdf
    return alpha


# %% deletable=false editable=false nbgrader={"cell_type": "code", "checksum": "38981f9b3430ff6141240c95c0a663fd", "grade": false, "grade_id": "helper_test", "locked": true, "schema_version": 3, "solution": false, "task": false}
# epsilon-greedy tests
Q1 = initialize_Q(states={1}, actions={'a'})
assert choose_epsilon_greedily(Q1, 1, 0.1) == 'a'

Q2 = initialize_Q(states={1}, actions={'a', 'b'})
Q2[1]['a'] = 1
Q2[1]['b'] = 0
assert choose_epsilon_greedily(Q2, 1, 0.0) == 'a'

epsilon = 0.1
k = 0
l = 0
for m in range(1000):
    action = choose_epsilon_greedily(Q2, 1, epsilon)
    k = k + 1 if action == 'a' else k
    l = l + 1 if action == 'b' else l
assert k/m >= (1-epsilon)
assert l/m > 0.0

# learning rate tests
x = (0, 0)
svc = {x: {'a': 0}}
assert 0.0 < get_alpha(x, 'a', svc, 1, 2) < 1.0
assert 0.0 < get_alpha(x, 'a', svc, 1e10, 2e10) < 1.0
svc = {x: {'a': 1e10}}
assert 0.0 < get_alpha(x, 'a', svc, 1, 2) < 1.0
assert 0.0 < get_alpha(x, 'a', svc, 1e10, 2e10) < 1.0


# %% [markdown] deletable=false editable=false nbgrader={"cell_type": "markdown", "checksum": "8f734d3c78f63d109912010158cf7037", "grade": false, "grade_id": "cell-0860dfc12d3aa182", "locked": true, "schema_version": 3, "solution": false}
# Finally, we can turn to the $Q$-learning algorithm. We have made a start with it already. 
#
# * Now, implement the $Q$-value update from Algorithm 1. The relevant line is marked with a comment in the code below. 

# %% deletable=false nbgrader={"cell_type": "code", "checksum": "a93dd12f3f7fe0f93e5d195d02ff29a6", "grade": false, "grade_id": "learn_q_def", "locked": false, "schema_version": 3, "solution": true, "task": false}
def learn_q(env, epsilon, gamma, A, B, num_episodes=250, max_steps=100, render=False, test=False):
    
    Q = initialize_Q(env.states, env.actions, scaling_factor=0.1)
    counts =  {x: {a: 0 for a in env.actions} for x in env.states}
    
    stats = {'avg_r_smoothed': 0, 'eps_goal_found': num_episodes, 
             'eps_goal_learned': num_episodes, 'max_r_smoothed': 0}
    
    if A>B:
        if not test:
            s = 'A cannot be greater than B. They are A = {} and B = {}.'.format(A, B) 
            print(s + ' Returning random policy and default learning statistics now.')
        env.close()
        stats = {'avg_r_smoothed': -np.inf, 'eps_goal_found': np.inf, 
             'eps_goal_learned': np.inf, 'max_r_smoothed': -np.inf}
        return {x: argmax_Q(Q, x) for x in env.states}, stats
    
    for l in range(num_episodes):
        # Reset for episod
        x = env.reset()
        done = False
        sum_of_r = 0

        for m in range(max_steps):
            # Pick action
            a = choose_epsilon_greedily(Q, x, epsilon)
            next_x, r, done = env.step(a)  
            
            alpha = get_alpha(x, a, counts, A, B)
                
            # Update Q-Table
            # YOUR CODE HERE
            counts[x][a] = counts[x][a] + 1
            next_best_action = argmax_Q(Q,next_x)
            bigterm = r + gamma*Q[next_x][next_best_action] - Q[x][a]
            Q[x][a] = Q[x][a] + alpha * bigterm


            # Increment
            x = next_x
            sum_of_r += r
            
            if render:
                env.render(Q)
        
            if done:
                # Set the Q-values of the terminal state to 0
                for action in Q[next_x].keys():
                    Q[next_x][action] = 0
                break
        
        # Track some statistics
        avg_r = sum_of_r / (m+1)
        stats['avg_r_smoothed'] = 0.95 * stats['avg_r_smoothed'] + 0.05 * avg_r
        
        if r == 100 and stats['eps_goal_found'] == num_episodes:
            stats['eps_goal_found'] = l
        if stats['avg_r_smoothed'] > 2.0 and stats['eps_goal_learned'] == num_episodes:
            stats['eps_goal_learned'] = l
        if stats['avg_r_smoothed'] > stats['max_r_smoothed']:
            stats['max_r_smoothed'] = stats['avg_r_smoothed']

        # Update plots
        if not test:
            env.render(Q, avg_r, stats['avg_r_smoothed'], l)
    
    env.close()
    return {x: argmax_Q(Q, x) for x in env.states}, stats


# %% [markdown] deletable=false editable=false nbgrader={"cell_type": "markdown", "checksum": "04567fb9a5df54fb047e9ccf90dad9cc", "grade": false, "grade_id": "cell-caf9570be22c6c44", "locked": true, "schema_version": 3, "solution": false}
# Your task is now to:
# * play around with the Q-learning for the Basic WindyGothenburg environment and its parameters. In particular:
#   * Choose a value for $\epsilon$
#   * Choose a value for the discount factor $\gamma$
#   * Choose a value for the count-based learning parameters $A$ and $B$
# * Your Q-learning will return the learned policy, which is evaluated in the test cell below. 
# * If your parameters produces policies that get you home in more than 75% of the learning repetitions, we are satisfied. ***[1p]*** 
# * _Hint_: Make sure to truly understand our evaluation criteria.
# * _Hint_: Each parameter affects the learning differently. What values for each parameter would likely produce the wanted behavior? 

# %% deletable=false nbgrader={"cell_type": "code", "checksum": "8a25af0498d3fb12f177e6bbe22e6ed9", "grade": false, "grade_id": "cell-fd1bedcc8b202c2b", "locked": false, "schema_version": 3, "solution": true, "task": false}
# If you would like to watch the episode, set render = True
# For updates only at the end of each episode, set render = False
# render = False is significantly faster.
render = False

# Choose values for epsilon, gamma, A, and B
epsilon = 0.3
gamma = 0.8
A = 75 # a in [5,150]
B = 150 # b=2a 
# YOUR CODE HERE


"""
!!! PLEASE COMMENT OUT THE TWO LINES BELOW BEFORE SUBMISSION !!!
"""

env = WindyGothenburg(**BASIC_ENV)
control_policy = learn_q(env, epsilon, gamma, A=A, B=B, num_episodes=250, render=render)

"""
!!! PLEASE COMMENT OUT THE TWO LINES ABOVE BEFORE SUBMISSION !!!
"""

# %% deletable=false editable=false nbgrader={"cell_type": "code", "checksum": "2a757128586d9bf89ed0546a14f136d3", "grade": false, "grade_id": "eval_param", "locked": true, "schema_version": 3, "solution": false, "task": false}
metrics = ['avg_success_rate', 'avg_eps_goal_found', 'avg_eps_goal_learned', 'avg_max_r_smoothed']

def eval_learning_params(EnvClass, env_config, learning_func, epsilon, gamma, A, B, pripps_reward=None, 
                         num_episodes=250, repeats=250, max_steps_for_success=20):
    """
    Evaluates the hyperparameters of a learning function 
    by repeating the learning several times. Each time, it 
    is checked whether the learned policy is close to the 
    true optimal policy.
    """
    averages = {'avg_success_rate': 0, 'avg_eps_goal_found': 0, 
                  'avg_eps_goal_learned': 0, 'avg_max_r_smoothed': 0}
    
    for _ in range(repeats):
        # Learn policy
        env = EnvClass(**env_config | {'pripps_reward': pripps_reward}, test=True)
        control_policy, stats = learning_func(env, epsilon=epsilon, gamma=gamma, A=A, B=B, num_episodes=num_episodes,
                                 max_steps=100, render=False, test=True)
        x = env.reset()
        done = False
        i = 0
        # Evaluate learned policy
        while not done and i < max_steps_for_success:
            x, r, done = env.step(control_policy.get(x))
            i += 1
        averages['avg_success_rate'] += 1 if r == 100 else 0
        averages['avg_eps_goal_found'] += stats['eps_goal_found']
        averages['avg_eps_goal_learned'] += stats['eps_goal_learned']
        averages['avg_max_r_smoothed'] += stats['max_r_smoothed']
        
    return {metric: value/repeats for metric, value in averages.items()} | {'epsilon': epsilon, 
                                                                            'gamma': gamma, 'A': A, 'B': B,
                                                                            'pripps_reward': pripps_reward}


# %% deletable=false editable=false nbgrader={"cell_type": "code", "checksum": "02b635fd8e3ce031ab2e0900f9c31fa0", "grade": true, "grade_id": "success_075", "locked": true, "points": 1, "schema_version": 3, "solution": false, "task": false}
BASIC_ENV = {'name': 'Basic Env', 
             'w': 12, 'h': 12,                    # width and height of the grid
             'obstacles': {(2,9), (3,9), (4,9),
                          (8,9), (9,9), (10,9)},  # Coordinates of obstacles on the grid 
             'water': set(),                      # Coordinates of the Göta Älv (not included in this env)
             'pripps_reward': None}               # Reward for passing by J.A. Pripps (not included in this env)

stats = eval_learning_params(WindyGothenburg, BASIC_ENV, learn_q, 
                             epsilon=epsilon, gamma=gamma, A=A, B=B, 
                             num_episodes=250, repeats=250, max_steps_for_success=16)

assert stats['avg_success_rate'] > 0.75, 'Got {} instead'.format(stats['avg_success_rate'])
print("The achieved average success rate was ", stats['avg_success_rate'])

# %% [markdown] deletable=false editable=false nbgrader={"cell_type": "markdown", "checksum": "5f9bc6083afe4492a4070e0597786894", "grade": false, "grade_id": "cell-01abc3c9adfaffd2", "locked": true, "schema_version": 3, "solution": false, "task": false}
# ---

# %% [markdown] deletable=false editable=false nbgrader={"cell_type": "markdown", "checksum": "38db040ad6473e128033d17471b9ade1", "grade": false, "grade_id": "cell-aa0bdaaf74b6e4e9", "locked": true, "schema_version": 3, "solution": false}
# ## Task 2.2
#
# ### Reflections on Hyperparameters of the Basic Environment *[1p]*
#
# You may have noticed by now that different $\epsilon$ or $\gamma$ lead to different learning behaviors. We have implemented a function for you that allows for plotting a grid of hyperparameters. This will hopefully give you better insight into tuning the hyperparameters of the reinforcement learning algorithm. Beside the average success rate (see code above), the function plots three more statistics that are the averages of these values:
#
# ![statistics](fig/learning_statistics.png)
#
# Although we choose to prioritize the success rate, a high average maximum reward is often the key metric reported on in many research papers. 
#
# The code below executes the learning in parallel on all available CPU-cores of your machine and plots the results. 

# %% deletable=false editable=false nbgrader={"cell_type": "code", "checksum": "cefdabfb032d2fdb571035bc31a0ab56", "grade": false, "grade_id": "cell-32a7d698848e259d", "locked": true, "schema_version": 3, "solution": false, "task": false}
import os
from itertools import product
from multiprocessing import Pool
from collections import namedtuple
from util import validate_grid_config, write_defs_to_file, plot_heatmaps

HyperParameterGrid = namedtuple('HyperParameterGrid', ['epsilon', 'gamma', 'A', 'B', 'pripps_reward'], 
                                defaults=([0], [0], [0], [0], [None]))

def eval_hyperparam_grid(env_config, param_grid, num_episodes=250, repeats=250, max_steps_for_success=20, 
                         log_y=False, save_name=None):
    """
    Executes the function `eval_learning_params` concurrently for 
    all combinations of the parameters in param_grid.
    """
    X, Y, x_name, y_name, title = validate_grid_config(env_config, param_grid)                
    write_defs_to_file(initialize_Q, argmax_Q, choose_epsilon_greedily, get_alpha, 
                       learn_q, eval_learning_params, f'./tmp.py')    # Prepare for multiprocessing    
    from tmp import task, learning_f
    print('Evaluating {} hyperparameter combinations ...'.format(len(X)*len(Y)))
    with Pool() as p:
        stats = p.starmap(task, product([WindyGothenburg], [env_config], [learning_f], 
                                        param_grid.epsilon, param_grid.gamma, 
                                        param_grid.A, param_grid.B, param_grid.pripps_reward,
                                        [num_episodes], [repeats], [max_steps_for_success]))
    fig = plot_heatmaps(stats, X, Y, metrics, x_name, y_name, save_name, title, log_y=log_y)
    os.remove(f'./tmp.py')
    return stats


# %% [markdown] deletable=false editable=false nbgrader={"cell_type": "markdown", "checksum": "e38da8b528534a82909bbe4eaf886269", "grade": false, "grade_id": "cell-5a2ae452bfb5dfc4", "locked": true, "schema_version": 3, "solution": false, "task": false}
# By that we can evaluate a hyperparameter grid like this: choose two hyperparameters that you would like to evaluate, define their levels as elements of a list (the combination of these become the data points of the grid), pass in the values of the other hyperparameters as lists with a single element. Here is an example: 

# %%
hpg = HyperParameterGrid(epsilon=[0.01, 0.1, 0.3, 0.5, 0.7, 0.9, 0.99], 
                         gamma=[0.01, 0.1, 0.3, 0.5, 0.7, 0.9, 0.99], 
                         A=[50], B=[100])

"""
!!! PLEASE COMMENT OUT THE TWO LINES BELOW BEFORE SUBMISSION !!!
"""

# stats = eval_hyperparam_grid(BASIC_ENV, hpg, max_steps_for_success=16, 
#                              save_name='hpg_basic_eps_gamma')

"""
!!! PLEASE COMMENT OUT THE TWO LINES ABOVE BEFORE SUBMISSION !!!
"""

# %% [markdown] deletable=false editable=false nbgrader={"cell_type": "markdown", "checksum": "3210d01ebc60bc10323f9e8909ca1bce", "grade": false, "grade_id": "cell-bf149830114caf1d", "locked": true, "schema_version": 3, "solution": false, "task": false}
# Now,
# * Reflect on the difference in hyperparameters that lead to a high average success rate (plot on the upper left) versus a high average maximum reward (plot on the upper right).
# * How is the optimal policy related to these two statistics?
# * Which of the two statistics is more relevant? And which hyperparameters would you choose? 
# * Write down your insights in a few brief sentences. ***[1p]***

# %% [markdown] deletable=false nbgrader={"cell_type": "markdown", "checksum": "18e6cacec4cd28a12f4bd23eb584ed01", "grade": true, "grade_id": "success_metrics", "locked": false, "points": 1, "schema_version": 3, "solution": true, "task": false}
# YOUR ANSWER HERE

# %% [markdown] deletable=false editable=false nbgrader={"cell_type": "markdown", "checksum": "08e1e20a11d3a49cc2553944021d6359", "grade": false, "grade_id": "cell-f61c81af7a807fb7", "locked": true, "schema_version": 3, "solution": false, "task": false}
# ---

# %% [markdown] deletable=false editable=false nbgrader={"cell_type": "markdown", "checksum": "ddf24c4703bfb1563cd23df823139416", "grade": false, "grade_id": "cell-9a248cdcf17a75eb", "locked": true, "schema_version": 3, "solution": false, "task": false}
# ## Task 2.3
#
# ### Find Hyperparameters of the River Environment that Lead to > 94% Success *[1p]*
#
# Now, we add back the Göta Älv to the environment. This will change a few things for the learning.
# * Tune the RL algorithm again such that a success rate of > 94 % is achieved. ***[1p]***

# %% deletable=false nbgrader={"cell_type": "code", "checksum": "65e7cb5eb217ff940f3744d004b060b7", "grade": false, "grade_id": "cell-b12cac2de52c8ccd", "locked": false, "schema_version": 3, "solution": true, "task": false}
RIVER_ENV = {'name': 'River Env',
             'w': 12, 'h': 12,                            # width and height of the grid
             'obstacles': {(2,9), (3,9), (4,9),
                          (8,9), (9,9), (10,9)},          # Coordinates of obstacles on the grid 
             'water': {(k, 12) for k in range(2, 12)},    # Coordinates of the Göta Älv 
             'pripps_reward': None}                       # Reward for passing by J.A. Pripps (not included in this env)

river_epsilon = None
river_gamma = None
river_A = None
river_B = None
# YOUR CODE HERE
raise NotImplementedError()

# %%
# Space for your own experiments and tests

"""
!!! PLEASE COMMENT OUT ALL CALLS TO eval_hyperparam_grid BEFORE SUBMISSION !!!
"""

# %% deletable=false editable=false nbgrader={"cell_type": "code", "checksum": "4584f47dfb05d39af18cb3191bb25774", "grade": true, "grade_id": "river_success", "locked": true, "points": 1, "schema_version": 3, "solution": false, "task": false}
RIVER_ENV = {'name': 'River Env',
             'w': 12, 'h': 12,                            # width and height of the grid
             'obstacles': {(2,9), (3,9), (4,9),
                          (8,9), (9,9), (10,9)},          # Coordinates of obstacles on the grid 
             'water': {(k, 12) for k in range(2, 12)},    # Coordinates of the Göta Älv 
             'pripps_reward': None}                       # Reward for passing by J.A. Pripps (not included in this env)

stats = eval_learning_params(WindyGothenburg, RIVER_ENV, learn_q, 
                             epsilon=river_epsilon, gamma=river_gamma, 
                             A=river_A, B=river_B, 
                             max_steps_for_success=20, repeats=1000)
assert stats['avg_success_rate'] > 0.94, 'Got {} instead'.format(stats['avg_success_rate'])
print("The achieved average success rate was ", stats['avg_success_rate'])

# %% [markdown] deletable=false editable=false nbgrader={"cell_type": "markdown", "checksum": "88cedf69547b48dde53e379a18bce22b", "grade": false, "grade_id": "cell-054e1636de18935e", "locked": true, "schema_version": 3, "solution": false, "task": false}
# ---

# %% [markdown] deletable=false editable=false nbgrader={"cell_type": "markdown", "checksum": "b2c82e0bedea193ca312da55725c1c1c", "grade": false, "grade_id": "cell-5d71ddfa490dff6f", "locked": true, "schema_version": 3, "solution": false, "task": false}
# ## Task 2.4
#
# ### Reflect on the Impact of Intermediate Rewards *[1p]*
#
# As the last part of the task, you will need to decide how much a visit to J.A. Pripps is worth to you. Note that this is strictly speaking not an element of the Q-learning algorithm, but rather an element of the reward function. While many publications on RL assume that the reward function is given, in practice, designing a good reward function, that enables the RL agent to learn the wanted behavior, is a difficult task. The complications often arise from balancing intermediate rewards, that can give continuous learning feedback, and terminal rewards, that indicate goal fulfillment. `pripps_reward` is an intermediate reward whereas the terminal rewards of `+100` for getting home or `-10` for falling into the river are terminal rewards.

# %% deletable=false nbgrader={"cell_type": "code", "checksum": "3212df35a1a20981720c0dfe38e98927", "grade": false, "grade_id": "pripps", "locked": false, "schema_version": 3, "solution": true, "task": false}
PRIPPS_ENV = {'name': 'Pripps Env',
             'w': 12, 'h': 12,                            # width and height of the grid
             'obstacles': {(2,9), (3,9), (4,9),
                          (8,9), (9,9), (10,9)},          # Coordinates of obstacles on the grid 
             'water': {(k, 12) for k in range(2, 12)},    # Coordinates of the Göta Älv 
             'pripps_reward': 1.828}                      # Reward for passing by J.A. Pripps 

pripps_epsilon = None
pripps_gamma = None
pripps_A = None
pripps_B = None
pripps_reward = 1.828
# YOUR CODE HERE
raise NotImplementedError()

# %%
# Space for your own experiments and tests
"""
!!! PLEASE COMMENT OUT ALL CALLS TO eval_hyperparam_grid BEFORE SUBMISSION !!!
"""

# %% deletable=false editable=false nbgrader={"cell_type": "code", "checksum": "d085d769ede1405d2f853179c1a315bd", "grade": false, "grade_id": "pripps_test", "locked": true, "schema_version": 3, "solution": false, "task": false}
PRIPPS_ENV = {'name': 'Pripps Env',
             'w': 12, 'h': 12,                            # width and height of the grid
             'obstacles': {(2,9), (3,9), (4,9),
                          (8,9), (9,9), (10,9)},          # Coordinates of obstacles on the grid 
             'water': {(k, 12) for k in range(2, 12)},    # Coordinates of the Göta Älv 
             'pripps_reward': 1.828}                      # Reward for passing by J.A. Pripps 

stats = eval_learning_params(WindyGothenburg, PRIPPS_ENV, learn_q, 
                             epsilon=pripps_epsilon, gamma=pripps_gamma, 
                             A=pripps_A, B=pripps_B, pripps_reward=pripps_reward,
                             max_steps_for_success=20)
assert stats['avg_success_rate'] > 0.9, 'Got {} instead'.format(stats['avg_success_rate'])
print("The achieved average success rate was ", stats['avg_success_rate'])

# %% [markdown] deletable=false editable=false nbgrader={"cell_type": "markdown", "checksum": "f74ceb02b6ad9c65f909a0e6bd0e874f", "grade": false, "grade_id": "cell-1b365d8cbf543f04", "locked": true, "schema_version": 3, "solution": false, "task": false}
# * Reflect on this part of the assignment and write down your insights in a few brief sentences. ***[1p]***
# * Make sure to touch upon:
#   * the final value of `pripps_reward` that you chose and why you have chosen that value;
#   * the impact of `pripps_reward` and $\gamma$ on solving the problem

# %% [markdown] deletable=false nbgrader={"cell_type": "markdown", "checksum": "31ffd2d7ccc589f3303950a7da251fc9", "grade": true, "grade_id": "pripps_reflections", "locked": false, "points": 1, "schema_version": 3, "solution": true, "task": false}
# YOUR ANSWER HERE

# %% [markdown] deletable=false editable=false nbgrader={"cell_type": "markdown", "checksum": "32f509d0422c74402a5de11b88aa801e", "grade": false, "grade_id": "cell-e53c87aa4cdb8edf", "locked": true, "schema_version": 3, "solution": false}
# ---

# %% [markdown] deletable=false editable=false nbgrader={"cell_type": "markdown", "checksum": "fdf45a1d97172bc8702a1c32c0d4b1f6", "grade": false, "grade_id": "cell-ca4413be58012d08", "locked": true, "schema_version": 3, "solution": false}
# That is all there is. If you are done,
#
# * Save the notebook
# * Upload the .ipynb file to Canvas
# * Tell your teammate how much you appreciated their invaluable insights and how fun it was to collaborate with them on the assignments.
