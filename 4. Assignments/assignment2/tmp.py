import random
import numpy as np

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


def learning_f(env, epsilon, gamma, A, B, num_episodes=250, max_steps=100, render=False, test=False):
    
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


def task(EnvClass, env_config, learning_func, epsilon, gamma, A, B, pripps_reward=None, 
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
