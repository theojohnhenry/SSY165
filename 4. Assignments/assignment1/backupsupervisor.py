# Now implement the function in codedddd
def supervisor(P, Sp, sigma_u):
    """
    Generates a nonblocking and controllable supervisor for the synchronized system P||Sp.
    
    :param P: automaton of the plant
    :param Sp: automaton of the specification
    :param sigma_u: set of uncontrollable events
    """
    # YOUR CODE HERE
    s_noll = synch(P,Sp)
    Q = s_noll.states
    sigma = s_noll.events
    delta = s_noll.trans
    Qm = s_noll.marked

    #uncontrollable states
    Quc = set()
    for uncontrollable_event in sigma_u:
        for state in s_noll.states:
            if is_defined_for_p(P.trans, state, {uncontrollable_event}) \
                    and not is_defined_for_q(Sp.trans, state, {uncontrollable_event}):
                Quc.add(state)

    #alg 3, genererar xk som är kombinationen av manuellt forbidden och uncontrollably forbidden
    k=0
    X = [s_noll.forbidden.copy().union(Quc)] ## snollforbidden fylls på med uncontrollable states
    while True:
        k = k+1
        Q_prim = coreach(sigma, delta, Qm, X[k-1])
        Q_bis = coreach(sigma_u, delta, Q.difference(Q_prim), set())
        X.append(X[k-1].union(Q_bis))
        if X[k] == X[k-1]:
            break


    s_states = Q.difference(X[k])
    print(f's states = {s_states}')

    if s_states == set():
        raise ValueError('value error, safe states are empty')


    #genererar alla tillåtna transitions
    s_trans = set()
    for trans in delta:
        if trans.source in s_states and trans.target in s_states:
            s_trans.add(trans)
    # generera marked, forbidden, init, events 


    s_init = s_noll.init
    s_events = s_noll.events
    s_marked = s_noll.marked.intersection(s_states)
    s_forbidden = s_noll.forbidden.intersection(s_states)
    #sammanst'll s 
    
    S = Automaton(
        states=s_states,
        init=s_init,
        events=s_events,
        trans=s_trans,
        marked=s_marked,
        forbidden=s_forbidden,
    )

    return S