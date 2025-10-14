import os
import platform
import subprocess

import numpy as np

from IPython.display import Image
from IPython.display import clear_output

from matplotlib.patches import Rectangle
from matplotlib.gridspec import GridSpec
import matplotlib.pyplot as plt

SYSTEM = platform.system()


def plot_automaton(automaton, file_name_no_extension):
    tmp_file = 'util\\tmp_aut.dot'
    with open(tmp_file, 'w') as tmp:
        tmp.write('digraph G {\n')

        for state in automaton.forbidden:
            tmp.write('\t"{}" [shape=box, color=red];\n'.format(state))

        for state in automaton.marked - automaton.forbidden:
            tmp.write('\t"{}" [shape=ellipse];\n'.format(state))

        for state in automaton.states - (automaton.marked | automaton.forbidden):
            tmp.write('\t"{}" [shape=plaintext];\n'.format(state))

        for transition in automaton.trans:
            tmp.write('\t"{}" -> "{}" [label="{}"];\n'.format(transition.source, transition.target, transition.event))

        tmp.write('\tinit [shape=plaintext, label=""];\n')
        tmp.write('\tinit -> "{}";\n'.format(automaton.init))

        tmp.write('}')

    pic = create_image(tmp_file, file_name_no_extension)
    os.remove(tmp_file)
    return pic


def plot_transitionsystem(system, file_name_no_extension):
    tmp_file = 'util\\tmp_aut.dot'
    with open(tmp_file, 'w') as tmp:
        tmp.write('digraph G {\n')

        for state in system.states:
            tmp.write('\t"{}" [shape=ellipse, label="{}"];\n'.format(state.name, state.labels))
        
        for i, state in enumerate(system.init):
            tmp.write('\tinit{} [shape=plaintext, label=""];\n'.format(i))
            tmp.write('\tinit{} -> "{}";\n'.format(i, state.name))

        for t in system.trans:
            tmp.write('\t"{}" -> "{}" [label="{}"];\n'.format(t.source.name, t.target.name, t.event))

        tmp.write('}')

    pic = create_image(tmp_file, file_name_no_extension)
    os.remove(tmp_file)
    return pic


def plot_petrinet(petrinet, file_name_no_extension):
    tmp_file = 'util\\tmp_net.dot'
    with open(tmp_file, 'w') as tmp:
        tmp.write('digraph G {\n')

        for place in petrinet.places:
            if place.marking <= 3:
                tokens = '&#9679;' * place.marking + '\\n'
            else:
                tokens = '&#9679;*' + str(place.marking) + '\\n'
            tmp.write('\t"{}" [shape=circle, label="{}"];\n'.format(place.label, tokens + place.label))

        for trans in petrinet.transitions:
            tmp.write(
                '\t"{}" [shape=rectangle, style=filled, fillcolor=grey, fixedsize=true, height=0.2, width=0.6, label="{}"];\n'.format(
                    trans, trans))

        for arc in petrinet.arcs:
            tmp.write(
                '\t"{}" -> "{}" [label="  {}"];\n'.format(arc.source, arc.target, arc.weight if arc.weight > 1 else ""))

        tmp.write('}')

    pic = create_image(tmp_file, file_name_no_extension)
    os.remove(tmp_file)
    return pic


def plot_digraph(digraph, file_name_no_extension):
    tmp_file = 'util\\tmp_graph.dot'
    with open(tmp_file, 'w') as tmp:
        tmp.write('digraph G {\n')

        tmp.write('\t"{}" [shape=box];\n'.format(digraph.init))

        for node in digraph.nodes - {digraph.init}:
            tmp.write('\t"{}" [shape=plaintext];\n'.format(node))

        for edge in digraph.edges:
            tmp.write('\t"{}" -> "{}" [label="  {}"];\n'.format(edge.source, edge.target, edge.label))

        tmp.write('}')

    pic = create_image(tmp_file, file_name_no_extension)
    os.remove(tmp_file)
    return pic


def create_image(dot_file, image_name):
    if SYSTEM == 'Windows':
        exe = 'util\\dot'
    else:
        try:
            subprocess.run(['dot', '-V'], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=True)
        except Exception as e:
            if SYSTEM == 'Darwin':
                raise OSError('It seems your version of MacOS does not have Graphviz installed!\n'
                                        'You have 3 options now:\n'
                                        '\t-Follow the instructions on http://graphviz.org/download/\n'
                                        '\t-Get https://www.anaconda.com/download/#macos and try "$ conda install graphviz"\n'
                                        '\t-Use another computer running Windows or Linux')
            else:
                raise OSError('Graphviz not found! Try: $ sudo apt install graphviz')
        else:
            exe = 'dot'

    try:
        subprocess.run([exe, '-Tpng', dot_file, '-o{}.png'.format(image_name)], stdout=subprocess.PIPE,
                       stderr=subprocess.STDOUT, check=True)
    except subprocess.CalledProcessError as e:
        raise ChildProcessError(e.stdout)

    return Image('{}.png'.format(image_name))


def shift_coordinate(state):
    x, y = state
    return x - .5, y - .5


def set_up_GBG(w, h, jarntorget, chalmers, home, obstacles, gota_alv):
    fig = plt.figure(figsize=(6, 9))
    gs = GridSpec(4, 3, figure=fig)
    
    ax = fig.add_subplot(gs[1:, :])
    ax.set_aspect(w/h)
    ax.set_title('WindyGothenburg')
    ax.set_xlim(-.5, w + .5)
    ax.set_ylim(-.5, h + .5)
    ax.hlines(range(h+1), -.5, w+1, color='grey', linewidth=.5)
    ax.vlines(range(w+1), -.5, h+1, color='grey', linewidth=.5)
    ax.annotate('JT', shift_coordinate(jarntorget), fontsize=25, color='r')
    ax.add_patch(Rectangle(shift_coordinate((0, h)), 2, 1, color='grey'))
    if chalmers:
        ax.annotate('P', shift_coordinate(chalmers), fontsize=25, color='b')
    ax.annotate('H', shift_coordinate(home), fontsize=25, color='g')
    ax.add_patch(Rectangle(shift_coordinate((w, h)), 1, 1, color='grey'))
    for tile in gota_alv:
        ax.add_patch(Rectangle(shift_coordinate(tile), 1, 1, color='b')) 
    for tile in obstacles:
        ax.add_patch(Rectangle(shift_coordinate(tile), 1, 1, color='k'))  
    grid_world = ax

    ax = fig.add_subplot(gs[0, :])
    ax.grid(True)
    ax.set_xlabel('Episode')
    ax.xaxis.set_label_position('top') 
    ax.xaxis.tick_top()
    ax.set_ylabel('Average Reward')
    ax.set_ylim(-2, 8)
    scatter_plot = ax
    
    return fig, grid_world, scatter_plot


def argmax_Q(Q, state):
    max_q = float("-inf")
    argmax_q = None
    for a, q in Q[state].items():
        if q > max_q:
            max_q = q
            argmax_q = a
    return argmax_q


def plot_Q(ax, Q, w, h):
    x = y = np.arange(0, h + 1, 1)
    X, Y = np.meshgrid(x, y)
    V = np.array([max(Q[(i, j)].values()) if Q.get((i, j)) else 0 for j in range(h+1) for i in range(w+1)])
    V = V.reshape(X.shape)
    Q_contours = ax.contourf(X, Y, V, levels=20, vmin=-10, vmax=90, cmap='RdYlGn', alpha=.5)
    labels = ax.clabel(Q_contours, inline=True, fontsize=10)
    symbls = {'north': 6, 'east': 5, 'south': 7, 'west': 4}
    policy_markers = [ax.scatter(*x, color='g', marker=symbls[argmax_Q(Q, x)]) for x in Q.keys()]
    return Q_contours, labels, policy_markers


def plot_pos(ax, pos):
    return ax.annotate('\N{sleeping face}',shift_coordinate(pos), fontsize=25)


def clear_plot(contours, labels, policy_markers):
    clear_output(wait=True)
    for c in contours.collections:
        c.remove()
    for l in labels:
        l.remove()
    for m in policy_markers:
        m.remove()


def plot_average_r(ax, avg_r, avg_r_smoothed, eps):
    ax.plot(eps, avg_r_smoothed, 'r_')
    return ax.plot(eps, avg_r, 'b.')


def plot_policy_evaluation(fig, ax, results, X, Y, results_name, x_name=None, y_name=None,
                           log_y=False, vmax=1.0, cmap='RdYlGn'):
    CS = ax.contourf(X, Y, results, levels=20,
                vmin=0, vmax=vmax, origin='lower', cmap=cmap)

    if x_name:
        ax.set_xlabel(x_name)
    
    if log_y:
        ax.set_yscale('log')
    plt.ylim([min(Y), max(Y)])
    
    if log_y and y_name:
        ax.set_ylabel('log({})'.format(y_name))
    elif y_name:
        ax.set_ylabel(y_name)

    # Make a colorbar for the ContourSet returned by the contourf call.
    cbar = fig.colorbar(CS, ax=ax)
    cbar.ax.set_ylabel(results_name)


def plot_heatmaps(stats, X, Y, metrics, x_name, y_name, save_name, title, log_y=False):
    stats = [[s[m] for m in metrics] for s in stats]
    stats = np.reshape(stats,(len(X), len(Y), len(metrics)))
    stats = np.transpose(stats, (1, 0, 2))
    fig, ((ax1, ax2), (ax3, ax4))  = plt.subplots(2, 2, sharex=True, sharey=True, 
                                                  figsize=(9, 6)) #, layout='constrained'
    
    plot_policy_evaluation(fig, ax1, stats[:,:,0], X, Y, 'Avg. Success Rates', y_name=y_name,
                             log_y=log_y, vmax=1.0)
    
    plot_policy_evaluation(fig, ax3, stats[:,:,1], X, Y, 'Avg. Episode Goal Found', x_name, y_name,
                             log_y=log_y, vmax=250, cmap='RdYlGn_r')
    
    plot_policy_evaluation(fig, ax4, stats[:,:,2], X, Y, 'Avg. Episode Goal Learned', x_name=x_name, 
                             log_y=log_y, vmax=250, cmap='RdYlGn_r')
    
    plot_policy_evaluation(fig, ax2, stats[:,:,3], X, Y, 'Avg. Max R Smoothed', log_y=log_y, vmax=6.0)
    
    fig.suptitle(title)
    if save_name:
        plt.savefig('fig/{}.png'.format(save_name))
    return fig