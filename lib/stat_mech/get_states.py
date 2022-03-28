import quimb.tensor as qtn
import quimb as qu
import pickle
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from pathlib import Path
import networkx as nx
from functools import reduce

from tebdm import TEBDm
import gc

import joblib
from read_states import get_correlations

def get_graph(grid):
    """Get a grid of measurement outcomes and make the dual lattice"""
    G = nx.Graph()
    edge_coord_map = {}
    for coord, weight in np.ndenumerate(grid):
        bondy, bondx = coord

        # even rows
        if bondy%2:
            site1 = bondy, (1+bondx)//2
            site2 = bondy+1, (bondx)//2
        else:
            # odd rows
            site1 = bondy, (bondx)//2
            site2 = bondy+1, (1+bondx)//2
            
        edge_coord_map[(site1, site2)] = (bondy, bondx)
        
        G.add_node(site1, pos=(np.array(site1)+np.array([0, (1/2)*(bondx%2)+(1/2)*(bondx==grid.shape[1]-1)*(1-bondy%2)]))[::-1])
        G.add_node(site2, pos=(np.array(site2)+np.array([0, (1/2)*(bondx%2)+(1/2)*(bondx==grid.shape[1]-1)*(1-bondy%2)]))[::-1])
            
        G.add_edge(site1, site2, weight=np.abs(1-np.abs(weight)), charge=weight)
    top_right_node = list(sorted(G.nodes))[-1]
    nx.set_node_attributes(G, {top_right_node:G.nodes[top_right_node]['pos']+np.array([1/2, 0])}, 'pos')
    return G, edge_coord_map

def break_path(path, G, pos, verbose=False):
    """
    break a path into pieces every time it crosses a non measured (charge 0) link. Get the charges for each path. 
    Args:
        path: list of nodes
        G: nx.Graph
        pos: Positions of the nodes.
    Returns:
        (list, list): the paths making up path, and the charges of each fragment
    """
    pathsum = 0
    inc_path = []
    paths = []
    charges = []
    for i in range(len(path)-1):
        data = G.get_edge_data(path[i],path[i+1])
        charge = (1+data['charge'])/2
        inc_path.append(path[i])
        if charge not in [0., 1.]:
            paths.append(inc_path)
            charges.append(pathsum)
            inc_path = []
            pathsum = 0
            continue
        sign = -2*(pos[path[i]][0]-pos[path[i+1]][0])
        if verbose:
            print(sign, pos[path[i]], pos[path[i+1]])
        pathsum += sign*charge
    inc_path.append(path[-1])
    paths.append(inc_path)
    charges.append(pathsum)
    return paths, charges

def get_charge(path, G, pos, verbose=False):
    """
    Get the charge of a path in G, by adding right moving +1 links with sign 1 and left moving +1 links with sign -1. 
    Args:
        path: list of nodes
        G: nx.Graph
        pos: Positions of the nodes.
    Returns:
        Integer, Integer: The estimated charge, and the number of unknown links on the path. (estimated charge is a lower bound, and estimated charge + unknown is an upper bound).
    """
    n_unknowns = 0
    pathsum = 0
    for i in range(len(path)-1):
        data = G.get_edge_data(path[i],path[i+1])
        charge = (1+data['charge'])/2
        if charge not in [0., 1.]:
            n_unknowns+=1
            continue
        sign = -2*(pos[path[i]][0]-pos[path[i+1]][0])
        if verbose:
            print(sign, pos[path[i]], pos[path[i+1]])
        pathsum += sign*charge
    return pathsum, n_unknowns

def plot_global_lattice(record, ax, which_colors='path', paths=None):
    G, edge_map = get_graph(record)

    pos=nx.get_node_attributes(G,'pos')

    top_right_node = list(sorted(G.nodes))[-1]
    
    def weight_fn(site1, site2, attrs):
        """
        returns the path we're most certain about. 1 for unmeasured links, 0 for measured links.
        """
        return attrs['weight']
    
    def high_charge_fn(site1, site2, attrs):
        """
        returns the path with the highest (definite) charge
        """
        rl = (1-2*(pos[site1][0]-pos[site2][0]))/2 # is the link right moving or left moving? (1 for left moving, 0 for right moving)
        return (0+rl if attrs['charge'] == 1 else 1) #(0 for right moving +1 measurements, 1 for everything else)
        
        
    fn = weight_fn
    shortest_path = nx.shortest_path(G, (0, 0), top_right_node, weight=fn) # This path is the one we're most certain about. 

    broken_path = break_path(shortest_path, G, pos)[0]


    def in_which_path(node, paths):
        return np.argmax([node in path for path in paths])

    if which_colors =='uniform':
    #uniform
        node_colors = ["lightgray" for n in G.nodes()]
    elif which_colors=='path':
    # to color according to shortest path
        node_colors = [f"C{in_which_path(n, broken_path)}" if n in shortest_path else "lightgray" for n in G.nodes()]
    elif which_colors=='loops':
    # color filled in paths
        node_colors = [f"C{in_which_path(n, paths)}" if n in reduce(lambda x, y:x+y, paths, []) else "lightgray" for n in G.nodes()]

    labels = nx.get_edge_attributes(G,'charge')

    edge_colors = ['skyblue' if G.get_edge_data(*edge)['charge']< 0 else 'indianred' if G.get_edge_data(*edge)['charge']> 0 else 'black' for edge in G.edges()]
    nx.draw_networkx_edges(G, pos, ax=ax, edge_color=edge_colors, width=6, alpha=1);

    nx.draw_networkx_nodes(G,pos, node_color=node_colors, node_size=60, ax=ax, alpha=1);

    ax.grid(visible=None)
    ax.set_aspect(aspect='auto')
    charge, var = get_charge(shortest_path, G, pos)
    ax.set_title(f'{charge} (+{var})');#, {round(np.mean(np.abs(record)),3)}');

plt.style.use('seaborn-whitegrid')

def state(N, Q) :
    psi = np.array([int(bin(n).count("1")==Q) for n in range(2**N)])
    return qu.qu(psi/np.sum(psi))

def evolve_state(p, L, T, Q, record, data_dir):
    Q = L//2-Q
    summer = qtn.tensor_gen.MPS_product_state([np.array([1.0, 1.0])] * L)
    #psi0 = summer.copy(deep=True) / (summer.H @ summer)  # start from all + state
    astate = state(L, Q)
    psi0 = qtn.MatrixProductState.from_dense(astate, [2]*L)

    tebd = TEBDm(p, psi0, H=np.zeros((4, 4)))

    tebd.split_opts["cutoff"] = 1e-8 
    be_b = []

    #print(record)
    try:
        for psit in tebd.at_times(np.arange(T), measurement_locations=record, progbar=False, tol=1e-3):    
            #be_b += [psit.entropy(L//2)]    
            pass
    except Exception:
        # no trajectories are compatible with this charge value. 
        return 0
    
    final_state = tebd.pt
    final_state.entanglement_history = np.array(be_b)
    final_state.norm = tebd.norm
    return np.prod(final_state.norm)

    #mat = get_correlations(final_state) 
    #os.makedirs(data_dir, exist_ok=True)
    #os.makedirs(data_dir + '/states', exist_ok=True)

    #qid = qtn.tensor_core.rand_uuid()
    ##qu.save_to_disk(final_state, data_dir + f'/states/{p},{L},{T},{qid}.dmp')
    #del final_state
    #del summer
    #del psi0
    #del tebd
    #del be_b
    #gc.collect()

    #return True

if __name__ == '__main__':
    Ls = [18]
    T = lambda L: 2*L
    data_dir = f'data'
    dirpaths = sorted(list(Path('../../processed_data/').glob('*.npy')))

    K = 100
    results = {}
    for dirpath in dirpaths[:1]:
        for Q in [0, 1]:
            arecord = np.load(dirpath)
            N_samples, _, L, T = arecord.shape
            p = str(dirpath.stem).split(',')[-1]
            T = (T-1)//2

            records = np.load(dirpath)[:K, Q, :, :-1]
            mats0 = joblib.Parallel(n_jobs=-1)(joblib.delayed(evolve_state)(0, L, T, 0, record.T, data_dir) for record in tqdm(records))
            mats1 = joblib.Parallel(n_jobs=-1)(joblib.delayed(evolve_state)(0, L, T, 1, record.T, data_dir) for record in tqdm(records))
            acc = np.mean(np.array(mats1)>np.array(mats0))
            results[(Q, L, p)] = acc


    pickle.dump(results, open('../../results', 'wb'))
    print(results)
    print('done, moving folder')
