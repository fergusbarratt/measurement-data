import networkx as nx
import numpy as np
from measurement_analysis.unmeasured_percolation import iterated_fill_out


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
    top_right_node = list(sorted(G.nodes()))[-1]
    nx.set_node_attributes(G, {top_right_node:G.nodes()[top_right_node]['pos']+np.array([1/2, 0])}, 'pos')
    return G, edge_coord_map


def get_array(graph, edge_map):
    edge_attrs = nx.get_edge_attributes(graph, 'charge')
    locs = sorted(list(edge_map.values()))
    arr = np.zeros(shape=np.array(locs[-1])+1) #make the array
    for edge, loc in edge_map.items():
        if edge in edge_attrs:
            arr[loc] = edge_attrs[edge]
        else:
            arr[loc] = edge_attrs[edge[::-1]]

    return arr

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


def complete_cycles(grid, length=0):
    """Complete unmeasured sharp sites using graph cycles. Still technically polynomial....
    Right now returns all the long path ways of making a single link.
    """
    G, edge_map = get_graph(grid)
    long_paths = []
    charges = []
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
        if weight==0:
            B_path = nx.shortest_path(G, site1, site2, weight='weight')
            B = nx.shortest_path_length(G, site1, site2, weight='weight')
            if B == length:
                long_paths.append(B_path)
                pos=nx.get_node_attributes(G,'pos')
                sign = -2*(pos[site1][0]-pos[site2][0])
                charge = sign*get_charge(B_path, G, pos)[0]
                charges.append(charge)
                nx.set_edge_attributes(G, {(site1, site2): {"weight": 1-charge, "charge":2*charge-1}})
    return get_array(G, edge_map), long_paths, charges

def in_which_path(node, paths):
    return np.argmax([node in path for path in paths])

    
def plot_global_lattice(record, ax, which_colors='loops', paths=None):
    G, edge_map = get_graph(record)

    pos=nx.get_node_attributes(G,'pos')

    top_right_node = list(sorted(G.nodes()))[-1]
    
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

def graph_predict(record):
    record = np.pad(record, pad_width = ((0, 0), (1, 2)), constant_values = ((0, 0), (-1, -1)))
    record = iterated_fill_out(np.copy(record))

    G, _ = get_graph(record)
    pos=nx.get_node_attributes(G,'pos')

    top_right_node = list(sorted(G.nodes()))[-1]
    shortest_path = nx.dijkstra_path(G, (0, 0), top_right_node)
    min_charge, var = get_charge(shortest_path, G, pos)
    
    return min_charge, var

def test_getarray():
    arecord = np.random.randint(0, 2, size=(36, 12))
    G, edge_map = get_graph(arecord)
    r2 = get_array(G, edge_map)
    assert np.allclose(arecord, r2)
    
N = 10
for _ in range(N):
    test_getarray()
