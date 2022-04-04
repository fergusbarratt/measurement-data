import numpy as np

def find_clusters_bond(grid, verbose=False):
    """
    Find clusters of sites, from a grid corresponding to locations of percolating bonds
    """
    
    grid = np.pad(grid, pad_width = ((0, 0), (1, 2)), constant_values = ((0, 0), (1, 1)))
    num_of_ones = (grid.shape[0]+1)*((grid.shape[1]+1)//2)
    
    # 1-D array of labels of all sites.
    ids = np.arange(num_of_ones).reshape(grid.shape[0]+1, ((grid.shape[1]+1)//2), order='F')
    ids[:, 0] = -1
    ids[:, -2] = -200
    ids[:, -1] = -200


     # a dictionary of cluster ids -> sites in cluster
    clusters = {a:set([tuple(x) for x in np.argwhere(np.isclose(ids, a))]) for a in ids.reshape(-1)}
    
    # 2-D array storing (y,x) coordinates of occupied bonds
    coords = [list(x) for x in np.argwhere(grid>0)]
    i = 0
    for coord in coords:
        bondy, bondx = coord
        
        # even rows
        if bondy%2:
            site1 = bondy, (1+bondx)//2
            site2 = bondy+1, (bondx)//2
            id1 = ids[site1]
            id2 = ids[site2]
        else:
            # odd rows
            site1 = bondy, (bondx)//2
            site2 = bondy+1, (1+bondx)//2
            id1 = ids[site1]
            id2 = ids[site2]
            
        if id1==id2:
            # already in the same cluster. 
            if verbose:
                print('already in the same cluster')
            pass
        else:
            if verbose:
                print('different clusters')
                
            # get the ids of the clusters on either side of the bond
            smaller_cluster_id, larger_cluster_id = [id1, id2][::(-1)**int(len(clusters[id1]) > len(clusters[id2]))]
            if verbose:
                print('clusters have size', id1, ':', len(clusters[id1]), ',', id2, ':', len(clusters[id2]), ', smaller_cluster has size', len(clusters[smaller_cluster_id]), 'changing', smaller_cluster_id, 'to', larger_cluster_id)
            for site_coord in clusters[smaller_cluster_id]:
                # set all the tags in the smaller cluster to those in the larger cluster
                ids[site_coord] = larger_cluster_id
                
            # join the smaller cluster to the larger cluster, and delete the smaller cluster
            clusters[larger_cluster_id] = clusters[larger_cluster_id].union(clusters[smaller_cluster_id])
            del clusters[smaller_cluster_id]
            
            if verbose:
                print()
    # return a dictionary mapping unique integer cluster ids to all of the points in that cluster, 
    # and the grid of sites, each site having its cluster id            
    return clusters, ids

def is_percolation(ids):
    return ids[0, 0]==ids[0, -1]
