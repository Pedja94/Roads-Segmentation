import numpy as np

def flatten(l):
    return [item for sublist in l for item in sublist]


def remove_small_terminal(G, weight='weight', min_weight_val=30, 
                          pix_extent=1300, edge_buffer=4, verbose=False):
    '''Remove small terminals, if a node in the terminal is within edge_buffer
    of the the graph edge, keep it'''
    deg = dict(G.degree())
    terminal_points = [i for i, d in deg.items() if d == 1]
    if verbose:
        print("remove_small_terminal() - N terminal_points:", len(terminal_points))
    edges = list(G.edges())
    for s, e in edges:
        if s == e:
            sum_len = 0
            vals = flatten([[v] for v in G[s][s].values()])
            for ix, val in enumerate(vals):
                sum_len += len(val['pts'])
            if sum_len < 3:
                G.remove_edge(s, e)
                continue
            
        # check if at edge
        sx, sy = G.nodes[s]['o']
        ex, ey = G.nodes[e]['o']
        edge_point = False
        for ptmp in [sx, sy, ex, ey]:
            if (ptmp < (0 + edge_buffer)) or (ptmp > (pix_extent - edge_buffer)):
                if verbose:
                    print("ptmp:", ptmp)
                    print("(pix_extent - edge_buffer):", (pix_extent - edge_buffer))
                    print("(ptmp > (pix_extent - edge_buffer):", (ptmp > (pix_extent - edge_buffer)))
                    print("ptmp < (0 + edge_buffer):", (ptmp < (0 + edge_buffer)))
                edge_point = True
            else:
                continue
        # don't remove edges near the edge of the image
        if edge_point:
            if verbose:
                print("(pix_extent - edge_buffer):", (pix_extent - edge_buffer))
                print("edge_point:", sx, sy, ex, ey, "continue")
            continue

        val = G[s][e]
        if verbose:
            print("val.get(weight, 0):", val.get(weight, 0) )
        if s in terminal_points and val.get(weight, 0) < min_weight_val:
            G.remove_node(s)
        if e in terminal_points and val.get(weight, 0) < min_weight_val:
            G.remove_node(e)

    return G


def cleanUpSmallEdges(G, min_spur_length_pix = 10, pix_extent = 256, verbose = False):

    for itmp in range(8):
        ntmp0 = len(G.nodes())
        if verbose:
            print("Clean out small terminals - round", itmp)
            print("Clean out small terminals - round", itmp, "num nodes:", ntmp0)

        # sknw attaches a 'weight' property that is the length in pixels
        remove_small_terminal(G, weight='weight',
                                min_weight_val=min_spur_length_pix,
                                pix_extent=pix_extent)
        # kill the loop if we stopped removing nodes
        ntmp1 = len(G.nodes())
        if ntmp0 == ntmp1:
            break
        else:
            continue

    return G