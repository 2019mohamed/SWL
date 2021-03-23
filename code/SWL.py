# -*- coding: utf-8 -*-
"""
Created on Mon Mar  1 22:19:01 2021

@author: M
"""

import gudhi
import itertools
from hashlib import blake2b
from collections import Counter
import dgl
import networkx as nx
import numpy as np
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
from dgl.nn import SumPooling
from scipy.spatial import distance
from dgl.nn.pytorch import GraphConv , GINConv ,SAGEConv ,GATConv


def draw_2d_simplicial_complex(simplices, pos=None, return_pos=False, ax = None):
    """
    Draw a simplicial complex up to dimension 2 from a list of simplices, as in [1].
        
        Args
        ----
        simplices: list of lists of integers
            List of simplices to draw. Sub-simplices are not needed (only maximal).
            For example, the 2-simplex [1,2,3] will automatically generate the three
            1-simplices [1,2],[2,3],[1,3] and the three 0-simplices [1],[2],[3].
            When a higher order simplex is entered only its sub-simplices
            up to D=2 will be drawn.
        
        pos: dict (default=None)
            If passed, this dictionary of positions d:(x,y) is used for placing the 0-simplices.
            The standard nx spring layour is used otherwise.
           
        ax: matplotlib.pyplot.axes (default=None)
        
        return_pos: dict (default=False)
            If True returns the dictionary of positions for the 0-simplices.
            
        References
        ----------    
        .. [1] I. Iacopini, G. Petri, A. Barrat & V. Latora (2019)
               "Simplicial Models of Social Contagion".
               Nature communications, 10(1), 2485.
    """
    #List of 0-simplices
    nodes =list(set(itertools.chain(*simplices)))
    
    #List of 1-simplices
    edges = list(set(itertools.chain(*[[tuple(sorted((i, j))) for i, j in itertools.combinations(simplex, 2)] for simplex in simplices])))

    #List of 2-simplices
    triangles = list(set(itertools.chain(*[[tuple(sorted((i, j, k))) for i, j, k in itertools.combinations(simplex, 3)] for simplex in simplices])))
    
    if ax is None: ax = plt.gca()
    ax.set_xlim([-1.1, 1.1])      
    ax.set_ylim([-1.1, 1.1])
    ax.get_xaxis().set_ticks([])  
    ax.get_yaxis().set_ticks([])
    ax.axis('off')
       
    if pos is None:
        # Creating a networkx Graph from the edgelist
        G = nx.Graph()
        G.add_edges_from(edges)
        # Creating a dictionary for the position of the nodes
        pos = nx.spring_layout(G)
        
    # Drawing the edges
    for i, j in edges:
        (x0, y0) = pos[i]
        (x1, y1) = pos[j]
        line = plt.Line2D([ x0, x1 ], [y0, y1 ],color = 'black', zorder = 1, lw=0.7)
        ax.add_line(line)
    # Filling in the triangles
    for i, j, k in triangles:
        (x0, y0) = pos[i]
        (x1, y1) = pos[j]
        (x2, y2) = pos[k]
        tri = plt.Polygon([ [ x0, y0 ], [ x1, y1 ], [ x2, y2 ] ],
                          edgecolor = 'black', facecolor = plt.cm.Blues(0.6),
                          zorder = 2, alpha=0.4, lw=0.5)
        ax.add_patch(tri);

    # Drawing the nodes 
    for i in nodes:
        (x, y) = pos[i]
        circ = plt.Circle([ x, y ], radius = 0.02, zorder = 3, lw=0.5,
                          edgecolor = 'Black', facecolor = u'#ff7f0e')
        ax.add_patch(circ);

    if return_pos: return pos

class List:

    def __init__(self, l):
        self.l = l
        self.even_k = ''+str(len(self.l))
        self.odd_k = ''
        
        
    def __eq__(self, other):
        if len(self.l) !=  len(other.l):
            return 0
        
        for i,j in zip(self.l , other.l):
            if i != j:
                return i != j
        
        return 1
        
    def __hash__(self):
        '''
        h = 1
        for n in self.l:
            h += hash(n)
        '''
        return hash(frozenset(self.l))
               
    def __str__(self):
        return str(self.l)  
    
    
def nei_agg (adjList , index_sim ,m , inv_m ,  even = True):
    x = []
    for adj_index in adjList:
        if even:
            x.append(inv_m[index_sim].odd_k)
            x.append(inv_m[adj_index].odd_k)
        else:
            x.append(inv_m[index_sim].even_k)
            x.append(inv_m[adj_index].even_k)
    
    x = sorted(x)
    if even:
        inv_m[index_sim].even_k += str(x)
    else:
        inv_m[index_sim].odd_k += str(x)
            
    

def SWL (adj , m , inv_m , K):
    items = []
    for k in range(1 , K+1):
        for index , sim in inv_m.items():
            nei_agg(adj[index] , index , m , inv_m , k % 2 == 0 )
        
        c = Counter()
        # count node labels
        if K %2 == 0:
            for d in inv_m.values():
                h = blake2b(digest_size=16)
                h.update(d.even_k.encode('ascii'))
                c.update([h.hexdigest()])
        else:
            for d in inv_m.values():
                h = blake2b(digest_size=16)
                h.update(d.odd_k.encode('ascii'))
                c.update([h.hexdigest()])           
            
        # sort the counter, extend total counts
        items.extend(sorted(c.items(), key=lambda x: x[0]))
        
    h = blake2b(digest_size=16)
    h.update(str(tuple(items)).encode('ascii'))
    h = h.hexdigest()
    return h
      
'''    
     >>> G1 = nx.Graph()
    >>> G1.add_edges_from([(1, 2, {'label': 'A'}),\
                           (2, 3, {'label': 'A'}),\
                           (3, 1, {'label': 'A'}),\
                           (1, 4, {'label': 'B'})])
    >>> G2 = nx.Graph()
    >>> G2.add_edges_from([(5,6, {'label': 'B'}),\
                           (6,7, {'label': 'A'}),\
                           (7,5, {'label': 'A'}),\
                           (7,8, {'label': 'A'})])    
'''


def Index_ST (st):
    
    m = dict()
    inv_m = dict()
    adj = dict()
        
    for index ,  s in enumerate(list(st.get_simplices())):
        sim = List(s[0])
        m[sim] = index
        inv_m [ index] = sim
    
    for index ,  s in enumerate(list(st.get_simplices())):
        N1 = st.get_star(s[0])
        N2 = list(itertools.combinations(s[0],len(s[0])-1))
        adj[index] = list()
        for p1 , p2 in N1:
            sim = List(p1)
            adj[index].append(m[sim])
            
        for p in N2:
            if len(p) > 0:
                sim = List(list(p))
                adj[index].append(m[sim])
    
    return m , inv_m , adj

def Build_Cliques(G):
    cliques = list(nx.algorithms.clique.enumerate_all_cliques(G))
    st = gudhi.SimplexTree()
    
    for clique in cliques:
        st.insert(clique)
    
    return st
    
def Draw_SC (simplices):
    plt.figure(figsize=(10,10))
    #ax = plt.subplot(111)
    draw_2d_simplicial_complex(simplices)
    plt.show()


K = 1
'''
st.insert([1,2])
st.insert([1,3])
st.insert([3,4])
st.insert([4,5])
st.insert([5,6])
st.insert([6,2])
st.insert([1,7])
st.insert([7,8])
st.insert([8,9])
st.insert([9,10])
st.insert([10,2])
'''
G1 = nx.Graph()

G1.add_edge(1,2)
G1.add_edge(1,3)
G1.add_edge(3,4)
G1.add_edge(4,5)
G1.add_edge(5,6)
G1.add_edge(6,2)
G1.add_edge(1,7)
G1.add_edge(7,8)
G1.add_edge(8,9)
G1.add_edge(9,10)
G1.add_edge(10,2)

'''
G1.add_edge(1,2)
G1.add_edge(2,5)
G1.add_edge(2,6)
G1.add_edge(5,6)
G1.add_edge(1,4)
G1.add_edge(1,3)
G1.add_edge(3,4)
'''
st = Build_Cliques(G1)



'''
st1.insert([1,2])
st1.insert([2,3])
st1.insert([3,4])
st1.insert([4,5])
st1.insert([5,6])
st1.insert([6,2])
st1.insert([1,7])
st1.insert([7,8])
st1.insert([8,9])
st1.insert([9,10])
st1.insert([10,1])
'''
G2 = nx.Graph()

G2.add_edge(1,2)
G2.add_edge(2,3)
G2.add_edge(3,4)
G2.add_edge(4,5)
G2.add_edge(5,6)
G2.add_edge(6,2)
G2.add_edge(1,7)
G2.add_edge(7,8)
G2.add_edge(8,9)
G2.add_edge(9,10)
G2.add_edge(1,10)


'''
G2.add_edge(1,2)
G2.add_edge(2,3)
G2.add_edge(4,5)
G2.add_edge(5,6)
G2.add_edge(1,4)
G2.add_edge(2,5)
G2.add_edge(3,6)
'''
st1 = Build_Cliques(G2)


m1 , inv1 , adj1 = Index_ST(st)
SWL1 = SWL(adj1, m1, inv1, K)

m2 , inv2 , adj2 = Index_ST(st1)
SWL2 = SWL(adj2, m2, inv2, K)

print(SWL1 == SWL2)
'''
if K%2 == 0:
    for index , sim in inv_m.items():
        for index1 , sim1 in inv_m.items():
            if sim.even_k == sim1.even_k and index != index1:
                print(sim,' ',sim1)
else:
    for index , sim in inv_m.items():
        for index1 , sim1 in inv_m.items():
            if sim.odd_k == sim1.odd_k and index != index1:
                print(sim,' ',sim1)

'''
print('-------------------------------------------------------')
Gs = nx.read_graph6('C:\\Users\\M\\Downloads\\perf7.g6')
print(len(Gs))  

'''
for g in Gs:
    st = Build_Cliques(g)
    SC = list()
    for sim in list(st.get_simplices()):
        SC.append(list(sim[0]))
    print(SC)
    try:
        Draw_SC(SC)
    except:
        print("Just graph!")
'''
def rGIN (g):
    g = dgl.from_networkx(g)
    f = np.random.standard_normal(size=(g.number_of_nodes(), 1))
    x = torch.tensor(f, dtype=torch.float)
    g.ndata['x'] = x
    lin = torch.nn.Linear(1, 1)
    conv = GINConv(lin, 'sum')
    res = conv(g, x)
    sumpool = SumPooling()
    return sumpool(g, res)[0].detach().numpy()


count = 0
SWL_Not_dis , rGIN_Not_dis = 0,0
for i , g1 in tqdm(enumerate(Gs)):
    for j , g2 in enumerate(Gs):
        if i < j and g1.number_of_edges() >0 and g2.number_of_edges() > 0:           
            count += 1
            st = Build_Cliques(g1)
            st1 = Build_Cliques(g2)
            m1 , inv1 , adj1 = Index_ST(st)
            SWL1 = SWL(adj1, m1, inv1, K)
            m2 , inv2 , adj2 = Index_ST(st1)
            SWL2 = SWL(adj2, m2, inv2, K)
            
            #ans1 = nx.algorithms.isomorphism.is_isomorphic(g1,g2)
            #print(ans1)
            SWL_Not_dis += (SWL1 == SWL2)
            rGIN_Not_dis += (distance.euclidean(rGIN(g1) , rGIN(g2)) < 0.01)
            #print(i,' ',j,' ',ans1,
                 # ' ',ans2,' ',ans1 == ans2)

print('SWL error rate',100*(SWL_Not_dis/count)
      ,' rGIN error rate',100*(rGIN_Not_dis/count))


