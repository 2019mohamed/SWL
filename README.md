# SWL
Simplicial Weisfeiler and Lehman for graph isomorphism test
from [Weisfeiler and Lehman Go Topological: Message Passing Simplicial Networks](https://arxiv.org/pdf/2103.03212.pdf)

perfect graphs with simpilical complex(i.e., cliques)
![GitHub Logo](/images/perfect1.png)
![GitHub Logo](/images/perfect2.png)
![GitHub Logo](/images/perfect3.png)

(Very simple survay)
In Message Passing Simplicial Networks(i.e., hight dim.) or Message Passing framwork, we have to define aggregation operator over neighbors of vertex. So, here, aggregation over faces and cofaces. we can aggrate over a lot of neighbors sets to get expressive poweful operator. However, SWL > 3-WL, rGIN model(i.e., GIN with random vertex features) >= 1-WL. Also, there exists "Graph Neural Network Expressivity via Subgraph Isomorphism Counting" model(i.e., GIN with structual vertex or edge features) > 3-WL, and "CRaWl (Convolutional Neural Networks for Random Walks)" model >= k-WL >= k-GNN !!! The story is not over... 


