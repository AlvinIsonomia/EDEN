import networkx as nx
from graphgym.register import register_feature_augment

from sklearn.decomposition import PCA
import numpy as np
import torch

def example_node_augmentation_func(graph, **kwargs):
    '''
    compute node clustering coefficient as feature augmentation
    :param graph: deepsnap graph. graph.G is networkx
    :param kwargs: required, in case additional kwargs are provided
    :return: List of node feature values, length equals number of nodes
    Note: these returned values are later processed and treated as node
    features as specified in "cfg.dataset.augment_feature_repr"
    '''
    return list(nx.clustering(graph.G).values())

def EDEN_augmentation_func(graph, **kwargs):
    '''
    compute node clustering coefficient as feature augmentation
    :param graph: deepsnap graph. graph.G is networkx
    :param kwargs: required, in case additional kwargs are provided
    :return: List of node feature values, length equals number of nodes
    Note: these returned values are later processed and treated as node
    features as specified in "cfg.dataset.augment_feature_repr"
    '''
    print("Using EDEN!")
    Distance = dict(nx.all_pairs_shortest_path_length(graph.G))
    num_node = len(graph.G)
    D = np.zeros([num_node, num_node])
    for i in range(num_node):
        for j in range(num_node):
            try:
                D[i,j] = Distance[i][j]
            except:
                D[i,j] = np.NAN
    cos_dis = np.cos(np.pi*D/np.nanmax(D, axis=0, keepdims=True))
    cos_dis[np.isnan(cos_dis)]=-1.5
    PCA1 = PCA(n_components=kwargs['feature_dim'])
    PCA_dis = PCA1.fit_transform(cos_dis)
    PCA_dis /= np.std(PCA_dis, axis=0, keepdims=True)
    EDEN = list(PCA_dis)
    
    return EDEN

def DistancePCA_func(graph, **kwargs):
    '''
    compute node clustering coefficient as feature augmentation
    :param graph: deepsnap graph. graph.G is networkx
    :param kwargs: required, in case additional kwargs are provided
    :return: List of node feature values, length equals number of nodes
    Note: these returned values are later processed and treated as node
    features as specified in "cfg.dataset.augment_feature_repr"
    '''
    print("Only USE PCA Distance!")
    Distance = dict(nx.all_pairs_shortest_path_length(graph.G))
    num_node = len(graph.G)
    D = np.zeros([num_node, num_node])
    for i in range(num_node):
        for j in range(num_node):
            try:
                D[i,j] = Distance[i][j]
            except:
                D[i,j] = np.NAN
    
    D[np.isnan(D)] = -1
    PCA1 = PCA(n_components=kwargs['feature_dim'])
    PCA_dis = PCA1.fit_transform(D)
    PCA_dis /= np.std(PCA_dis, axis=0, keepdims=True)
    EDEN = list(PCA_dis)
    
    return EDEN

def MinMaxPCA_func(graph, **kwargs):
    '''
    compute node clustering coefficient as feature augmentation
    :param graph: deepsnap graph. graph.G is networkx
    :param kwargs: required, in case additional kwargs are provided
    :return: List of node feature values, length equals number of nodes
    Note: these returned values are later processed and treated as node
    features as specified in "cfg.dataset.augment_feature_repr"
    '''
    print("USE PCA Normalized Distance!")
    Distance = dict(nx.all_pairs_shortest_path_length(graph.G))
    num_node = len(graph.G)
    D = np.zeros([num_node, num_node])
    for i in range(num_node):
        for j in range(num_node):
            try:
                D[i,j] = Distance[i][j]
            except:
                D[i,j] = np.NAN
    D = (D- np.nanmin(D))/(np.nanmax(D-np.nanmin(D)))
    D[np.isnan(D)] = -1
    PCA1 = PCA(n_components=kwargs['feature_dim'])
    PCA_dis = PCA1.fit_transform(D)
    PCA_dis /= np.std(PCA_dis, axis=0, keepdims=True)
    EDEN = list(PCA_dis)
    
    return EDEN

def ReverseMinMaxPCA_func(graph, **kwargs):
    '''
    compute node clustering coefficient as feature augmentation
    :param graph: deepsnap graph. graph.G is networkx
    :param kwargs: required, in case additional kwargs are provided
    :return: List of node feature values, length equals number of nodes
    Note: these returned values are later processed and treated as node
    features as specified in "cfg.dataset.augment_feature_repr"
    '''
    print("Reverse Normalized PCA Distance!")
    Distance = dict(nx.all_pairs_shortest_path_length(graph.G))
    num_node = len(graph.G)
    D = np.zeros([num_node, num_node])
    for i in range(num_node):
        for j in range(num_node):
            try:
                D[i,j] = Distance[i][j]
            except:
                D[i,j] = np.NAN
    D = -(D- np.nanmin(D))/(np.nanmax(D)-np.nanmin(D))
    D += 1
    #cos_dis = np.cos(np.pi*D/np.nanmax(D, axis=0, keepdims=True))
    D[np.isnan(D)] = -1
    PCA1 = PCA(n_components=kwargs['feature_dim'])
    PCA_dis = PCA1.fit_transform(D)
    PCA_dis /= np.std(PCA_dis, axis=0, keepdims=True)
    EDEN = list(PCA_dis)
    
    return EDEN

def EDEN_augmentation_trick(graph, **kwargs):
    '''
    compute node clustering coefficient as feature augmentation
    :param graph: deepsnap graph. graph.G is networkx
    :param kwargs: required, in case additional kwargs are provided
    :return: List of node feature values, length equals number of nodes
    Note: these returned values are later processed and treated as node
    features as specified in "cfg.dataset.augment_feature_repr"
    '''
    print("Using EDEN! (Lagest Principal Component Excluded)")
    Distance = dict(nx.all_pairs_shortest_path_length(graph.G))
    num_node = len(graph.G)
    D = np.zeros([num_node, num_node])
    for i in range(num_node):
        for j in range(num_node):
            try:
                D[i,j] = Distance[i][j]
            except:
                D[i,j] = np.NAN
    cos_dis = np.cos(np.pi*D/np.nanmax(D, axis=0, keepdims=True))
    cos_dis[np.isnan(cos_dis)]=-1.5
    PCA1 = PCA(n_components=kwargs['feature_dim']+1)
    PCA_dis = PCA1.fit_transform(cos_dis)
    #PCA_dis /= np.std(PCA_dis, axis=0, keepdims=True)
    EDEN = list(PCA_dis[:,1:])
    
    return EDEN

def benchmark_func(graph, **kwargs):
    A = np.array(nx.adjacency_matrix(graph).todense())
    A = -A
    for i in range(len(A)):
        degree_i = graph.degree[i]
        A[i][i] = A[i][i] + degree_i  
    PCA1 = PCA(n_components=len(G))
    lap = PCA1.fit_transform(A)
    lap = lap[:, -kwargs['feature_dim']-1: -1] 
    return lap

def RNI_func(graph, **kwargs):
    A = np.array(nx.adjacency_matrix(graph).todense())
    node_num = len(A)
    rni = np.random.rand(node_num,kwargs['feature_dim'])
    return rni

register_feature_augment('example', example_node_augmentation_func)
register_feature_augment('node_EDEN', EDEN_augmentation_func)

register_feature_augment('node_PCA', DistancePCA_func)
register_feature_augment('node_MMPCA', MinMaxPCA_func)
register_feature_augment('node_RMMPCA', ReverseMinMaxPCA_func)

register_feature_augment('node_EDEN_trick', EDEN_augmentation_trick)
register_feature_augment('node_benchmark', benchmark_func)

register_feature_augment('node_RNI', RNI_func)

