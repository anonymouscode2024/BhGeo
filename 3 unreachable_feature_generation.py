# -*- coding: UTF-8 -*-

import json
import os
import numpy as np
import networkx as nx
# import matplotlib.pyplot as plt
import pandas as pd
import random

from collections import defaultdict

import pandas
fr1 = open('tokyo_processed/tky_edges_remove_lost_median.txt', 'r', encoding='UTF-8')
fr2 = open('tokyo_processed/tky_ip_id.txt', 'r', encoding='UTF-8')


edge_list = [] #
edge_delay_list = [] #
# edge_lost_count_list = [] #
edge_hop_list = [] #
edge_ip1_list = [] #
edge_ip2_list = [] #
edge_ip3_list = [] #
edge_ip4_list = [] #
edge_ip5_list = []
edge_ip6_list = []
edge_ip7_list = []
edge_ip8_list = [] #

node_list = [] #
node_delay_list = [] #
node_ip1_list=[] #
node_ip2_list=[] 
node_ip3_list=[] #
node_ip4_list=[] #

id_ip_dict={}

for line in fr2.readlines():
    #
    # ip,id,delay
    str_list = line.strip().split(sep=',')
    ip =str_list[0]
    ip_list = ip.split('.')#4个字段特征!!!!
    id = int(str_list[1])
    id_ip_dict[id]=ip
    delay = str_list[2]
    node_delay_list.append(float(str_list[2]))#
    node_list.append([ip,id])#ip数字特征
    node_ip1_list.append(int(ip_list[0]))
    node_ip2_list.append(int(ip_list[1]))
    node_ip3_list.append(int(ip_list[2]))
    node_ip4_list.append(int(ip_list[3]))

for line in fr1.readlines():
    # 0	2	0.148	1	45.32.78.42	100.100.200.1
    # id id delay hop
    str_list = line.strip().split(sep='\t')
    id1=int(str_list[0])
    id2 = int(str_list[1])
    ip1=id_ip_dict[id1]
    ip2 = id_ip_dict[id2]
    ip1_list = ip1.split('.')
    ip2_list = ip2.split('.')  #!

    edge_list.append([id1, id2])  #
    edge_delay_list.append(float(str_list[2]))
    edge_hop_list.append(int(str_list[3]))

    edge_ip1_list.append(int(ip1_list[0]))
    edge_ip2_list.append(int(ip1_list[1]))
    edge_ip3_list.append(int(ip1_list[2]))
    edge_ip4_list.append(int(ip1_list[3]))
    edge_ip5_list.append(int(ip2_list[0]))
    edge_ip6_list.append(int(ip2_list[1]))
    edge_ip7_list.append(int(ip2_list[2]))
    edge_ip8_list.append(int(ip2_list[3]))

from sklearn.preprocessing import KBinsDiscretizer
from sklearn.preprocessing import MinMaxScaler

node_delay_transform = MinMaxScaler(feature_range=(0,1))
node_delay_transformed = node_delay_transform.fit_transform(np.array(node_delay_list).reshape(-1,1))
node_ip1_tf = MinMaxScaler(feature_range=(0,1))
node_ip1_tfed = node_ip1_tf.fit_transform(np.array(node_ip1_list).reshape(-1,1))
node_ip2_tf = MinMaxScaler(feature_range=(0,1))
node_ip2_tfed = node_ip1_tf.fit_transform(np.array(node_ip2_list).reshape(-1,1))
node_ip3_tf = MinMaxScaler(feature_range=(0,1))
node_ip3_tfed = node_ip1_tf.fit_transform(np.array(node_ip3_list).reshape(-1,1))
node_ip4_tf = MinMaxScaler(feature_range=(0,1))
node_ip4_tfed = node_ip1_tf.fit_transform(np.array(node_ip4_list).reshape(-1,1))

edge_delay_transform = MinMaxScaler(feature_range=(0,1))
edge_delay_transformed = edge_delay_transform.fit_transform(np.array(edge_delay_list).reshape(-1,1))
edge_hop_tf = MinMaxScaler(feature_range=(0,1))
edge_hop_tfed = edge_hop_tf.fit_transform(np.array(edge_hop_list).reshape(-1,1))
edge_ip1_tf  = MinMaxScaler(feature_range=(0,1))
edge_ip1_tfed = edge_ip1_tf.fit_transform(np.array(edge_ip1_list).reshape(-1,1))
edge_ip2_tf  = MinMaxScaler(feature_range=(0,1))
edge_ip2_tfed = edge_ip2_tf.fit_transform(np.array(edge_ip2_list).reshape(-1,1))
edge_ip3_tf  = MinMaxScaler(feature_range=(0,1))
edge_ip3_tfed = edge_ip3_tf.fit_transform(np.array(edge_ip3_list).reshape(-1,1))
edge_ip4_tf  = MinMaxScaler(feature_range=(0,1))
edge_ip4_tfed = edge_ip4_tf.fit_transform(np.array(edge_ip4_list).reshape(-1,1))
edge_ip5_tf  = MinMaxScaler(feature_range=(0,1))
edge_ip5_tfed = edge_ip5_tf.fit_transform(np.array(edge_ip5_list).reshape(-1,1))
edge_ip6_tf  = MinMaxScaler(feature_range=(0,1))
edge_ip6_tfed = edge_ip6_tf.fit_transform(np.array(edge_ip6_list).reshape(-1,1))
edge_ip7_tf  = MinMaxScaler(feature_range=(0,1))
edge_ip7_tfed = edge_ip7_tf.fit_transform(np.array(edge_ip7_list).reshape(-1,1))
edge_ip8_tf  = MinMaxScaler(feature_range=(0,1))
edge_ip8_tfed = edge_ip8_tf.fit_transform(np.array(edge_ip8_list).reshape(-1,1))


file1 = open('tokyo_processed/tky_ip_feature_median.txt', 'w', encoding='utf-8')
file2 = open('tokyo_processed/tky_edge_feature_median.txt', 'w', encoding='utf-8')


for i,k in enumerate(edge_list):
    file2.write(str(k[0])+','+str(k[1])+','+str(edge_delay_transformed[i][0])+','+str(edge_hop_tfed[i][0])+','
                +str(edge_ip1_tfed[i][0])+','+str(edge_ip2_tfed[i][0])+','+str(edge_ip3_tfed[i][0])+','+str(edge_ip4_tfed[i][0])
                +','+str(edge_ip5_tfed[i][0])+','+str(edge_ip6_tfed[i][0])+','+str(edge_ip7_tfed[i][0])+','+str(edge_ip8_tfed[i][0])+'\n')


for i,k in enumerate(node_list):
    file1.write(str(k[0])+','+str(k[1])+','+str(node_delay_transformed[i][0])+','+str(node_ip1_tfed[i][0])
                +','+str(node_ip2_tfed[i][0])+','+str(node_ip3_tfed[i][0])+','+str(node_ip4_tfed[i][0])+'\n')


file1.close()
file2.close()
