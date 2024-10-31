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
random.seed(0)

def mostnum(num):
    counts = np.bincount(num)
    return np.argmax(counts)

def mediannum(num):
    listnum = [num[i] for i in range(len(num))]
    listnum.sort()
    lnum = len(num)
    if lnum % 2 == 1:
        i = int((lnum + 1) / 2) - 1
        return listnum[i]
    else:
        i = int(lnum / 2) - 1
        return (listnum[i] + listnum[i + 1]) / 2


fr1 = open('tokyo_processed/tky_dst_path_.json', 'r', encoding='UTF-8')


fr2 = open('tokyo_processed/tky_edges.txt', 'r', encoding='UTF-8')


fr3 = open('tokyo_processed/tky_ip_id.txt', 'r', encoding='UTF-8')

fr4 = open('tokyo_processed/test_ip_id.txt','r')#

fw1 = open('tokyo_processed/tky_edges_remove_lost_median.txt', 'w',
          encoding='utf-8')


id_delay_dict = {}
id_ip={}#
for line in fr3.readlines():#
    str_list = line.strip('\n').split(',')#
    id = str_list[1]
    delay = str_list[2]
    ip = str_list[0]
    id_delay_dict[id]=float(delay)
    id_ip[id] = ip

full_edge_dict = {}
for line in fr2.readlines():#
    str_list = line.strip('\n').split(',')
    id1 = str_list[0]
    id2 = str_list[1]
    delay = str_list[2]
    ip1 = str_list[3]
    ip2 = str_list[4]
    full_edge_dict[id1+','+id2]=delay


test_ip_id={}
for line in fr4.readlines():#
    #180.161.208.248,439,1
    str_list = line.strip('\n').split(',')
    ip = str_list[0]
    # id = int(str_list[1])
    id = str_list[1]
    test_ip_id[ip]=id

processed_edge_dict ={}#
processed_edge_dict2 ={}#
for line in fr1.readlines():#
    str_list = line.strip('\n').split(';')
    ip_delay = str_list[0].split(',')
    dstip = ip_delay[0]#

    delay = ip_delay[1]#
    lost_num = str_list[1]#
    lost_list = str_list[2]#
    router_list = str_list[3].split(',')[:-1]#

    lost_count=0#
    pre_router = router_list[0]#

    for i, router in enumerate(router_list[1:]):
        temp_router = router#
        if(temp_router=='lost'):#
            lost_count+=1
            continue

        temp_edge = pre_router +',' +temp_router#

        if (temp_edge in processed_edge_dict2):  # 
            lost_count=0
            pre_router=temp_router
            continue

        if(temp_edge not in processed_edge_dict):#

            if(temp_edge in full_edge_dict):
                delay = full_edge_dict[temp_edge]
            else:
                delay1 = id_delay_dict[pre_router]
                delay2 = id_delay_dict[temp_router]
                delay  = delay2  -  delay1

            processed_edge_dict[temp_edge]=[delay,lost_count]#
            if(lost_count==0):#
                processed_edge_dict2[temp_edge] = [delay, lost_count]
        else:
        # 
            if lost_count <  processed_edge_dict[temp_edge][1] :
                processed_edge_dict[temp_edge][1] = lost_count
            if lost_count==0:#
                processed_edge_dict2[temp_edge] = [processed_edge_dict[temp_edge][0],0]
        # 
        lost_count = 0
        pre_router = temp_router

#
delayperhop_train_val = []  #
lost_counts_train_val=[]
for k,v in processed_edge_dict.items():
    temp_edge = k
    [pre_router, temp_router] = temp_edge.split(',')
    ip1 = id_ip[pre_router]
    ip2 = id_ip[temp_router]
    if (ip1 not in test_ip_id and ip2 not in test_ip_id):#
        delay = v[0]
        lost_count = v[1]
        delayperhop_train_val.append(float(delay)/float(lost_count+1))
        lost_counts_train_val.append(lost_count)#

#
from numpy import mean
median_delay = mediannum(delayperhop_train_val)
average_delay = mean(delayperhop_train_val)
max_count= max(lost_counts_train_val)#
min_count= min(lost_counts_train_val)#
most_count = mostnum(lost_counts_train_val)
for k,v in processed_edge_dict.items():

    temp_edge = k
    [pre_router, temp_router] = temp_edge.split(',')
    ip1 = id_ip[pre_router]
    ip2 = id_ip[temp_router]
    if (ip1 not in test_ip_id and ip2 not in test_ip_id):
        delay = v[0]
        lost_count = v[1]
        fw1.write(str(pre_router) + '\t' + str(temp_router) + '\t' + str(delay) + '\t' + str(lost_count) + '\t' + str(ip1) + '\t' + str(ip2) +'\n')  # 
        # fw2.write(str(pre_router) + ',' + str(temp_router) + ',' + str(delay) + ',' + str(lost_count) + ',' + str(ip1) + ',' + str(ip2) + '\n')  # 
        # fw3.write(str(pre_router) + ',' + str(temp_router) + ',' + str(delay) + ',' + str(lost_count) + ',' + str(ip1) + ',' + str(ip2) + '\n')  # 

    else:#

        # lost_count = random.randint(min_count,max_count)
        lost_count = random.randint(min_count,most_count)
        delay1 = median_delay * (lost_count+1)#
        # delay2 = average_delay * (lost_count)

        fw1.write(str(pre_router)+'\t'+str(temp_router)+'\t'+str(delay1)+'\t'+str(lost_count)+ '\t' + str(ip1) + '\t' + str(ip2)+'\n')#
        # fw2.write(str(pre_router) + ',' + str(temp_router) + ',' + str(delay2) + ',' + str(lost_count) + ',' + str(ip1) + ',' + str(ip2)+ '\n')  # 
        # fw3.write(str(pre_router) + ',' + str(temp_router) + ',' + str(0) + ',' + str(0) + ',' + str(
        #     ip1) + ',' + str(ip2) + '\n')  # 
fw1.close()
