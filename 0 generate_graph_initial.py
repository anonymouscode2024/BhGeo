# -*- coding: UTF-8 -*-

'''
'''

import json
import os
import numpy as np
import networkx as nx
import random

random.seed(0)
from collections import defaultdict

fr1 = open('tokyo_landmark.txt','r',encoding='UTF-8')

landmark_info_dict={}#defaultdict(list)

for line in fr1.readlines():
    str_list = line.strip().split(sep='\t')
    ip=str_list[0]
    city = "tokyo"
    lat = str_list[1]
    lon = str_list[2]
    landmark_info_dict[ip]=[city,lat,lon]


path = os.listdir(os.getcwd())

folder_city_name='tokyo_raw'
allpath = os.getcwd()
folderlist=[folder_city_name]
zaima = 0
i = 0

removed_dstip={}
kept_dstip={}

for p in folderlist:#

    ip_id_dict = defaultdict(int)  
    edge_dict = defaultdict(list)  
    srcip_id_dict = defaultdict(int) 
    dstip_id_dict = defaultdict(list)  

    ip_delaylist = defaultdict(list)  
    edge_rttlist = defaultdict(int)

    if os.path.isdir(allpath + os.sep + p): 
        os.chdir(allpath+os.sep+p)
        # print(p)
        chpath = os.listdir(os.getcwd())
        filei = 0
        for chp in chpath:


            f1 = open(chp, "r")
            if filei%10 ==1 :
                print (filei)

            filei = filei+1
            try:
                for line in f1:
                    data = json.loads(line)
                    if 'stop_reason' not in data.keys():
                        continue
                    if data['stop_reason'] != 'COMPLETED':#for demo test, you can only focus on completed routing , for real unreachbale targets, we should remove this condition,any target can be processed if they have some routing paths
                        continue


                    dstIP = data['dst']#

                    delay = float(data['hops'][-1]['rtt'])

                    srcIP = data['src']
                    hops = data['hops']

                    ip_id_dict.setdefault(srcIP,len(ip_id_dict))#
                    srcip_id  = ip_id_dict[srcIP]
                    srcip_id_dict[srcIP]=srcip_id 
                    ip_delaylist[srcIP]=[0]  

                    ip_id_dict.setdefault(dstIP, len(ip_id_dict))  
                    dstip_id = ip_id_dict[dstIP]
                    if dstIP not in dstip_id_dict:
                        dstip_id_dict[dstIP].append(dstip_id)


                    ip_delaylist.setdefault(dstIP,[])
                    ip_delaylist[dstIP].append(delay)#

                    # delay = hops[-1]['rtt']#
                    # hop_count=0#
                    previous_id = srcip_id#
                    previous_rtt = 0
                    for hop in hops:
                        hopIP = hop['addr']
                        current_rtt = hop['rtt']

                        if (hopIP == '124.74.38.142'):#
                            continue

                        ip_id_dict.setdefault(hopIP, len(ip_id_dict))  # 
                        current_id = ip_id_dict[hopIP]
                        temp_edge = str( previous_id)+','+str(current_id)

                        ip_delaylist.setdefault(hopIP, [])
                        ip_delaylist[hopIP].append(current_rtt)  #
                        # id_delaylist[hopIP].append(current_rtt)

                        previous_id=current_id

                        edge_dict.setdefault(temp_edge,[])
                        # edge_dict[temp_edge].append(current_edge_delay)
                      
                        # previous_rtt = current_rtt

            except Exception as e:
                print('Error:', e)
    #
    # 
    file1 = open("E:\\1 rawdata_preprocess\\tokyo_processed\\tky_ip_id.txt", 'w',
                 encoding='utf-8')
    file2 = open('E:\\1 rawdata_preprocess\\tokyo_processed\\tky_edges.txt', 'w',
                 encoding='utf-8')
    file3 = open('E:\\1 rawdata_preprocess\\tokyo_processed\\tky_srcip_id.txt', 'w',
                 encoding='utf-8')
    file4 = open('E:\\1 rawdata_preprocess\\tokyo_processed\\tky_dstip_id_allinfo.txt', 'w',
                 encoding='utf-8')

    temp_list = []#np.array()
    id_ip_mindelay_dict={}
    for key,content in ip_delaylist.items():
        ip=key
        id=ip_id_dict[ip]
        delay_list = content
        min_delay = min(delay_list)
        id_ip_mindelay_dict[id]=[ip,min_delay]

        file1.write(ip + ',' + str(id) + ','+ str(min_delay)+'\n')



    for key, content in edge_dict.items():
        id_id = key.split(',')
        id1 = int(id_id[0])
        id2 = int(id_id[1])
        ip1 = id_ip_mindelay_dict[id1][0]
        ip2 = id_ip_mindelay_dict[id2][0]
        delay1 = id_ip_mindelay_dict[id1][1]
        delay2 = id_ip_mindelay_dict[id2][1]
        delay = delay2 - delay1

        file2.write(key + ',' + str(delay) +','+ip1+','+ip2+'\n')



    for srcip, id in srcip_id_dict.items():
        file3.write(srcip+','+str(id)+'\n')

    file1.close()
    file2.close()
    file3.close()

    from tqdm import tqdm

    for dstip, id_city in tqdm(dstip_id_dict.items()):
        id = id_city[0]
        file4.write(dstip+','+str(id)+',')
        content= landmark_info_dict[dstip]
        file4.write(content[0] + ',')
        file4.write(content[1] + ',')
        file4.write(content[2] + '\n')


    file4.close()

