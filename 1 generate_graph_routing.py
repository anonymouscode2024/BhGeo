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

fr1 = open('tokyo_landmark.txt','r',encoding='UTF-8')


landmark_info_dict={}#

for line in fr1.readlines():
    str_list = line.strip().split(sep='\t')
    ip=str_list[0]
    city = "tokyo"
    lat = str_list[1]
    lon = str_list[2]
    landmark_info_dict[ip]=[city,lat,lon]


folder_city_name='tokyo_raw'

fw = open('tokyo_processed/tky_dst_path.json', 'w')

allpath = os.getcwd()#
folderlist=[folder_city_name]#
zaima = 0
i = 0

removed_dstip={}#
kept_dstip={}
for p in folderlist:#文件夹名称

    ip_id_dict = defaultdict(int)  #
    edge_dict = defaultdict(list)  # 
    srcip_id_dict = defaultdict(int)  # 
    dstip_id_dict = defaultdict(list)  # 

    dstip_routerlist = defaultdict(dict)  # 

    if os.path.isdir(allpath + os.sep + p): #

        os.chdir(allpath+os.sep+p)
        chpath = os.listdir(os.getcwd())#
        filei = 0
        for chp in chpath:#

            f1 = open(chp, "r")
            if filei%10 ==2 :#
                print (filei)
                # break
            # print(filei)

            filei = filei+1
            # line_count=0
            try:
                for line in f1:#
                    # print(line_count)
                    # line_count+=1
                    data = json.loads(line)
                    if 'stop_reason' not in data.keys():#
                        continue
                    if data['stop_reason'] != 'COMPLETED':
                        continue

                    dstIP = data['dst']#

                    delay = float(data['hops'][-1]['rtt'])

                    srcIP = data['src']
                    hops = data['hops']
                    full_hops =  data['hop_count']#
                    founded_hops = len(hops)#
                    lost_num = full_hops - founded_hops

                    ip_id_dict.setdefault(srcIP,len(ip_id_dict))#
                    srcip_id  = ip_id_dict[srcIP]
                    srcip_id_dict[srcIP]=srcip_id #

                    ip_id_dict.setdefault(dstIP, len(ip_id_dict))  #
                    dstip_id = ip_id_dict[dstIP]
                    if dstIP not in dstip_id_dict:
                        dstip_id_dict[dstIP].append(dstip_id)

                    dstip_routerlist.setdefault(dstIP,{})
                    dstip_routerlist[dstIP].setdefault(full_hops,[])
                    router_list = [srcip_id]
                    router_list.extend(["lost"] * full_hops)

                    previous_id = srcip_id#

                    previous_rtt = 0
                    for hop in hops:
                        hopIP = hop['addr']

                        # if (hopIP == '124.74.38.142'):
                        #      # print('1')
                        #     continue

                        current_rtt = hop['rtt']  #

                        probe_ttl = int(hop['probe_ttl'])#
                        if(hopIP not in ip_id_dict):
                            ip_id_dict[hopIP]=len(ip_id_dict)  #

                        current_id = ip_id_dict[hopIP]

                        router_list[probe_ttl]=current_id#!!!!!!!!!!!!!!!!!!!!

                    flag = 0#

                    #
                    if (len(dstip_routerlist[dstIP][full_hops]) == 0):  #
                        # 101.95.111.145,13;1;2,;2,3,lost,4,5,6,7,8,9,10,11,12,1,
                        dstip_routerlist[dstIP][full_hops].append(router_list)
                        continue


                    for router_list_id,old_router_list in enumerate(dstip_routerlist[dstIP][full_hops]):
                        # flag = 0#
                        possbile_router_list =  old_router_list
                        
                        for k,c in enumerate(old_router_list):
                              if(router_list[k]!='lost'):
                                if(c == 'lost'):
                                    possbile_router_list[k] = router_list[k]
                                else:
                                    if(c!=router_list[k]):
                                        flag = 1#
                                        break

                        #
                        #
                        #
                        if(flag==0):
                            dstip_routerlist[dstIP][full_hops][router_list_id]=possbile_router_list
                            break#
                        else:
                            if(router_list_id == len(dstip_routerlist[dstIP][full_hops])-1):#
                                dstip_routerlist[dstIP][full_hops].append(router_list)
                                break#按
                            else:
                                flag=0 ##

            except Exception as e:
                print('Error:', e)

    for dstIP,content in dstip_routerlist.items():
        for full_hops, router_list_for_this_fullhops in content.items():
            for router_list in router_list_for_this_fullhops:
                fw.write(dstIP+','+str(full_hops)+';'+str('lostnum')+';')
                for loc,temp_router in enumerate(router_list):
                    if(temp_router == 'lost'):
                        fw.write(str(loc)+',')
                fw.write(';')
                for loc,temp_router in enumerate(router_list):
                    fw.write(str(temp_router)+',')
                fw.write('\n')

    fw.close()







