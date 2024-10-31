import random as rd

rd.seed(0)


def train_test_val(filepath):
    fr3 = open(filepath, 'r', encoding='UTF-8')
    target_lat_label_dict = {}
    #45.32.78.42,0,0
    for line in fr3.readlines():
        str_list = line.strip('\r\n').split(sep=',')
        node_id = int(str_list[1])
        target_lat_label_dict[node_id] = 0

    train_node_id_list = []  # 7:2:1
    val_node_id_list = []
    test_node_id_list = []


    for key, item in target_lat_label_dict.items():
        node_id = int(key)
        rd_number = rd.random()
        if (rd_number < 0.1):
            test_node_id_list.append(node_id)
        elif (rd_number < 0.3) & (rd_number >= 0.1):
            val_node_id_list.append(node_id)
        else:
            train_node_id_list.append(node_id)

    return [train_node_id_list,val_node_id_list,test_node_id_list]

if __name__ == '__main__':

    result_path1 = './tokyo_processed/train_ip_id.txt'  #
    result_path2 = './tokyo_processed/val_ip_id.txt'  #
    result_path3 = './tokyo_processed/test_ip_id.txt'  #

    fw1 = open(result_path1, 'w')
    fw2 = open(result_path2, 'w')
    fw3 = open(result_path3, 'w')
    fr1 = open("./tokyo_processed/tky_ip_id.txt","r")

    ip_id_dict = {}
    for line in fr1.readlines():
        str_list = line.strip().split(',')
        ip = str_list[0]
        id = int(str_list[1])
        ip_id_dict[id] = ip

    train_node_id_list,val_node_id_list,test_node_id_list =  train_test_val("./tokyo_processed/tky_ip_id.txt")

    for id,ip in ip_id_dict.items():
        if(id in train_node_id_list):
            fw1.write(ip+','+str(id)+'\n')
            fw1.flush()
        elif (id in val_node_id_list):
            fw2.write(ip+','+str(id)+'\n')
            fw2.flush()
        elif(id in test_node_id_list):#
            fw3.write(ip+','+str(id)+'\n')
            fw3.flush()


    fw1.close()
    fw2.close()
    fw3.close()
