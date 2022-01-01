import networkx as nx
# from torch_geometric.data import Dataset
# from torch_geometric.utils.convert import from_networkx
import json
import math


def build_KG_graph(json_file):
    coocurence = {}
    pill_occurence = {}
    diag_occurence = {}
    
    def convert_KG_data(json_file):
        with open(json_file) as f:
            data = json.load(f)
        for pres in data:
            for pill in pres['drugname']:
                pill_occurence[pill] = pill_occurence.get(pill, 0) + 1
                for diag in pres['diagnose']:
                    diag_code = str.strip(str.split(diag, ' - ')[0])
                    diag_occurence[diag_code] = diag_occurence.get(diag_code, 0) + 1
                    if coocurence.get(pill) is None:
                        coocurence[pill] = {}
                    if coocurence.get(diag_code) is None:
                        coocurence[diag_code] = {}
                    coocurence[pill][diag_code] = coocurence[pill].get(diag_code, 0) + 1
                    coocurence[diag_code][pill] = coocurence[diag_code].get(pill, 0) + 1

    convert_KG_data(json_file)

    def tf_idf(pill, diag):
        tf = coocurence[pill][diag] / diag_occurence[diag]
        idf =  math.log( sum(diag_occurence.values()) / pill_occurence[pill])

        return tf * idf

    weighted_edges = {}
    for pill in pill_occurence.keys():
        for diag in coocurence[pill].keys():
            # print(f'pill: {pill} diag: {diag}')
            if weighted_edges.get(pill) is None:
                weighted_edges[pill] = {}
            weighted_edges[pill][diag] = tf_idf(pill, diag)

    # print(weighted_edges)
    with open('data/prescriptions/pill_data.dat', 'w') as f:
        for pill in weighted_edges.keys():
            for diag, weight in weighted_edges[pill].items():
                # print('im here')
                f.write(pill + '\\' + diag + '\\' + str(weight) + '\n')


def prepare_prescription_dataset(json_file, graph_path='./data/converted_graph/vectors_u.dat'):
    import os

    training_path = './data/prescriptions/train/'
    test_path = './data/prescriptions/test/'
    mapped_pills_dict = {}

    def read_graph_embedding(graph_path):
        g_embedding = {}
        with open(graph_path) as f:
            lines = f.readlines()
            for line in lines:
                pill_name, ebd = line.strip().split('\\')
                g_embedding[pill_name] = ebd
                mapped_pills_dict[pill_name] = ""
        return g_embedding
    
    g_embedding = read_graph_embedding(graph_path)

    drugs = list(mapped_pills_dict.keys())
    with open(json_file) as f:
        data = json.load(f)
        for pres in data:
            filename = pres['id']
            # drugs = pres['drugname']
            if (os.path.isfile(training_path + filename)):
                with open(training_path + filename) as f:
                    data_inner = json.load(f)
                    for box in data_inner:
                        if box['label'] == 'drugname':
                            mapped_name = box['mapping']
                            actual_name = box['text'][3:]
                            if actual_name in drugs:
                                # print(f'The intial mapped name is {mapped_pills_dict.get(actual_name, "NONE")}')
                                mapped_pills_dict[actual_name] = mapped_name
                                print(f'{actual_name} -> {mapped_name}')  
                            else: 
                                print(f'{actual_name} -> NONE') 

            elif (os.path.isfile(test_path + filename)):
                with open(test_path + filename) as f:
                    data_inner = json.load(f)
                    for box in data_inner:
                        if box['label'] == 'drugname':
                            mapped_name = box['mapping']
                            actual_name = box['text'][3:]
                            if actual_name in drugs:
                                # print(f'The intial mapped name is {mapped_pills_dict.get(actual_name, "NONE")}')
                                mapped_pills_dict[actual_name] = mapped_name
                                print(f'{actual_name} -> {mapped_name}')  
                            else: 
                                print(f'{actual_name} -> NONE')  
            else: 
                print(f'{filename} is not found')
    # print(len(mapped_pills_dict.keys()))
    # for pill in mapped_pills_dict.keys():
    #     if mapped_pills_dict[pill] == "":
    #         print(pill)

    with open('./data/converted_graph/mapped_pills.dat', 'w') as f:
        for pill, mapped_pill in mapped_pills_dict.items():
            if mapped_pill != "":
                f.write(pill + '\\' + mapped_pill + '\\' + g_embedding[pill] + '\n')


def test():
    import os 

    path = 'data/prescriptions/condensed_data.json'
    training_path = './data/prescriptions/train/'
    test_path = './data/prescriptions/test/'

    with open(path) as f:
        data = json.load(f)
        file_list = [p['id'] for p in data]
        for root, dir, files in os.walk(test_path):
            for file in files:
                assert(file in file_list)

def condensed_result_file():
    import os 

    path = 'data/prescriptions/result.json'
    training_path = './data/prescriptions/train/'
    test_path = './data/prescriptions/test/'

    condensed_data = []
    with open(path) as f:
        data = json.load(f)

        for p in data:
            filename = p['id']
            if os.path.isfile(training_path + filename):
                condensed_data.append(p)        
            elif os.path.isfile(test_path + filename):
                condensed_data.append(p)
    
    with open('./data/prescriptions/condensed_data.json', 'w', encoding='utf8') as f:
        json.dump(condensed_data, f, ensure_ascii=False)

if __name__ == '__main__':
    # build_KG_graph('data/prescriptions/condensed_data.json')
    prepare_prescription_dataset('data/prescriptions/condensed_data.json')
    # condensed_result_file()
    # test()