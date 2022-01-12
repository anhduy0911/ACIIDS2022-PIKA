import config as CFG
import os, shutil
import numpy as np
import json

def generate_new_dataset():
    # def get_label_index():
    #     labels = []
    #     g_embedding = []
    #     with open(CFG.g_embedding_path) as f:
    #         lines = f.readlines()
    #         for line in lines:
    #             _, mapped_pill, ebd = line.strip().split('\\')
    #             labels.append(mapped_pill)
    #             g_embedding.append([float(x) for x in ebd.split(' ')])

    #     return labels, g_embedding
    
    # labels, _ = get_label_index()

    train_pres = [d.name for d in os.scandir(CFG.train_folder) if d.is_dir()]
    train_limit = 40
    test_limit = 20
    drug_dict = {'train': {},'test':{}}
    for pres in train_pres:
        drugs = [d.name for d in os.scandir(CFG.train_folder + pres) if d.is_dir()]
        for drug in drugs:
            for _, _, files in os.walk(CFG.train_folder + pres + '/' + drug):
                for file in files:
                    if not os.path.isdir(CFG.train_folder_new + drug):
                        os.makedirs(CFG.train_folder_new + drug)
                        drug_dict['train'][drug] = 0
                    if not os.path.isdir(CFG.test_folder_new + drug):
                        os.makedirs(CFG.test_folder_new + drug)
                        drug_dict['test'][drug] = 0
                        
                    if drug_dict['train'][drug] < train_limit:
                        shutil.copy(CFG.train_folder + pres + '/' + drug + '/' + file, CFG.train_folder_new + drug)
                        drug_dict['train'][drug] += 1
                        continue
                    if drug_dict['test'][drug] < test_limit:
                        print('test_fd')
                        shutil.copy(CFG.train_folder + pres + '/' + drug + '/' + file, CFG.test_folder_new + drug)
                        drug_dict['test'][drug] += 1
                        continue

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
                pill_name, ebd = line.strip().split('@')
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

    with open('./data/converted_graph/mapped_pills_deepwalk_w.dat', 'w') as f:
        for pill, mapped_pill in mapped_pills_dict.items():
            if mapped_pill != "":
                f.write(pill + '\\' + mapped_pill + '\\' + g_embedding[pill] + '\n')


def test_dataset(g_embedding_path):
    drugs = [d.name for d in os.scandir(CFG.test_folder_new) if d.is_dir()]
    # print(len(drugs))
    def get_label_index():
        labels = []
        g_embedding = []
        with open(g_embedding_path) as f:
            lines = f.readlines()
            for line in lines:
                _, mapped_pill, ebd = line.strip().split('\\')
                labels.append(mapped_pill)
                g_embedding.append([float(x) for x in ebd.split(' ')])

        return labels, np.array(g_embedding)
    
    labels, g_embedding = get_label_index()
    print(g_embedding.shape)
    condensed_g_embedding = {}
    for drug in drugs:
        idxs = [i for i, x in enumerate(labels) if x == drug]
        # print(idxs)
        drug_emds = g_embedding[idxs]
        # print(drug_emds.shape)
        condensed_g_embedding[drug] = np.mean(drug_emds, axis=0).squeeze().tolist()

    print(len(condensed_g_embedding.keys()))
    json.dump(condensed_g_embedding, open('data/converted_graph/condened_g_embedding_deepwalk_w.json', 'w'))

if __name__ == "__main__":
    
    test_dataset('./data/converted_graph/mapped_pills_deepwalk_w.dat')