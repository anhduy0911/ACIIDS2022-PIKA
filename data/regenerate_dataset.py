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
        

def test_dataset():
    drugs = [d.name for d in os.scandir(CFG.test_folder_new) if d.is_dir()]
    # print(len(drugs))
    def get_label_index():
        labels = []
        g_embedding = []
        with open(CFG.g_embedding_path) as f:
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

    json.dump(condensed_g_embedding, open('data/converted_graph/condened_g_embedding.json', 'w'))

if __name__ == "__main__":
    test_dataset()