# general
seed_number = 911
train_folder = "data/pills/train/"
test_folder = "data/pills/test/"
train_folder_new = "data/pills/train_new/"
train_folder_v2 = "data/pills/train_v2/"
train_folder_fewshot = "data/pills/data_train_fewshot/train_new/"
test_folder_new = "data/pills/test_new/"
test_folder_v2 = "data/pills/test_v2/"
g_embedding_path = "data/converted_graph/mapped_pills.dat"
g_embedding_condensed = "data/converted_graph/condened_g_embedding_deepwalk_w.json"
map_class_to_idx = 'data/converted_graph/mapdict.json'
g_embedding_features = 64
n_class = 76
num_workers = 0
eval_steps = 5
early_stop = 20
backbone_path='logs/checkpoints/Rn_101_equal_ds_best.pt'
log_dir_run='logs/runs/'
log_dir_data='logs/data/'

buffer_size=1000
# image parameters
image_size = 224
chanel_mean = [0.485, 0.456, 0.406]
chanel_std = [0.229, 0.224, 0.225]

# backbone model config
image_model_name = "resnet50"
image_pretrained = True
image_trainable = True
repeat = 10
