# general
seed_number = 911
train_folder = "data/pills/train/"
test_folder = "data/pills/test/"
train_folder_new = "data/pills/train_new/"
test_folder_new = "data/pills/test_new/"
g_embedding_path = "data/converted_graph/mapped_pills.dat"
g_embedding_condensed = "data/converted_graph/condened_g_embedding_deepwalk_w.json"
g_embedding_features = 64
n_class = 76
num_workers = 4
eval_steps = 5
early_stop = 20
backbone_path='logs/checkpoints/baseline_best.pt'
log_dir_run='logs/runs/'
log_dir_data='logs/data/'

# image parameters
image_size = 224
chanel_mean = [0.485, 0.456, 0.406]
chanel_std = [0.485, 0.456, 0.406]

# backbone model config
image_model_name = "resnet50"
image_pretrained = True
image_trainable = True
repeat = 10