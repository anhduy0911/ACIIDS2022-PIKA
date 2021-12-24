# general
seed_number = 911
train_folder = "data/pills/train/"
test_folder = "data/pills/test/"
g_embedding_path = "data/converted_graph/mapped_pills.dat"
g_embedding_features = 128
n_class = 89
num_workers = 4

# image parameters
image_size = 224
chanel_mean = [0.485, 0.456, 0.406]
chanel_std = [0.485, 0.456, 0.406]

# backbone model config
image_model_name = "resnet50"
image_pretrained = True
image_trainable = True
repeat = 10
