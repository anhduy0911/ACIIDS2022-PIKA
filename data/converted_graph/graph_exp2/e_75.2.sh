CUDA_VISIBLE_DEVICES=2 python main.py --name KG_assisted_rn50_js_e75_2 --backbone resnet50 --val-batch-size 32 --batch-size 64 --loss js --g_emd_path ./data/converted_graph/graph_exp2/name_pill_weighted_e75.json