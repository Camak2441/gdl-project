python calc_metrics.py ../outputs/results/llm_gpt-4.1-nano/test.pth --multi --split_dir '../data/dataset_balanced/all_minilm_l6v2;all_minilm_l6v2;all_minilm_l6v2/test'
python calc_metrics.py ../outputs/results/llm_gpt-4.1-nano_single/test.pth --split_dir '../data/dataset_single_balanced/all_minilm_l6v2;all_minilm_l6v2;all_minilm_l6v2/test'
python calc_metrics.py \
'../outputs/results/rag_gpt-4.1-nano_k10/QueryMpn(e_enc="all_minilm_l6v2",n_enc="all_minilm_l6v2",q_enc="all_minilm_l6v2",multi=true,hidden_dims=[128,128,128],out_dim=1)/test.pth' \
\--multi --split_dir '../data/dataset_balanced/all_minilm_l6v2;all_minilm_l6v2;all_minilm_l6v2/test'
python calc_metrics.py \
'../outputs/results/rag_gpt-4.1-nano_single_k10/QueryMpn(e_enc="all_minilm_l6v2",n_enc="all_minilm_l6v2",q_enc="all_minilm_l6v2",multi=false,hidden_dims=[128,128,128],out_dim=1)/test.pth' \
--split_dir '../data/dataset_single_balanced/all_minilm_l6v2;all_minilm_l6v2;all_minilm_l6v2/test'
python calc_metrics.py \
'../outputs/results/rag_gpt-4.1-nano_single_k10/QueryMpn(e_enc="all_minilm_l6v2",n_enc="all_minilm_l6v2",q_enc="all_minilm_l6v2",multi=true,hidden_dims=[128,128,128],out_dim=1)/test.pth' \
--split_dir '../data/dataset_single_balanced/all_minilm_l6v2;all_minilm_l6v2;all_minilm_l6v2/test'
python calc_metrics.py \
'../outputs/results/rag_gpt-4.1-nano_single_k10/VirtualNodeWrapper(e_enc="all_minilm_l6v2",n_enc="all_minilm_l6v2",q_enc="all_minilm_l6v2",multi=true,model="Gat(hidden_dims=[64,64,64],out_dim=1,heads=4)",query_mlp=false)/test.pth' \
--split_dir '../data/dataset_single_balanced/all_minilm_l6v2;all_minilm_l6v2;all_minilm_l6v2/test'