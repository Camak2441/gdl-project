echo multi all
python make_latex_table.py qigat vngnn vngnn2 qmpn qsgnn llm_gpt-4.1-nano rag_gpt-4.1-nano_k10/qmpn --multi --metrics recall --by-qtype
python make_latex_table.py qigat vngnn vngnn2 qmpn qsgnn llm_gpt-4.1-nano rag_gpt-4.1-nano_k10/qmpn --multi --metrics precision --by-qtype
python make_latex_table.py qigat vngnn vngnn2 qmpn qsgnn llm_gpt-4.1-nano rag_gpt-4.1-nano_k10/qmpn --multi --metrics per_q_recall --by-qtype
python make_latex_table.py qigat vngnn vngnn2 qmpn qsgnn llm_gpt-4.1-nano rag_gpt-4.1-nano_k10/qmpn --multi --metrics per_q_precision --by-qtype
echo single all
python make_latex_table.py qigat vngnn vngnn2 qmpn qsgnn llm_gpt-4.1-nano_single rag_gpt-4.1-nano_single_k10/qmpn --metrics recall_1 --by-qtype
python make_latex_table.py qigat vngnn vngnn2 qmpn qsgnn llm_gpt-4.1-nano_single rag_gpt-4.1-nano_single_k10/qmpn --metrics recall_5 --by-qtype