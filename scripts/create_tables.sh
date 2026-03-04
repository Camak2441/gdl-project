echo multi all
python make_latex_table.py qigat vngnn vngnn2 qmpn qsgnn llm_gpt-4.1-nano rag_gpt-4.1-nano_k10/qmpn --multi --metrics recall precision accuracy f1
echo multi unseen
python make_latex_table.py qigat vngnn vngnn2 qmpn qsgnn --multi --metrics recall precision accuracy f1 --scene-split unseen
echo multi seen
python make_latex_table.py qigat vngnn vngnn2 qmpn qsgnn --multi --metrics recall precision accuracy f1 --scene-split seen
echo single all
python make_latex_table.py qigat vngnn vngnn2 qmpn qsgnn llm_gpt-4.1-nano_single rag_gpt-4.1-nano_single_k10/qmpn --metrics recall_1 recall_3 recall_5 recall_10
echo single unseen
python make_latex_table.py qigat vngnn vngnn2 qmpn qsgnn --metrics recall_1 recall_3 recall_5 recall_10 --scene-split unseen
echo single seen
python make_latex_table.py qigat vngnn vngnn2 qmpn qsgnn --metrics recall_1 recall_3 recall_5 recall_10 --scene-split seen