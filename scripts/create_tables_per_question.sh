echo multi all
python make_latex_table.py qigat vngnn vngnn2 qmpn qsgnn llm_gpt-4.1-nano rag_gpt-4.1-nano_k10/qmpn --multi --metrics per_q_recall per_q_precision per_q_accuracy per_q_f1
echo multi unseen
python make_latex_table.py qigat vngnn vngnn2 qmpn qsgnn --multi --metrics per_q_recall per_q_precision per_q_accuracy per_q_f1 --scene-split unseen
echo multi seen
python make_latex_table.py qigat vngnn vngnn2 qmpn qsgnn --multi --metrics per_q_recall per_q_precision per_q_accuracy per_q_f1 --scene-split seen