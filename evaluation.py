import json
from collections import *
def eval_current_state(queries):
	all_measured = 0
	all_right = 0
	all_wrong = 0
	all_none = 0
	for k, v in queries.items():
		if v['gold_entity'][0] not in v['candidate_entity']:
			all_none += 1
		else:
			predict_idx = v['candidate_score'].index(max(v['candidate_score']))
			gold_idx = v['candidate_entity'].index(v['gold_entity'][0])
			if gold_idx == predict_idx:
				all_right += 1
			else:
				all_wrong += 1
		all_measured += 1

	accuracy = float(all_right) / all_measured
	r1 = all_measured, all_wrong, accuracy, all_measured - all_none
	# print(all_right)
	# print(all_none)
	return r1

def eval_current_state_fahrni(queries):
	def render_f1(corr, precDenom, recDenom):
		prec = float(corr) / precDenom
		rec = float(corr) / recDenom
		return 'Prec = {}/{} = {}, Rec = {}/{} = {}, F1 = {}'.format(
			corr, precDenom, prec,
			corr, recDenom, rec,
			2 * prec * rec / (prec + rec or 1)
		)
	counter = defaultdict(lambda: 0)
	for k, v in queries.items():
		label = None
		
		predict_idx = v['candidate_score'].index(max(v['candidate_score']))
		predict_alia = v['candidate_entity'][predict_idx]
		gold_alia = v['gold_entity']
		if len(gold_alia) == 1 and (gold_alia[0] == '-NIL-' or gold_alia[0] == 'NIL'):
			if predict_alia == '-NIL-' or predict_alia == 'NIL':
				label = 'cNIL'
			else:
				label = 'wNIL_KB'
		elif predict_alia in gold_alia:
			label = 'cKB'
		elif predict_alia == '-NIL-' or predict_alia == 'NIL':
			label = 'wKB_NIL'
		else:
			label = 'wKB_KB'
		# if v['gold_candidate_alia'][0] not in v['candidate_alia']:
		# 	label = 'wKB_KB'
		# else:
		# 	predict_idx = v['candidate_score'].index(max(v['candidate_score']))
		# 	predict_alia = v['candidate_alia'][predict_idx]
		# 	gold_alia = v['gold_candidate_alia']
			# if gold_alia == predict_alia:
			# 	if gold_alia == '-NIL-':
			# 		label = 'cNIL'
			# 	else:
			# 		label = 'cKB'
			# else:
			# 	if gold_alia =='-NIL-':
			# 		label = 'wNIL_KB'
			# 	elif predict_alia == '-NIL-':
			# 		label = 'wKB_NIL'
			# 	else:
			# 		label = 'wKB_KB'

		counter[label] += 1
    
	rr = 'KB: {}'.format(render_f1(counter['cKB'], 
		counter['cKB'] + counter['wKB_KB'] + counter['wNIL_KB'], 
		counter['cKB'] + counter['wKB_KB'] + counter['wKB_NIL']))
	print (rr)
	if counter['cNIL']:
		rr2 = 'NIL: {}'.format(render_f1(counter['cNIL'], 
			counter['cNIL'] + counter['wKB_NIL'], 
			counter['cNIL'] + counter['wNIL_KB']))
		print (rr2)
		rr += '; '+rr2

	return counter, rr
