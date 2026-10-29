import csv
import hashlib
import json
import statistics
from pathlib import Path

OUT = Path(__file__).resolve().parent
BASE = OUT.parent
SOURCE = BASE / 'all_epochs_four_sets.csv'
KEYS = ['RMSE_m', 'REL', 'MAE_m', 'delta_1_05_percent', 'delta_1_10_percent', 'delta_1_25_percent']
SETS = ['synthetic-val', 'synthetic-test', 'real-val', 'real-test']
rows = list(csv.DictReader(SOURCE.open()))
epochs = sorted({int(r['epoch']) for r in rows})
d = {(int(r['epoch']), r['dataset']): {k: float(r[k]) for k in KEYS} for r in rows}
counts = {r['dataset']: int(r['images']) for r in rows}
assert len(rows) == 52 and len(epochs) == 13
assert all((e, s) in d for e in epochs for s in SETS)
assert all(r['status'] == 'completed' for r in rows)
original_hashes = json.loads((BASE / 'provenance/artifact_sha256.json').read_text())
assert hashlib.sha256(SOURCE.read_bytes()).hexdigest() == original_hashes[SOURCE.name]

def means(subset):
    return {e: {k: statistics.mean(d[e, s][k] for s in subset) for k in KEYS} for e in epochs}

def winners(values):
    return {k: min(epochs, key=lambda e: ((-1 if k.startswith('delta') else 1)*values[e][k], e)) for k in KEYS}

all_mean = means(SETS)
syn_mean = means(SETS[:2])
rank = {}
for s in SETS:
    for k in KEYS:
        ordered = sorted(epochs, key=lambda e: d[e, s][k], reverse=k.startswith('delta'))
        for e in epochs:
            rank[e, s, k] = statistics.mean([j + 1 for j, x in enumerate(ordered) if d[x, s][k] == d[e, s][k]])
rank_all = {e: statistics.mean(rank[e, s, k] for s in SETS for k in KEYS) for e in epochs}
rank_real = {e: statistics.mean(rank[e, 'real-val', k] for k in KEYS) for e in epochs}
weighted = {e: sum(d[e, s]['RMSE_m']*counts[s] for s in SETS)/sum(counts.values()) for e in epochs}
def six_metric_scores(values):
    output = {e: dict(ranks=[], minmax_losses=[], relative_losses=[]) for e in epochs}
    for k in KEYS:
        ordered = sorted(epochs, key=lambda e: values[e][k], reverse=k.startswith('delta'))
        low, high = min(values[e][k] for e in epochs), max(values[e][k] for e in epochs)
        for e in epochs:
            value = values[e][k]
            output[e]['ranks'].append(statistics.mean(i+1 for i,x in enumerate(ordered) if values[x][k]==value))
            regret = high-value if k.startswith('delta') else value-low
            reference = high if k.startswith('delta') else low
            output[e]['minmax_losses'].append(regret/(high-low) if high>low else 0.)
            output[e]['relative_losses'].append(regret/reference if reference else 0.)
    for item in output.values():
        item['mean_rank'] = statistics.mean(item['ranks'])
        item['mean_minmax_loss'] = statistics.mean(item['minmax_losses'])
        item['mean_relative_loss'] = statistics.mean(item['relative_losses'])
    return output
scopes = dict(four_sets_equal_mean=all_mean, real_val={e:d[e,'real-val'] for e in epochs},
              synthetic_two_equal_mean=syn_mean, synthetic_val={e:d[e,'synthetic-val'] for e in epochs})
six_scores = {scope:six_metric_scores(values) for scope,values in scopes.items()}
selected = {scope:min(epochs,key=lambda e:(scores[e]['mean_rank'],e)) for scope,scores in six_scores.items()}
for method in ['mean_rank','mean_minmax_loss','mean_relative_loss']:
    assert min(epochs,key=lambda e:six_scores['real_val'][e][method]) == 57
with (OUT/'real_val_six_metric_ranking.csv').open('w',newline='') as f:
    w=csv.DictWriter(f,fieldnames=['epoch',*KEYS,'mean_rank','mean_minmax_loss','mean_relative_loss'])
    w.writeheader()
    for e in sorted(epochs,key=lambda e:six_scores['real_val'][e]['mean_rank']):
        w.writerow(dict(epoch=e,**d[e,'real-val'],**{k:six_scores['real_val'][e][k] for k in ['mean_rank','mean_minmax_loss','mean_relative_loss']}))
manifest = json.loads((BASE/'manifest.json').read_text())
candidate_paths = {c['epoch']:c for c in manifest['candidates']}
(OUT/'six_metric_selection.json').write_text(json.dumps(dict(
    user_requirement='real-val最好不能只看real-val的rmse，需要综合六个指标',
    metric_order=KEYS, aggregation='Within each requested scope, average each metric equally over datasets, then average six metric ranks equally. Single-set scopes require no dataset averaging.',
    scores=six_scores, selected_epochs=selected,
    checkpoints={scope:candidate_paths[e] for scope,e in selected.items()},
    historical_rmse_only_selection=dict(epoch=37,path=str(BASE/'best_real_val_network.pth')),
    note='The historical RMSE-only result and file are preserved. The current six-metric real-val recommendation is epoch57. No statistical significance is claimed.'),indent=2,ensure_ascii=False))
result = dict(source=str(SOURCE), source_sha256=hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
    epochs=epochs, missing_epochs=json.loads((BASE/'manifest.json').read_text())['missing_epochs'], counts=counts,
    method='Each dataset has equal weight; average each metric independently, never sum unlike units. No model was rerun.',
    all_four_mean=all_mean, synthetic_two_mean=syn_mean,
    all_four_metric_winners=winners(all_mean), synthetic_two_metric_winners=winners(syn_mean),
    individual_set_metric_winners={s: winners({e:d[e,s] for e in epochs}) for s in SETS},
    sensitivity=dict(all_24_equal_rank=rank_all, real_val_6_equal_rank=rank_real,
                     all_images_weighted_mean_per_image_RMSE=weighted,
                     best_all_rank=min(rank_all,key=rank_all.get),best_real_rank=min(rank_real,key=rank_real.get),
                     best_image_weighted_rmse=min(weighted,key=weighted.get),synthetic_image_fraction=940/1226),
    interpretation='User clarified real-val must consider six metrics: epoch57 is the new six-metric recommendation. Historical epoch37 is RMSE-only. Four-set equal means favor epoch37; alternative per-dataset mean ranks or image weighting can favor epoch56. Four-set comparison is retrospective reporting.')
(OUT/'analysis.json').write_text(json.dumps(result,indent=2,ensure_ascii=False))
for name, values in [('four_set_macro_average.csv',all_mean),('synthetic_macro_average.csv',syn_mean)]:
    with (OUT/name).open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=['epoch',*KEYS]);w.writeheader()
        w.writerows(dict(epoch=e,**values[e]) for e in epochs)
with (OUT/'selection_sensitivity.csv').open('w',newline='') as f:
    w=csv.writer(f);w.writerow(['epoch','four_set_equal_mean_RMSE_m','four_set_equal_metric_mean_rank','four_set_image_weighted_mean_RMSE_m','real_val_RMSE_m','real_val_equal_metric_mean_rank','synthetic_equal_mean_RMSE_m'])
    for e in epochs:w.writerow([e,all_mean[e]['RMSE_m'],rank_all[e],weighted[e],d[e,'real-val']['RMSE_m'],rank_real[e],syn_mean[e]['RMSE_m']])
lines=['分析范围：13个保留权重，47个epoch缺失；不能推断所有60轮的全局最优。',
'最新要求：real-val需要综合六项指标，不能仅看RMSE。本报告据此更新结论。',
'结论：四集等权综合最佳epoch37；real-val六项综合最佳epoch57；两个synthetic集合等权最佳epoch60；若仅看synthetic-val，六项等权平均名次最佳epoch56。',
'综合口径：每个集合25%权重，对RMSE、REL、MAE、三个delta分别求均值；不把不同单位的六项指标直接相加。epoch37的六项均值全部为13个候选中的最佳。',
'合成口径：synthetic-val与synthetic-test各50%，分别平均六项指标；epoch60的RMSE、REL、MAE、delta1.05、delta1.10最好；delta1.25最高的是epoch38。',
'real-val各项冠军：RMSE和delta1.25为epoch37；REL、MAE和delta1.10为epoch57；delta1.05为epoch58。',
'real-val按六项等权平均名次：epoch57=2.166667，epoch56=2.5，epoch37=3.333333，epoch58=4.333333。越小越好。',
'交叉检查：将每项转换为0到1的最差-最好区间损失再等权平均，或转换为相对该项最佳值的损失再等权平均，real-val冠军也都是epoch57。评分是本次分析规则，不是论文官方指标；不代表统计显著性。',
'取舍：epoch57在real-val的REL、MAE、delta1.10三项第一，RMSE第三、delta1.05第五、delta1.25第二；对epoch37和56分别有四项更好、两项更差。',
'若对四集24项平均名次，epoch56最佳。名次法丢失指标差距大小，不等同于按指标数值平均。',
'若按1226张图直接加权平均每图RMSE，epoch56为0.02733964666442757m，epoch37为0.027417698433147182m；synthetic样本占76.67%，不再是四集等权。此值不是所有像素合并后计算的RMSE。',
'epoch37与56的real-val RMSE差仅0.00003981884472029m（约0.1134%），这是数值排名，不是统计显著性判断。',
'四个集合结果用于事后比较；按用户新要求，real-val综合推荐为epoch57。旧best_real_val_network.pth和旧报告保留epoch37的RMSE单项选择历史，没有静默覆盖。新选择与固定权重路径见six_metric_selection.json。没有重跑推理或训练。',
'', '四集等权六指标均值（epoch RMSE_m REL MAE_m delta1.05% delta1.10% delta1.25%）：']
for e in sorted(epochs,key=lambda e:all_mean[e]['RMSE_m']):lines.append(str(e)+' '+ ' '.join(f'{all_mean[e][k]:.9f}' for k in KEYS))
lines+=['','两个synthetic集合等权六指标均值（列同上）：']
for e in sorted(epochs,key=lambda e:syn_mean[e]['RMSE_m']):lines.append(str(e)+' '+ ' '.join(f'{syn_mean[e][k]:.9f}' for k in KEYS))
(OUT/'REPORT.txt').write_text('\n'.join(lines)+'\n')
print(json.dumps({k:result[k] for k in ['all_four_metric_winners','synthetic_two_metric_winners','individual_set_metric_winners']},indent=2))
print('FOUR-SET MEANS')
for e in [37,52,56,57,60]:print(e,*[round(all_mean[e][k],6 if k in KEYS[:3] else 2) for k in KEYS])
print('SYN MEANS')
for e in [56,59,60]:print(e,*[round(syn_mean[e][k],6 if k in KEYS[:3] else 2) for k in KEYS])
