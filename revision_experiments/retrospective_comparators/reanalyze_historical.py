from pathlib import Path
import csv,hashlib,json,math,statistics
from datetime import datetime,timezone
import numpy as np

WORK=Path(r'rehabal_exp')
OUT=WORK/'results'/'retrospective_additional_comparators'
ORIG=WORK/'original_code'/'RehabAL-Pose_Code'
BUDGETS=[.05,.1,.2,.4]
TARGETS=[418,837,1673,3347]

def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def read_csv(path):
    with path.open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
def equal_rows(a,b):
    if len(a)!=len(b):return False
    for x,y in zip(a,b):
        if set(x)!=set(y):return False
        for k in x:
            if x[k]==y[k]:continue
            try:
                if not math.isclose(float(x[k]),float(y[k]),rel_tol=0,abs_tol=1e-14):return False
            except (TypeError,ValueError):return False
    return True

def main():
    dest=OUT/'analysis_payload.json'
    if dest.exists():
        old=dest.read_text(encoding='utf-8')
        raise RuntimeError('Existing output preserved; inspect before a new run')
    audit_path=WORK/'results'/'provenance'/'existing_additional_comparators_audit.json'
    audit=json.loads(audit_path.read_text(encoding='utf-8'))
    sources={};groups={}
    for record in audit['local_csv_audit']['groups']:
        p=Path(record['path'])
        assert sha(p)==record['sha256'],'Audited source CSV changed'
        sources[record['name']]={'path':str(p),'sha256':sha(p)}
        groups[record['name']]=read_csv(p)
    anchor_path=ORIG/'outputs'/'experiments'/'clinician_i4_uniform30'/'clinician_curve.csv'
    anchor_server=WORK/'server_metadata'/'outputs'/'experiments'/'clinician_i4_uniform30'/'clinician_curve.csv'
    anchor=read_csv(anchor_path)
    assert equal_rows(anchor,read_csv(anchor_server)),'Packaged anchor differs from audited server copy'
    sources['clinician_i4_uniform30']={'path':str(anchor_path),'sha256':sha(anchor_path),
        'audited_server_copy':str(anchor_server),'audited_server_copy_sha256':sha(anchor_server),'records_match':True}
    groups['clinician_i4_uniform30']=anchor
    specs=[
      ('I3','A0','Original self-training (A0)','clinician_i4_uniform30','A',None),
      ('I3','A_conf','Confidence gate (A_conf)','clinician_i3_conf','A_conf',None),
      ('I3','B_sigma0','GT correction + rejection (B, sigma=0)','clinician_i4_uniform30','B',0.),
      ('I3','ref','Full GT (ref)','clinician_i4_uniform30','ref',None),
      ('I7','temporal_anchor','Temporal (anchor)','clinician_i4_uniform30','ref',None),
      ('I7','random','Random (unseeded acquisition RNG)','clinician_i7_random','ref',None),
      ('I7','uncertainty','Least-confidence score','clinician_i7_uncertainty','ref',None),
      ('I7','centroid_proxy','Centroid-distance proxy','clinician_i7_coreset','ref',None),
      ('I7','rehabal_score','RehabAL score','clinician_i7_rehabal','ref',None)]
    per_seed=[];raw=[]
    for experiment,key,label,source,condition,sigma in specs:
        records=[r for r in groups[source] if r['condition']==condition and
                 (sigma is None or float(r['sigma'])==sigma)]
        assert len(records)==24 and {int(r['seed']) for r in records}==set(range(6))
        for seed in range(6):
            r=sorted((v for v in records if int(v['seed'])==seed),key=lambda z:float(z['budget']))
            assert [float(v['budget']) for v in r]==BUDGETS
            assert [int(v['n_labeled']) for v in r]==TARGETS
            result={'experiment':experiment,'series':key,'label':label,'seed':seed,
                    'source_group':source,'condition_in_source':condition,'sigma_px':sigma,
                    'auc_budget_min':.05,'auc_budget_max':.4,'auc_width':.35,
                    'n_budgets':4,'endpoint_budget':.4,'historical_result_reuse':True}
            for metric in ['map50_95','map50']:
                values=[float(v[metric]) for v in r]
                assert all(math.isfinite(v) and 0<=v<=1 for v in values)
                direct=sum((values[i]+values[i+1])/2*(BUDGETS[i+1]-BUDGETS[i]) for i in range(3))/.35
                via_numpy=float(np.trapz(values,BUDGETS)/.35)
                assert math.isclose(direct,via_numpy,rel_tol=0,abs_tol=1e-14)
                result['normalized_auc_'+metric]=direct
                result['endpoint_'+metric]=values[-1]
            per_seed.append(result)
            for v in r:
                raw.append({'experiment':experiment,'series':key,'seed':seed,'budget':float(v['budget']),
                   'accepted_frame_count':int(v['n_labeled']),'map50_95':float(v['map50_95']),
                   'map50':float(v['map50']),'source_group':source,'historical_result_reuse':True})
    summary=[]
    for experiment,key,label,source,condition,sigma in specs:
        rows=[r for r in per_seed if r['experiment']==experiment and r['series']==key]
        assert len(rows)==6
        item={'experiment':experiment,'series':key,'label':label,'recorded_training_seeds':6,
              'population_sd_ddof':0,'source_group':source,'condition_in_source':condition,
              'historical_result_reuse':True}
        for metric in ['normalized_auc_map50_95','endpoint_map50_95','normalized_auc_map50','endpoint_map50']:
            values=[r[metric] for r in rows]
            item[metric+'_mean']=statistics.mean(values)
            item[metric+'_population_sd']=statistics.pstdev(values)
            assert math.isclose(item[metric+'_population_sd'],float(np.std(values,ddof=0)),rel_tol=0,abs_tol=1e-14)
        summary.append(item)
    tables={'i3_per_seed':[r for r in per_seed if r['experiment']=='I3'],
            'i3_summary':[r for r in summary if r['experiment']=='I3'],
            'i7_per_seed':[r for r in per_seed if r['experiment']=='I7'],
            'i7_summary':[r for r in summary if r['experiment']=='I7'],
            'audited_source_budget_curves':raw}
    payload={'status':'complete_historical_descriptive_reanalysis','created_at_utc':datetime.now(timezone.utc).isoformat(),
        'metric':'source-domain REHAB24-6 Camera17 validation pose mAP50-95; mAP50 also retained in CSV',
        'normalized_auc_definition':'trapezoidal integral across accepted-label budgets [.05,.1,.2,.4] divided by .35',
        'mean_and_spread':'six recorded training runs; population standard deviation ddof=0; not confidence intervals',
        'hypothesis_tests_performed':False,'new_experiments_performed':False,'penn_or_new_baseline_files_read':False,
        'combined_with_prospective_main_analysis':False,'audit_sha256':sha(audit_path),
        'source_csvs':sources,'code_sha256':sha(Path(__file__)),'tables':tables,
        'limitations':['Historical result reuse, not new independent experiments.',
          'I7 uses full-GT ref labeling; it tests acquisition sensitivity.',
          'I7 centroid-distance score proxy does not invoke greedy k-center selection.',
          'I7 random acquisition RNG is not tied to the six training seeds.',
          'Historical source/input manifests absent; saved configuration and completed-run records were audited.',
          'No Penn PCK or new-policy result was accessed.']}
    with dest.open('x',encoding='utf-8') as f:json.dump(payload,f,ensure_ascii=False,indent=2,allow_nan=False)
    print(json.dumps({'status':payload['status'],'I3_per_seed_rows':24,'I7_per_seed_rows':30,
       'summaries':[{'series':r['series'],'auc_mean':r['normalized_auc_map50_95_mean'],'auc_sd':r['normalized_auc_map50_95_population_sd'],'map40_mean':r['endpoint_map50_95_mean'],'map40_sd':r['endpoint_map50_95_population_sd']} for r in summary]}))
if __name__=='__main__':main()
