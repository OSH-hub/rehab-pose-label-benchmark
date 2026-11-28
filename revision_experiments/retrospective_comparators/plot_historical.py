from pathlib import Path
import json, argparse
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
OUT=Path(__file__).absolute().parent.parent
parser=argparse.ArgumentParser(); parser.add_argument('--replace-existing',action='store_true'); args=parser.parse_args()
payload=json.loads((OUT/'analysis_payload.json').read_text(encoding='utf-8'))
assert payload['status']=='complete_historical_descriptive_reanalysis'
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.titlesize':11,'axes.labelsize':10,
                     'pdf.fonttype':42,'ps.fonttype':42,'savefig.facecolor':'white','axes.linewidth':.7})
CONFIGS=[
 ('i3','historical_i3_confidence_comparator','I3: confidence-gated self-training',
  ['A0','A_conf','B_sigma0','ref'],
  ['Self-training (A0)','Confidence gate (A_conf)','GT correction + rejection (B)','Full GT (ref)'],
  'A_conf accepts a frame when mean predicted keypoint confidence is at least 0.5.\nB is the total adaptive policy: GT-derived correction (sigma=0) plus segmentation-based rejection.'),
 ('i7','historical_i7_acquisition_sensitivity','I7: full-GT acquisition sensitivity',
  ['temporal_anchor','random','uncertainty','centroid_proxy','rehabal_score'],
  ['Temporal (historical anchor)','Random*','Least-confidence score','Centroid-distance proxy','RehabAL score'],
  '*Random acquisition RNG was not fixed by the recorded training seed.\nThe centroid-distance score proxy does not invoke greedy k-center selection.')]
for group,basename,title,order,labels,note in CONFIGS:
    targets=[OUT/(basename+'.png'),OUT/(basename+'.pdf')]
    for p in targets:
        if p.exists():
            previous=p.read_bytes()
            if not args.replace_existing:raise RuntimeError('Existing plots preserved; inspect and use --replace-existing for approved visual repair')
            assert previous[:4] in (b'\x89PNG',b'%PDF')
    summaries={r['series']:r for r in payload['tables'][group+'_summary']}
    rows=payload['tables'][group+'_per_seed']
    fig,axes=plt.subplots(1,2,figsize=(11.8,6.2),sharey=True)
    fig.subplots_adjust(left=.285,right=.97,bottom=.35,top=.80,wspace=.18)
    fig.suptitle(title+'\nHistorical result reuse',x=.52,y=.975,fontsize=14,fontweight='bold',linespacing=1.5)
    for ax,metric,panel,title_x in zip(axes,['normalized_auc_map50_95','endpoint_map50_95'],['A','B'],['Normalized AUC, 5-40% budget','mAP50-95 at 40% budget']):
        for i,key in enumerate(order):
            data=sorted((r for r in rows if r['series']==key),key=lambda r:r['seed'])
            assert len(data)==6 and [r['seed'] for r in data]==list(range(6))
            ys=i+np.linspace(-.14,.14,6)
            ax.scatter([r[metric] for r in data],ys,s=25,color='#4477AA',alpha=.80,edgecolor='white',linewidth=.4,zorder=3)
            avg=summaries[key][metric+'_mean'];sd=summaries[key][metric+'_population_sd']
            ax.errorbar(avg,i,xerr=sd,fmt='D',markersize=5,color='#222222',capsize=4,lw=1.4,zorder=4)
        ax.set_xlim(0,1);ax.set_xticks(np.arange(0,1.01,.2));ax.set_ylim(len(order)-.48,-.48)
        ax.set_yticks(range(len(order)));ax.set_yticklabels(labels)
        ax.set_title(panel+'  '+title_x,loc='left',pad=14)
        ax.set_xlabel('Pose mAP50-95 scale (0-1)',labelpad=9)
        ax.grid(axis='x',color='#DFE3E7',lw=.6);ax.set_axisbelow(True)
        ax.tick_params(axis='y',length=0,pad=12)
        for spine in ['top','right','left']:ax.spines[spine].set_visible(False)
    handles=[Line2D([0],[0],marker='o',color='none',markerfacecolor='#4477AA',markeredgecolor='white',markersize=6,label='Six recorded training runs'),
             Line2D([0],[0],marker='D',color='#222222',markersize=5,lw=1.4,label='Mean +/- population SD (ddof=0)')]
    fig.legend(handles=handles,loc='lower center',bbox_to_anchor=(.60,.17),frameon=False,ncol=2,fontsize=9.1,handlelength=2)
    fig.text(.04,.126,'REHAB24-6 Camera17 internal validation. These are source-domain mAP results, not Penn Action PCK.',fontsize=9.1,color='#333333')
    fig.text(.04,.066,note,fontsize=9.0,color='#333333',linespacing=1.5)
    fig.savefig(targets[0],dpi=300)
    fig.savefig(targets[1],metadata={'Title':title+' - Historical result reuse','Subject':'Audited historical source-domain mAP; six records; no new experiment','Creator':'Matplotlib'})
    plt.close(fig)
    print(str(targets[0]))
    print(str(targets[1]))
