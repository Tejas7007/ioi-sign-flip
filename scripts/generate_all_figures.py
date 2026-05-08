"""Generate all 18 figures from result JSONs."""
import json, os, numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

os.makedirs("figures", exist_ok=True)
plt.rcParams.update({'font.size':11, 'axes.titlesize':13, 'axes.labelsize':12, 'legend.fontsize':9, 'figure.dpi':150, 'savefig.dpi':300})

def load(p):
    with open(p) as f: return json.load(f)
def save(name):
    plt.savefig('figures/%s.png'%name, bbox_inches='tight')
    plt.savefig('figures/%s.pdf'%name, bbox_inches='tight')
    plt.close(); print("  %s"%name)

p160=load('results/pythia_160m_component_emergence.json')
p410=load('results/pythia_410m_component_emergence.json')
p1b=load('results/pythia_1b_component_emergence.json')
pile=load('results/pythia_160m_pile_vs_synthetic.json')
mega=load('results/pythia_160m_mega_experiments.json')
ranks=load('results/pythia_160m_rank_progression.json')
ts=load('results/pythia_160m_trajectories_sensitivity_wang.json')
stan=load('results/stanford_gpt2_ioi.json')
pol=load('results/stanford_gpt2_polish.json')
vol=load('results/stanford_gpt2_projections_volatile.json')
poly=load('results/polypythias_ioi.json')
ri=load('results/retrain_ioi_analysis.json')
rd=load('results/retrain_160m_deep_analysis.json')

# Fig01: Universal dip
fig,ax=plt.subplots(figsize=(10,5))
ax.plot([r['step'] for r in p160['results']],[r['performance']['accuracy']*100 for r in p160['results']],'o-',color='#2196F3',label='Pythia-160M',ms=3,lw=1.5)
ax.plot([r['step'] for r in p410['results']],[r['performance']['accuracy']*100 for r in p410['results']],'s-',color='#FF9800',label='Pythia-410M',ms=3,lw=1.5)
ax.plot([r['step'] for r in p1b['results']],[r['performance']['accuracy']*100 for r in p1b['results']],'^-',color='#4CAF50',label='Pythia-1B',ms=3,lw=1.5)
sx=sorted(stan['part1_sweep'].keys(),key=lambda x:int(x.split('_')[1]))
ax.plot([int(s.split('_')[1]) for s in sx],[stan['part1_sweep'][s]['accuracy']*100 for s in sx],'D-',color='#E91E63',label='Stanford GPT-2 (alias)',ms=3,lw=1.5)
bx=sorted(stan['part3_second_seed'].keys(),key=lambda x:int(x.split('_')[1]))
ax.plot([int(s.split('_')[1]) for s in bx],[stan['part3_second_seed'][s]['accuracy']*100 for s in bx],'v-',color='#9C27B0',label='Stanford GPT-2 (battlestar)',ms=3,lw=1.5)
rx=sorted(ri['checkpoints'].keys(),key=lambda x:int(x.split('_')[1]))
ax.plot([int(s.split('_')[1]) for s in rx],[ri['checkpoints'][s]['accuracy']*100 for s in rx],'x-',color='#795548',label='Pythia-160M retrained (seed=42)',ms=3,lw=1)
ax.axhline(y=50,color='grey',ls='--',alpha=0.5,label='Chance')
ax.set_xlabel('Training Step');ax.set_ylabel('IOI Accuracy (%)');ax.set_title('Universal Below-Chance Dip Across Model Families and Seeds')
ax.set_xscale('log');ax.set_xlim(100,500000);ax.set_ylim(0,105);ax.legend(loc='lower right');ax.grid(True,alpha=0.3)
save('fig01_universal_dip')

# Fig02: PolyPythias
fig,axes=plt.subplots(1,3,figsize=(14,4.5),sharey=True)
ckpts=[0,512,1000,2000,3000,5000,8000,10000,33000,143000]
for idx,(title,labels,colors) in enumerate([("Different Seeds",['seed1','seed3','seed5'],['#1976D2','#388E3C','#F57C00']),("Data Order Only",['data-seed1','data-seed2','data-seed3'],['#7B1FA2','#C2185B','#00796B']),("Weight Init Only",['weight-seed1','weight-seed2','weight-seed3'],['#5D4037','#455A64','#BF360C'])]):
    ax=axes[idx]
    for label,color in zip(labels,colors):
        if label in poly and 'checkpoints' in poly[label]:
            s,a=[],[]
            for step in ckpts:
                sk='step_%d'%step
                if sk in poly[label]['checkpoints']:s.append(step);a.append(poly[label]['checkpoints'][sk]['accuracy']*100)
            ax.plot(s,a,'o-',color=color,label=label,ms=4,lw=1.5)
    ax.axhline(y=50,color='grey',ls='--',alpha=0.5);ax.set_xlabel('Training Step')
    if idx==0:ax.set_ylabel('IOI Accuracy (%)')
    ax.set_title(title);ax.set_xscale('log');ax.set_xlim(300,200000);ax.set_ylim(0,105);ax.legend(loc='lower right',fontsize=8);ax.grid(True,alpha=0.3)
plt.suptitle('IOI Dip Across 9 PolyPythias-160M Variants',fontsize=14,y=1.02);plt.tight_layout();save('fig02_polypythias')

# Fig03: High-res Stanford transition
fig,ax=plt.subplots(figsize=(10,5))
hr=pol['exp3_phase_transition'];hs=sorted(hr.keys(),key=lambda x:int(x.split('_')[1]))
ax.plot([int(s.split('_')[1]) for s in hs],[hr[s]['accuracy']*100 for s in hs],'o-',color='#E91E63',ms=3,lw=1,alpha=0.8)
v=vol.get('volatile_retest',{})
for sk,vv in v.items():
    ax.errorbar(int(sk.split('_')[1]),vv['accuracy_n600']*100,yerr=vv['std_err']*200,fmt='s',color='red',ms=6,capsize=3,zorder=5)
ax.axhline(y=50,color='grey',ls='--',alpha=0.5)
ax.axvspan(500,1450,alpha=0.08,color='blue',label='Descent');ax.axvspan(1450,2500,alpha=0.08,color='red',label='Noisy bottom');ax.axvspan(2500,5000,alpha=0.08,color='green',label='Noisy recovery')
ax.set_xlabel('Training Step');ax.set_ylabel('IOI Accuracy (%)');ax.set_title('High-Resolution IOI Transition in Stanford GPT-2 Small')
ax.set_xlim(400,5200);ax.set_ylim(0,60);ax.legend(loc='upper right');ax.grid(True,alpha=0.3);save('fig03_highres_transition')

# Fig04: Rank progression
fig,(ax1,ax2)=plt.subplots(1,2,figsize=(12,5))
so=['step_1000','step_2000','step_3000','step_5000','step_8000','step_143000'];sl=[1000,2000,3000,5000,8000,143000]
ir=[ranks['rank_progression'][s]['median_io_rank'] for s in so];sr=[ranks['rank_progression'][s]['median_s_rank'] for s in so]
ip=[ranks['rank_progression'][s]['mean_io_prob']*100 for s in so];sp=[ranks['rank_progression'][s]['mean_s_prob']*100 for s in so]
ax1.plot(sl,ir,'o-',color='#2196F3',label='IO rank',lw=2,ms=6);ax1.plot(sl,sr,'s-',color='#F44336',label='S rank',lw=2,ms=6)
ax1.set_xlabel('Step');ax1.set_ylabel('Median Rank');ax1.set_title('Name Token Rank');ax1.set_xscale('log');ax1.set_yscale('log');ax1.legend();ax1.grid(True,alpha=0.3);ax1.invert_yaxis()
ax2.plot(sl,ip,'o-',color='#2196F3',label='IO prob',lw=2,ms=6);ax2.plot(sl,sp,'s-',color='#F44336',label='S prob',lw=2,ms=6)
ax2.set_xlabel('Step');ax2.set_ylabel('Mean Probability (%)');ax2.set_title('Name Token Probability');ax2.set_xscale('log');ax2.legend();ax2.grid(True,alpha=0.3)
plt.suptitle('Pythia-160M: When Names Enter Consideration',fontsize=14,y=1.02);plt.tight_layout();save('fig04_rank_progression')

# Fig05: Original head trajectories
fig,(ax1,ax2)=plt.subplots(1,2,figsize=(12,5))
traj=ts['head_trajectories'];ss=sorted(traj.keys(),key=lambda x:int(x.split('_')[1]));steps=[int(s.split('_')[1]) for s in ss]
ax1.plot(steps,[traj[s].get('L8H9',{}).get('attn_S2',0) for s in ss],'o-',color='#F44336',label='to S2',lw=2,ms=4)
ax1.plot(steps,[traj[s].get('L8H9',{}).get('attn_IO',0) for s in ss],'s-',color='#2196F3',label='to IO',lw=2,ms=4)
ax1.set_xlabel('Step');ax1.set_ylabel('Attention');ax1.set_title('L8H9 Attention (seed=1234)');ax1.set_xscale('log');ax1.set_xlim(1,200000);ax1.legend();ax1.grid(True,alpha=0.3)
for h,c in [('L0H10','#FF9800'),('L8H9','#F44336'),('L1H8','#4CAF50')]:
    ax2.plot(steps,[traj[s].get(h,{}).get('attn_S2',0) for s in ss],'o-',color=c,label=h,lw=1.5,ms=3)
ax2b=ax2.twinx();ax2b.plot(steps,[traj[s].get('accuracy',0)*100 for s in ss],'--',color='grey',alpha=0.5);ax2b.set_ylabel('Acc (%)',color='grey')
ax2.set_xlabel('Step');ax2.set_ylabel('Attn to S2');ax2.set_title('Head Trajectories (seed=1234)');ax2.set_xscale('log');ax2.set_xlim(1,200000);ax2.legend(loc='upper left');ax2.grid(True,alpha=0.3)
plt.tight_layout();save('fig05_head_trajectories_original')

# Fig06: Mechanism comparison
fig,(ax1,ax2)=plt.subplots(1,2,figsize=(12,5))
np_=['L8H9\n(S-inhib)','L8H1\n(Name mover)','L9H1\n(Neg NM)'];dp=[5.743,0.526,-1.020]
b1=ax1.bar(range(3),dp,color=['#F44336','#2196F3','#FF9800'],edgecolor='black',lw=0.5)
ax1.set_xticks(range(3));ax1.set_xticklabels(np_,fontsize=9);ax1.set_ylabel('Projection (IO-S)');ax1.set_title('Pythia-160M (seed=1234): 10.8:1')
ax1.axhline(y=0,color='black',lw=0.5);ax1.grid(True,alpha=0.3,axis='y')
for b,v in zip(b1,dp):ax1.text(b.get_x()+b.get_width()/2,b.get_height()+0.15,'%.1f'%v,ha='center',fontsize=10,fontweight='bold')
spr=vol.get('stanford_projections',{})
ns=['L10H10\n(S-inhib)','L10H4\n(Name mover)','L11H11\n(Neg NM)']
ds=[spr.get('L10H10',{}).get('proj_diff',1.889),spr.get('L10H4',{}).get('proj_diff',0.798),spr.get('L11H11',{}).get('proj_diff',-1.152)]
b2=ax2.bar(range(3),ds,color=['#F44336','#2196F3','#FF9800'],edgecolor='black',lw=0.5)
ax2.set_xticks(range(3));ax2.set_xticklabels(ns,fontsize=9);ax2.set_ylabel('Projection (IO-S)');ax2.set_title('Stanford GPT-2: 2.4:1')
ax2.axhline(y=0,color='black',lw=0.5);ax2.grid(True,alpha=0.3,axis='y')
for b,v in zip(b2,ds):ax2.text(b.get_x()+b.get_width()/2,b.get_height()+0.08,'%.1f'%v,ha='center',fontsize=10,fontweight='bold')
plt.suptitle('S-Suppression vs IO-Copying: Cross-Family',fontsize=14,y=1.02);plt.tight_layout();save('fig06_mechanism_comparison')

# Fig07: Sensitivity
fig,(ax1,ax2)=plt.subplots(1,2,figsize=(12,5))
sens=ts['sensitivity'];taus=[0.005,0.01,0.02,0.05,0.1,0.2];step='step_143000'
if step in sens:
    nms=[sens[step]['thresholds']['tau_%.3f'%t]['name_movers'] for t in taus]
    negs=[sens[step]['thresholds']['tau_%.3f'%t]['negative_nm'] for t in taus]
    x=range(len(taus))
    ax1.bar([i-0.15 for i in x],nms,0.3,color='#2196F3',label='Name Movers');ax1.bar([i+0.15 for i in x],negs,0.3,color='#F44336',label='Negative NM')
    ax1.set_xticks(x);ax1.set_xticklabels(['%.3f'%t for t in taus]);ax1.set_xlabel('Threshold');ax1.set_ylabel('Heads');ax1.set_title('Step 143000');ax1.legend();ax1.grid(True,alpha=0.3,axis='y')
for tau in [0.02,0.1,0.2]:
    pcts=[sens[s]['thresholds']['tau_%.3f'%tau]['pct_classified']*100 if s in sens else 0 for s in ['step_512','step_1000','step_2000','step_3000','step_5000','step_143000']]
    ax2.plot(range(6),pcts,'o-',label='tau=%.2f'%tau,lw=2,ms=5)
ax2.set_xticks(range(6));ax2.set_xticklabels(['512','1K','2K','3K','5K','143K']);ax2.set_xlabel('Step');ax2.set_ylabel('% Classified');ax2.set_title('Rate Across Training');ax2.legend();ax2.grid(True,alpha=0.3)
plt.suptitle('Sensitivity of Ablation-Based Classification',fontsize=14,y=1.02);plt.tight_layout();save('fig07_sensitivity')

# Fig08: Pile vs synthetic
fig,ax=plt.subplots(figsize=(10,5))
ss_,sa_,ps_,pa_=[],[],[],[]
for r in pile['results']:
    if r['synthetic']:ss_.append(r['step']);sa_.append(r['synthetic']['accuracy']*100)
    if r['pile'] and r['pile']['accuracy'] is not None:ps_.append(r['step']);pa_.append(r['pile']['accuracy']*100)
ax.plot(ss_,sa_,'o-',color='#2196F3',label='Synthetic',ms=3,lw=1.5);ax.plot(ps_,pa_,'s-',color='#F44336',label='Pile',ms=3,lw=1.5)
ax.axhline(y=50,color='grey',ls='--',alpha=0.5);ax.set_xlabel('Step');ax.set_ylabel('Accuracy (%)');ax.set_title('Pythia-160M: Synthetic vs Pile')
ax.set_xscale('log');ax.set_xlim(1,200000);ax.set_ylim(0,105);ax.legend();ax.grid(True,alpha=0.3);save('fig08_pile_vs_synthetic')

# Fig09: Wang classification
fig,ax=plt.subplots(figsize=(10,5))
wang=ts['wang_classification'];ws=sorted(wang.keys(),key=lambda x:int(x.split('_')[1]));wsteps=[int(s.split('_')[1]) for s in ws]
for role,label,color in [('name_mover','Name Mover','#2196F3'),('s_inhibition','S-Inhibition','#F44336'),('duplicate_token','Dup Token','#4CAF50'),('previous_token','Prev Token','#FF9800')]:
    ax.plot(wsteps,[wang[s]['counts'].get(role,0) for s in ws],'o-',color=color,label=label,lw=2,ms=5)
ax.set_xlabel('Step');ax.set_ylabel('Heads');ax.set_title('Wang et al. Classification (seed=1234)');ax.set_xscale('log');ax.legend();ax.grid(True,alpha=0.3);save('fig09_wang_classification')

# Fig10: Pile ablation
fig,ax=plt.subplots(figsize=(8,5))
pd_=mega['exp_c_pile_ablation'];models=['pythia_160m','pythia_410m','pythia_1000m'];labels=['160M\n(L8H9)','410M\n(L4H6)','1B\n(L11H0)']
base=[pd_[m]['pile_baseline_acc']*100 for m in models];abl=[pd_[m]['pile_ablated_acc']*100 for m in models];diffs=[pd_[m]['pile_ablation_diff']*100 for m in models]
x=range(3);ax.bar([i-0.15 for i in x],base,0.3,color='#2196F3',label='Baseline');ax.bar([i+0.15 for i in x],abl,0.3,color='#F44336',label='Ablated')
ax.set_xticks(x);ax.set_xticklabels(labels);ax.set_ylabel('Pile Accuracy (%)');ax.set_title('Dominant Head Ablation on Natural IOI');ax.legend();ax.grid(True,alpha=0.3,axis='y')
for i,d in enumerate(diffs):ax.text(i,max(base[i],abl[i])+2,'%.1fpp'%d,ha='center',fontsize=10,fontweight='bold',color='red')
save('fig10_pile_ablation')

# Fig11: Mechanism summary table
fig,ax=plt.subplots(figsize=(11,6))
data=[['','Pythia-160M\n(seed=1234)','Pythia-160M\n(seed=42)','Stanford GPT-2'],
['Dominant head','L8H9','L2H6','L10H10'],['Mechanism','Direct S2 attn','Indirect (relay)','Direct S2 attn'],
['S2 attention','92.5%','0.3%','59.3%'],['Projection diff','+5.74','~0','+1.89'],
['Actual S-inhib','L8H9','L6H7','L10H10'],['S-inhib S2 attn','92.5%','70.1%','59.3%'],
['Name mover','L8H1','L7H7','L10H4'],['Acc @ step 10K','100%','87.7%','59.3%']]
ax.axis('off');table=ax.table(cellText=data[1:],colLabels=data[0],loc='center',cellLoc='center')
table.auto_set_font_size(False);table.set_fontsize(10);table.scale(1.2,1.8)
for j in range(4):table[(0,j)].set_facecolor('#E3F2FD');table[(0,j)].set_text_props(fontweight='bold')
ax.set_title('Cross-Seed and Cross-Family Mechanism Comparison',fontsize=14,pad=20);save('fig11_mechanism_summary')

# Fig12: Retrained trajectory
fig,ax=plt.subplots(figsize=(10,5))
rt=ri['checkpoints'];rs=sorted(rt.keys(),key=lambda x:int(x.split('_')[1]))
ax.plot([int(s.split('_')[1]) for s in rs],[rt[s]['accuracy']*100 for s in rs],'o-',color='#795548',ms=3,lw=1)
ax.axvline(x=600,color='#FF9800',ls=':',alpha=0.5,label='L1H9 era');ax.axvline(x=1400,color='#F44336',ls=':',alpha=0.5,label='L6H7 era');ax.axvline(x=2600,color='#9C27B0',ls=':',alpha=0.5,label='L2H6 era')
ax.axhline(y=50,color='grey',ls='--',alpha=0.5);ax.set_xlabel('Step');ax.set_ylabel('Accuracy (%)');ax.set_title('Retrained Pythia-160M (seed=42): 103 Dense Checkpoints')
ax.set_xscale('log');ax.set_xlim(8,12000);ax.set_ylim(0,100);ax.legend(loc='upper left');ax.grid(True,alpha=0.3);save('fig12_retrained_trajectory')

# Fig13: Head succession retrained
fig,(ax1,ax2)=plt.subplots(2,1,figsize=(12,8),sharex=True)
traj2=rd['exp2_trajectories'];ts2=sorted(traj2.keys(),key=lambda x:int(x.split('_')[1]));steps2=[int(s.split('_')[1]) for s in ts2]
for h,c,l in [('L6H7','#F44336','L6H7 (S-inhib)'),('L2H6','#9C27B0','L2H6 (relay)'),('L8H9','#2196F3','L8H9 (inactive)'),('L1H9','#FF9800','L1H9 (early)')]:
    ax1.plot(steps2,[traj2[s].get(h,{}).get('end_to_S2',0) for s in ts2],'-',color=c,label=l,lw=1.5)
ax1.set_ylabel('Attn to S2');ax1.set_title('Head Competition (seed=42)');ax1.legend(loc='upper left');ax1.grid(True,alpha=0.3)
ax2.plot(steps2,[traj2[s].get('L1H8',{}).get('s2_to_s1',0) for s in ts2],'-',color='#4CAF50',label='L1H8 S2->S1',lw=1.5)
ax2b=ax2.twinx();ax2b.plot(steps2,[traj2[s].get('accuracy',0)*100 for s in ts2],'--',color='grey',alpha=0.5);ax2b.set_ylabel('Acc (%)',color='grey')
ax2.set_xlabel('Step');ax2.set_ylabel('Attention');ax2.set_title('Duplicate Token Head + Accuracy');ax2.set_xscale('log');ax2.set_xlim(8,12000);ax2.legend(loc='upper left');ax2.grid(True,alpha=0.3)
plt.tight_layout();save('fig13_head_succession_retrained')

# Fig14: Projections retrained
fig,ax=plt.subplots(figsize=(10,5))
proj=rd['exp1_projections'];ps2=sorted(proj.keys(),key=lambda x:int(x.split('_')[1]));psteps=[int(s.split('_')[1]) for s in ps2]
for h,c,l in [('L6H7','#F44336','L6H7 (S-inhib)'),('L2H6','#9C27B0','L2H6 (relay)'),('L8H9','#2196F3','L8H9 (S-promoting)')]:
    ax.plot(psteps,[proj[s].get(h,{}).get('proj_diff',0) for s in ps2],'o-',color=c,label=l,ms=4,lw=1.5)
ax.axhline(y=0,color='black',lw=0.5);ax.set_xlabel('Step');ax.set_ylabel('Projection (IO-S)');ax.set_title('Output Projections (seed=42)')
ax.set_xscale('log');ax.legend();ax.grid(True,alpha=0.3);save('fig14_projections_retrained')

# Fig15: Linear probes
fig,ax=plt.subplots(figsize=(10,5))
probes=rd['exp6_linear_probes'];pss=sorted(probes.keys(),key=lambda x:int(x.split('_')[1]))
colors=['#E3F2FD','#90CAF9','#42A5F5','#1E88E5','#0D47A1']
for idx,step in enumerate(pss):
    accs=probes[step]['layer_accuracies'];layers=sorted(accs.keys(),key=lambda x:int(x.split('_')[1]))
    ax.plot([int(l.split('_')[1]) for l in layers],[accs[l]*100 for l in layers],'o-',color=colors[idx],label='Step %s'%step.split('_')[1],lw=2,ms=5)
ax.set_xlabel('Layer');ax.set_ylabel('Probe Accuracy (%)');ax.set_title('Linear Probes: IO Identity (seed=42)')
ax.set_xticks(range(12));ax.legend();ax.grid(True,alpha=0.3);ax.set_ylim(20,105);save('fig15_linear_probes')

# Fig16: Causal tracing
ct=rd['exp7_causal_tracing'];cts=sorted(ct.keys(),key=lambda x:int(x.split('_')[1]))
fig,axes=plt.subplots(1,len(cts),figsize=(5*len(cts),5),sharey=True)
if len(cts)==1:axes=[axes]
pc={'S1':'#F44336','S2':'#FF9800','IO':'#2196F3','END':'#4CAF50'}
for idx,step in enumerate(cts):
    ax=axes[idx];pr=ct[step]['position_recovery']
    for pos in ['S1','S2','IO','END']:
        ls_,rs_=[],[]
        for l in range(12):
            key='layer_%d_%s'%(l,pos)
            if key in pr:ls_.append(l);rs_.append(pr[key]['recovery_fraction']*100)
        if ls_:ax.plot(ls_,rs_,'o-',color=pc[pos],label=pos,ms=4,lw=1.5)
    ax.axhline(y=0,color='grey',ls='--',alpha=0.3);ax.axhline(y=100,color='grey',ls='--',alpha=0.3)
    ax.set_xlabel('Layer');ax.set_title('Step %s'%step.split('_')[1])
    if idx==0:ax.set_ylabel('Recovery (%)')
    ax.legend(fontsize=8);ax.grid(True,alpha=0.3);ax.set_xticks(range(12))
plt.suptitle('Causal Tracing: Position-Specific Recovery (seed=42)',fontsize=14,y=1.02);plt.tight_layout();save('fig16_causal_tracing')

# Fig17: Path patching
fig,ax=plt.subplots(figsize=(10,5))
pp=rd['exp3_path_patching'];step='step_10000'
if step in pp:
    heads=[h['head'] for h in pp[step]];d_io=[h['delta_IO'] for h in pp[step]];d_s2=[h['delta_S2'] for h in pp[step]]
    x=range(len(heads));ax.bar([i-0.15 for i in x],d_io,0.3,color='#2196F3',label='Change in IO attn');ax.bar([i+0.15 for i in x],d_s2,0.3,color='#F44336',label='Change in S2 attn')
    ax.set_xticks(x);ax.set_xticklabels(heads,rotation=45,ha='right');ax.axhline(y=0,color='black',lw=0.5)
    ax.set_ylabel('Attention Change');ax.set_title('Path Patching: Heads Dependent on L2H6 (Step 10000)');ax.legend();ax.grid(True,alpha=0.3,axis='y')
save('fig17_path_patching')

# Fig18: Original vs retrained
fig,(ax1,ax2)=plt.subplots(1,2,figsize=(12,5))
comp=rd['exp5_comparison'];orig=comp['original_seed1234'];retr=comp['retrained_seed42']
oh=[h['head'] for h in orig['top5_heads']];od=[h['delta'] for h in orig['top5_heads']]
rh=[h['head'] for h in retr['top5_heads']];rdd=[h['delta'] for h in retr['top5_heads']]
y=range(5);ax1.barh([i+0.15 for i in y],od,0.3,color='#2196F3',label='Original',alpha=0.7);ax1.barh([i-0.15 for i in y],rdd,0.3,color='#795548',label='Retrained',alpha=0.7)
ax1.set_yticks(y);ax1.set_yticklabels(['%s / %s'%(o,r) for o,r in zip(oh,rh)],fontsize=8)
ax1.set_xlabel('Delta IOI');ax1.set_title('Top 5 Heads');ax1.legend();ax1.grid(True,alpha=0.3,axis='x')
bars=ax2.bar(['Original\n(seed=1234)','Retrained\n(seed=42)'],[orig['accuracy']*100,retr['accuracy']*100],color=['#2196F3','#795548'],edgecolor='black')
ax2.set_ylabel('Accuracy (%)');ax2.set_title('Step 10000');ax2.set_ylim(0,105);ax2.grid(True,alpha=0.3,axis='y')
for b,a,h in zip(bars,[orig['accuracy']*100,retr['accuracy']*100],[oh[0],rh[0]]):
    ax2.text(b.get_x()+b.get_width()/2,b.get_height()+2,'%.1f%%\n(%s)'%(a,h),ha='center',fontsize=10)
plt.tight_layout();save('fig18_original_vs_retrained')

print("\nAll 18 figures generated!")
