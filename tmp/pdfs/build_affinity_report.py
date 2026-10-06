import json, math
from pathlib import Path
import numpy as np
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak, KeepTogether
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.graphics.shapes import Drawing, Rect, String, Line, PolyLine
from reportlab.lib.pagesizes import A4

ROOT=Path('/Users/gaspard/Documents/Sharpness_experiments')
OUT=ROOT/'results/affinity_report'
S=json.loads((OUT/'analysis.json').read_text())
R=next(r for r in S['runs'] if 'affinity' in r); A=R['affinity']; N=A['names']
M=np.array(A['magnitude']); Q1=np.array(A['q25']); Q3=np.array(A['q75']); REL=np.array(A['relative'])
SHORT=['Tribe 4x4','Tribe 4x3','Majority 17','Parity 5','Majority 5','Ordered vote','Parity 8','Majority 8']
IDs=[f'T{i+1}'for i in range(8)]
BLUE=colors.HexColor('#153D54'); TEAL=colors.HexColor('#007C91'); INK=colors.HexColor('#152D3A'); GREY=colors.HexColor('#5B6A72'); PALE=colors.HexColor('#EDF3F5')
styles=getSampleStyleSheet()
styles.add(ParagraphStyle(name='TitleX',fontName='Helvetica-Bold',fontSize=28,leading=31,textColor=BLUE,spaceAfter=16))
styles.add(ParagraphStyle(name='DeckX',fontName='Helvetica',fontSize=13,leading=18,textColor=GREY,spaceAfter=16))
styles.add(ParagraphStyle(name='HeadX',fontName='Helvetica-Bold',fontSize=17,leading=21,textColor=BLUE,spaceAfter=12))
styles.add(ParagraphStyle(name='SubX',fontName='Helvetica-Bold',fontSize=11,leading=14,textColor=TEAL,spaceBefore=10,spaceAfter=6))
styles.add(ParagraphStyle(name='BodyX',fontName='Helvetica',fontSize=10,leading=14,textColor=INK,spaceAfter=9))
styles.add(ParagraphStyle(name='SmallX',fontName='Helvetica',fontSize=8.3,leading=11,textColor=GREY,spaceAfter=7))
styles.add(ParagraphStyle(name='CellX',fontName='Helvetica',fontSize=8.3,leading=11,textColor=INK))
styles.add(ParagraphStyle(name='CellHeadX',fontName='Helvetica-Bold',fontSize=8.3,leading=11,textColor=colors.white))
styles.add(ParagraphStyle(name='MathX',fontName='Courier',fontSize=9,leading=14,textColor=BLUE,spaceAfter=10,leftIndent=9))
story=[]
def p(t,style='BodyX'): return Paragraph(t,styles[style])
def add(t,style='BodyX'):story.append(p(t,style))
def head(t):add(t,'HeadX')
def sub(t):add(t,'SubX')
def page():story.append(PageBreak())
def table(rows,widths):
 data=[[p(str(c),'CellHeadX' if i==0 else 'CellX')for c in row]for i,row in enumerate(rows)]
 t=Table(data,colWidths=widths,hAlign='LEFT',repeatRows=1)
 t.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),BLUE),('VALIGN',(0,0),(-1,-1),'TOP'),('LEFTPADDING',(0,0),(-1,-1),7),('RIGHTPADDING',(0,0),(-1,-1),7),('TOPPADDING',(0,0),(-1,-1),7),('BOTTOMPADDING',(0,0),(-1,-1),7),('ROWBACKGROUNDS',(0,1),(-1,-1),[colors.white,PALE]),('LINEBELOW',(0,0),(-1,0),.6,BLUE)]))
 story.append(t);story.append(Spacer(1,9))
def nm(s):return SHORT[N.index(s)]
def footer(c,doc):
 c.setStrokeColor(colors.HexColor('#CEDAE0'));c.line(43,39,552,39)
 c.setFont('Helvetica',8);c.setFillColor(GREY);c.drawString(43,26,'Sharpness experiments | Affinity analysis | 6 October 2026');c.drawRightString(552,26,str(doc.page))
def line_chart(series,labels,title,log=False,height=190):
 d=Drawing(500,height);x0=47;y0=35;w=427;h=height-69
 points=[v for s in series for _,v in s if v>0 or not log]
 z=lambda v: math.log10(max(v,1e-8)) if log else v
 low=math.floor(min(map(z,points))) if log else 0
 high=math.ceil(max(map(z,points))) if log else max(points)*1.12
 palette=[TEAL,colors.HexColor('#BB7043'),colors.HexColor('#6266AB')]
 d.add(String(0,height-12,title,fontName='Helvetica-Bold',fontSize=10,fillColor=BLUE))
 for tick in np.linspace(low,high,5):
  y=y0+h*(tick-low)/(high-low)
  d.add(Line(x0,y,x0+w,y,strokeColor=colors.HexColor('#DEE7EB'),strokeWidth=.5))
  text=(f'1e{tick:.0f}'if log else f'{tick:.3f}')
  d.add(String(x0-6,y-3,text,textAnchor='end',fontSize=7,fillColor=GREY))
 for tick in [0,10000,20000,30000]:
  x=x0+w*tick/30000;d.add(String(x,y0-12,f'{tick/1000:.0f}k',textAnchor='middle',fontSize=8,fillColor=GREY))
 for index,s in enumerate(series):
  coords=[]
  for x,y in s:coords.extend([x0+w*x/30000,y0+h*(z(y)-low)/(high-low)])
  d.add(PolyLine(coords,strokeColor=palette[index],strokeWidth=1.7))
  d.add(Rect(48+index*145,5,9,5,fillColor=palette[index],strokeColor=None));d.add(String(62+index*145,4,labels[index],fontSize=8,fillColor=INK))
 return d

def heatmap():
 d=Drawing(500,385);x0=65;y0=65;cell=44;maxv=.042
 d.add(String(65,370,'Target task (column)',fontName='Helvetica-Bold',fontSize=10,fillColor=BLUE))
 d.add(String(0,347,'Source',fontName='Helvetica-Bold',fontSize=9,fillColor=BLUE))
 for i in range(8):
  d.add(String(x0-10,y0+(7-i)*cell+17,IDs[i],textAnchor='end',fontName='Helvetica-Bold',fontSize=10,fillColor=BLUE))
  d.add(String(x0+i*cell+22,y0+8*cell+7,IDs[i],textAnchor='middle',fontName='Helvetica-Bold',fontSize=10,fillColor=BLUE))
  for j in range(8):
   v=M[i,j];t=min(v/maxv,1);col=colors.Color(.95*(1-t)+.03*t,.98*(1-t)+.43*t,.98*(1-t)+.54*t)
   x=x0+j*cell;y=y0+(7-i)*cell
   d.add(Rect(x,y,cell,cell,fillColor=col,strokeColor=colors.white,strokeWidth=1))
   if i==j:d.add(Rect(x+1,y+1,cell-2,cell-2,fillColor=None,strokeColor=BLUE,strokeWidth=1))
   d.add(String(x+22,y+18,f'{v*1000:.2f}',textAnchor='middle',fontName='Helvetica-Bold'if i==j else 'Helvetica',fontSize=9,fillColor=colors.white if t>.57 else INK))
 d.add(String(65,44,'Values: median |affinity| x 1,000; larger = stronger loss disturbance.',fontSize=8.5,fillColor=GREY))
 d.add(String(65,29,'Outlined diagonal = source task acting on itself.',fontSize=8.5,fillColor=GREY))
 return d

def bars(values,title,height=225):
 d=Drawing(500,height);x0=109;w=329;maxv=max(values)*1.15
 d.add(String(0,height-12,title,fontName='Helvetica-Bold',fontSize=10,fillColor=BLUE))
 order=np.argsort(values)[::-1]
 for row,i in enumerate(order):
  y=height-43-row*22
  d.add(String(x0-7,y+3,SHORT[i],textAnchor='end',fontSize=8,fillColor=INK))
  d.add(Rect(x0,y,w*values[i]/maxv,13,fillColor=TEAL,strokeColor=None))
  d.add(String(x0+w*values[i]/maxv+5,y+3,f'{values[i]*1000:.2f}',fontSize=8,fillColor=INK))
 d.add(String(x0,7,'Average of seven cell medians; units: MSE x 1,000.',fontSize=8,fillColor=GREY))
 return d

add('How strongly do the tasks disturb one another?','TitleX')
add('Experiment report with a focus on the affinity matrix. Typical magnitudes are compared after learning; transient spikes do not drive the conclusions.','DeckX')
add('<b>Principal result.</b> The learned-phase matrix is almost entirely negative, but its size has clear structure. Typical cross-task effects range from <b>0.000487 to 0.02779 MSE</b>, a <b>57-fold</b> difference. The sign alone hides this structure.')
add('<b>Strongest directed effect:</b> Majority 5 &rarr; ordered vote (0.02779). The tribe pair and the parity pair also show strong effects in both directions. Majority 5 has the largest average outgoing disturbance (0.01044); it is also the most affected target on that average (0.00960).')
add('<b>Interpretation.</b> Every source moves the same parameter distance, 0.02, even when its loss gradient is small. All eight self-effects are typically harmful. These results describe sensitivity to a finite hypothetical step around a learned solution; they do not establish that actual AdamW updates are hurting learning.')
# Bin evaluation loss medians to keep spikes from dominating the trajectory.
ev=np.array(R['eval_series']);curve=[]
for lo in range(0,30000,500):
 v=ev[(ev[:,0]>=lo)&(ev[:,0]<lo+500)]
 if len(v):curve.append((np.median(v[:,0]),np.median(v[:,1])))
story.append(line_chart([curve],['Held-out loss'],'Learning curve: 500-step median evaluation loss',True,180))
add('Run oo085cja | 30,000 updates | 300 affinity checkpoints | Primary comparison: 197 learned checkpoints at steps 10,000-30,000. Final evaluation accuracy: 100%; final MSE: 0.00001593.','SmallX')
page()

head('What is measured, and how spikes are handled')
sub('A directed, finite-step loss difference')
add('For source task i, its mean-loss gradient is normalized to a unit vector u_i. The diagnostic then evaluates every target j after a hypothetical parameter displacement of length r = 0.02.')
add('u_i = g_i / ||g_i||<br/>A_ij = L_j(theta) - L_j(theta - r*u_i)','MathX')
add('Rows are sources; columns are targets. Positive A means the target loss improves; negative A means it rises. The diagonal measures the source acting on itself. The report compares <b>H_ij = median_t |A_ij(t)|</b>, with signed medians exported separately. For the primary matrix, all 64 signed medians are negative.')
sub('Comparison cohort')
add('The first checkpoint with every task at least 99% accurate is step 5,410. To allow an additional settling period, the main window is steps 10,000-30,000 inclusive: 201 affinity checkpoints. A checkpoint is retained only if <b>every task has held-out accuracy at least 99%</b>. Four short performance excursions fail this rule: steps 10,600, 17,900, 24,700 and 28,400. This leaves 197 checkpoints.')
add('Medians suppress extreme affinity readings without selecting individual entries by their signs. Interquartile ranges describe typical temporal variation. A 10% trimmed mean removes the lowest and highest 19 absolute values separately from each 197-point series; it serves as a second summary. Ungated and later-window comparisons check that the conclusions are not an artifact of the four exclusions.')
add('The filtering rule is conditional on the model being learned. It intentionally does not quantify the frequency or severity of failures. IQRs are variation over time, not statistical confidence intervals. All measurements use the same seed and repeated fixed probes.','SmallX')
sub('Task key')
table([['ID','Readable name','Logged task'],*[ [IDs[i],SHORT[i],N[i]]for i in range(8)]],[36,145,314])
add('The model uses 20 tokens: 3 selector bits and a 17-bit payload. The overall majority task covers all 17 payload bits. All tasks are sampled with equal probability. Ordered vote uses signed -1/+1 targets; the other tasks use 0/1 targets.','SmallX')
page()

head('Typical affinity magnitude matrix')
add('The comparison below is in raw mean-squared-loss units. Multiply each displayed number by 0.001. A darker cell means that source task causes a larger typical change in the target loss. Every signed cell median is negative; the figure shows magnitude so that relative size is legible.')
story.append(heatmap())
table([['ID','Task','ID','Task'],*[ [IDs[i],SHORT[i],IDs[i+4],SHORT[i+4]]for i in range(4)]],[35,210,35,215])
add('<b>Visible structure.</b> T1-T2 (tribes) and T4-T7 (parity) form strong pairs. T5-T6 (Majority 5 and ordered vote) also has strong transfer disturbance. Many tribe-to-parity effects are much smaller. The diagonal, at 0.0203-0.0408, is generally larger than cross-task effects.')
add('Across all retained off-diagonal observations, the median absolute effect is 0.00459 MSE; the middle 50% is 0.00208-0.01091. This pooled summary differs from averaging the 56 separate cell medians.','SmallX')
page()

head('Which directed effects are strongest?')
add('Direction matters: i &rarr; j and j &rarr; i can have different magnitudes. The table lists the largest cell medians and their temporal IQRs. All numbers below are MSE x 1,000.')
rows=[['Source -> target','Magnitude','IQR','Reverse']]
for v in A['pairs'][:10]:rows.append([f"{nm(v['source'])} -> {nm(v['target'])}",f"{v['magnitude']*1000:.2f}",f"{v['q25']*1000:.2f}-{v['q75']*1000:.2f}",f"{v['reverse']*1000:.2f}"])
table(rows,[240,70,100,85])
sub('Three substantial pair relationships')
add('<b>Tribe 4x4 and Tribe 4x3:</b> 0.02316 versus 0.01940, respectively. These are closely related functions, so the effect shows that a shared or aligned direction can substantially disturb another already-learned task.')
add('<b>Parity 8 and Parity 5:</b> 0.02298 versus 0.02047. The pair remains among the strongest in the final 5,000 updates, at 0.01881 and 0.01672. Its magnitude decreases as training progresses, while the pair relationship remains.')
add('<b>Majority 5 and ordered vote:</b> 0.02779 versus 0.01767. Majority 5 &rarr; ordered vote is 1.57 times the reverse direction. Its broad IQR reflects temporal variability; a highest rank does not mean it is largest at every checkpoint.')
sub('Small effects are informative too')
add('Parity 5 &rarr; Tribe 4x4 is the weakest median cross-task effect (0.000487). Parity 8 &rarr; Tribe 4x4 is also small (0.000830). The largest effect is 57 times the smallest. These weak cells distinguish relatively protected targets and directions within an otherwise negative matrix.')
add('These are disturbances at one finite parameter radius. They are not estimates of how much a target would forget after one normal training update.','SmallX')
page()

head('Source disturbance and target vulnerability')
add('Each score averages seven off-diagonal cell medians with equal weight. An outgoing score answers how strongly one source typically disturbs the other tasks. An incoming score answers how strongly a target is typically disturbed by the other sources.')
src=[x['source_score']for x in A['task_summary']];tar=[x['target_score']for x in A['task_summary']]
story.append(bars(src,'Outgoing: average disturbance caused',211));story.append(bars(tar,'Incoming: average disturbance received',211))
add('<b>Outgoing:</b> Majority 5 ranks first, followed by Tribe 4x3 and Tribe 4x4. Its score is about 1.90 times the smallest source score, overall Majority 17. <b>Incoming:</b> Majority 5 ranks first; the other targets fall in a narrower band, approximately 0.00667-0.00788.')
sub('Scaling by target self-sensitivity')
add('A complementary measure divides H_ij by H_jj. Averaged over other sources, this ratio is largest for Majority 5 (<b>47.2%</b>), then Majority 17 (32.0%) and Tribe 4x3 (31.1%). Ordered vote and Parity 8 are near 16.3%. Thus ordered vote receives one very large effect but is also highly self-sensitive.')
add('This ratio compares medians at the same radius. It is a sensitivity reference, not a causal correction for curvature. The ordered task also uses a different label scale, so raw MSE rankings should not be read as scale-free task rankings.','SmallX')
page()

head('Do the conclusions survive spike suppression?')
add('The main findings survive both the removal of low-accuracy checkpoints and alternative time windows. Spearman rank correlation below compares the ordering of the 56 off-diagonal cell magnitudes against the primary matrix.')
rows=[['Window','Checkpoints','Rank correlation','Largest outgoing source']]
for w in A['windows']:
 label=f"{w['start']:,}-30,000"+(' | learned'if w['gate']else' | all')
 rows.append([label,w['n'],f"{w['rank_correlation']:.3f}",SHORT[int(np.argmax(w['source_scores']))]])
table(rows,[188,77,98,132])
add('Keeping all 201 checkpoints in the main time window yields a rank correlation of <b>0.9996</b>. The maximum change in any matrix-cell median is 7.8%. The spike exclusions therefore have little influence on the typical matrix. The 10% trimmed-mean magnitudes have a rank correlation of <b>0.990</b> with the primary medians.')
add('The final 5,000-update window has a rank correlation of <b>0.966</b>. Majority 5 remains the largest outgoing source in every window. The strongest directed effect remains Majority 5 &rarr; ordered vote, although in the final window it is almost tied with Tribe 4x4 &rarr; Tribe 4x3: 0.02345 versus 0.02323.')
sub('Magnitude does evolve after classification is learned')
curve=np.array(A['curve']);story.append(line_chart([curve[:,[0,1]].tolist(),curve[:,[0,2]].tolist()],['Cross-task magnitude','Self magnitude'],'1,000-step medians across checkpoints and matrix cells',False,200))
add('The curve uses temporal medians without the accuracy filter so it is a separate overview, not a plot of the primary cohort. Values may shrink over time; some smaller cells change considerably in relative terms. The report therefore emphasizes stable broad relationships rather than a fixed ranking of every small effect.','SmallX')
page()

head('Why negative affinity can coexist with learning')
sub('Finite steps near a learned solution')
add('The normalized step has length 0.02 even if ||g_i|| becomes small. A local Taylor expansion of the target loss gives:')
add('A_ij approximately r * grad(L_j).u_i<br/>                  - (r^2/2) * u_i^T H_j u_i','MathX')
add('When the first-order improvement is small and curvature in that direction is positive, the second term can dominate, giving negative affinity. The independently sampled source and target probes can add another mismatch. In this dataset, <b>100% of retained diagonal observations are negative</b>. Typical self-disturbance is 0.0203-0.0408, far above baseline task probe losses, roughly 0.00001-0.00008.')
add('This supports finite-radius sensitivity or overshoot as an explanation. The data do not separate the curvature term from probe mismatch, and no radius sweep is available to test the explanation directly.')
sub('Affinity and cosine alignment give different answers')
add('In the same late time window, <b>27 of 28 pairwise median gradient cosines are positive</b>. The tribe pair has cosine 0.690; the parity pair has cosine 0.652; Majority 5 versus ordered vote has cosine 0.500. Their strong negative finite-step affinities therefore should not be labeled pervasive first-order gradient conflict. The cosine and affinity diagnostics also use different fixed probes (256 versus 64 inputs).')
sub('Ordered vote is almost a constant task here')
add('The function returns +1 when Isordered on the full 17-bit payload is strictly positive. Exact enumeration gives <b>18 negative payloads among 131,072</b>, or <b>0.0137%</b>. Recreating the seeded probes gives zero negative cases in both affinity sets (64 + 64 inputs), and zero among its 320 held-out inputs.')
add('Its 100% held-out accuracy from early training does not show recognition of those rare negatives. The large Majority 5 &rarr; ordered-vote cell still measures real loss sensitivity on the logged positive examples, but its semantic interpretation is sensitivity of a nearly constant signed-output task.')
add('All conclusions concern a functional diagnostic. It preserves the trained parameters and does not apply a real update. AdamW uses adaptive moments and weight decay, unlike the normalized raw-gradient displacement.','SmallX')
page()

head('Experiment context, evidence and useful next checks')
sub('What can be compared across the archived experiments')
rows=[['Run','Task family / updates','Final eval MSE','Final accuracy','Affinity']]
for run in S['runs']:
 c=run['config'];family='Mixed continuous tasks'if run['id']=='wgrcwba5'else('Eight tribe variants'if run['id']=='y9er89ad'else'Mixed binary tasks')
 rows.append([run['id'],f"{family}<br/>{c['train_num_steps']:,} updates",f"{run['final']['eval/loss']:.6g}",f"{run['final']['eval/accuracy']*100:.2f}%",'300 checkpoints'if'affinity'in run else'Not logged'])
table(rows,[68,158,89,80,100])
add('All three use the same seed, 4-layer transformer with model width 64, AdamW learning rate 0.0001 and weight decay 0.01. Task definitions, payload lengths, targets and training duration differ. The older runs provide learning and final loss-space context; they cannot establish whether the newest affinity matrix is better or worse, because they did not log this metric.')
add('Final loss-space overlap reports in the older runs are separate diagnostics. Their interference matrices count geometric overlap directions; they are not the directed finite-step affinity matrix analyzed here. The newest run disabled final loss-space searches.')
sub('Evidence and reproducibility')
add('This report uses only local archived experiment data. All 300 scalar affinity matrices were checked for exact agreement with their corresponding saved W&B matrix tables. Every source was valid, all entries were finite, and the logged radius was 0.02 throughout. No trained model weights were available for a new measurement.')
add('Primary source: unzipped_results/results/wandb/wandb/run-20261006_101155-oo085cja/, including run-oo085cja.wandb, config.yaml, wandb-summary.json and the saved affinity tables. Definition and implementation: src/metrics.py, task_affinity_metrics; task labels: src/functions.py; experiment setup: test.py.','SmallX')
add('Reproduce the numerical analysis with scripts/summarize_affinity.py. CSV exports in results/affinity_report include signed and magnitude matrices, IQR bounds, trimmed means, target-self ratios, task summaries, directed-pair rankings and the checkpoint selection. analysis.json contains the window comparisons and plotted summaries.','SmallX')
sub('Three follow-up experiments that would resolve remaining questions')
add('<b>1. Sweep the radius.</b> Measure the same checkpoints and probes at 0.002, 0.005, 0.01 and 0.02. A quadratic dependence or a sign reversal at smaller radii would help distinguish overshoot from first-order conflict.')
add('<b>2. Compare with actual optimizer displacements.</b> Evaluate transfer under the recorded AdamW update scale and direction to test whether the diagnostic ranks actual update interference.')
add('<b>3. Include ordered-vote negatives deliberately.</b> Measure separate positive and rare-negative strata. This would test the branch absent from the current held-out and affinity probes; preserving their natural prevalence separately would retain comparability.')

pdf=OUT/'experiment_affinity_report.pdf'
doc=SimpleDocTemplate(str(pdf),pagesize=A4,rightMargin=43,leftMargin=43,topMargin=45,bottomMargin=53,title='Sharpness experiments: affinity magnitude report',author='Experiment analysis')
doc.build(story,onFirstPage=footer,onLaterPages=footer)
print(pdf)
