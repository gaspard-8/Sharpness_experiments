"""Build the PDF and HTML report from the audited offline analysis tables.

Uses the bundled document runtime (reportlab, numpy, pandas).
"""
import html
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak
from reportlab.graphics.shapes import Drawing, Line, Rect, String, Circle, PolyLine
from reportlab.graphics import renderSVG

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results/latest_learning_report'
DATA=OUT/'data'
PDF=ROOT/'output/pdf/learning_dynamics_report.pdf'
A=json.loads((DATA/'analysis.json').read_text())
P=json.loads((DATA/'plot_data.json').read_text())
T=pd.read_csv(DATA/'task_summary.csv')
D=pd.read_csv(DATA/'diagnostics.csv').set_index('_step')
PAIRS=pd.read_csv(DATA/'affinity_pairs.csv')
SEL=pd.read_csv(DATA/'checkpoint_selection.csv')
NAMES=T.task.tolist()
LABELS=T.label.tolist()
ABBR=['T55','T54','M29','P29','P10','M10','WV','SW']
PALETTE=['#C08212','#397E4A','#2865A8','#7043A3','#C0477A','#168C9E','#93644D','#DC6A36']
INK=colors.HexColor('#173042'); MUTED=colors.HexColor('#596C77'); GRID=colors.HexColor('#DFE6E9')
WIDTH=A4[0]-96


def txt(d,x,y,s,size=9,color=INK,anchor='start',bold=False):
    d.add(String(x,y,str(s),fontName='Helvetica-Bold' if bold else 'Helvetica',fontSize=size,fillColor=color,textAnchor=anchor))


def line(d,x1,y1,x2,y2,color=GRID,width=.6,dash=None):
    obj=Line(x1,y1,x2,y2,strokeColor=color,strokeWidth=width)
    if dash: obj.strokeDashArray=dash
    d.add(obj)


def poly(d,x,y,color,width=1):
    coords=[float(z) for pair in zip(x,y) for z in pair]
    if len(coords)>=4: d.add(PolyLine(coords,strokeColor=color,strokeWidth=width,fillColor=None))


def learning_figure():
    fig=Drawing(WIDTH,506)
    xs=np.array(P['steps']); logx=np.log10(xs)
    for i,n in enumerate(NAMES):
        col=i%2; row=i//2; left=col*252+34; bottom=403-row*121
        w=199; h=82
        X=lambda t:left+(np.log10(np.asarray(t))-1)/(math.log10(30000)-1)*w
        Y=lambda a:bottom+(np.asarray(a)-.3)/.7*h
        txt(fig,left,bottom+h+9,LABELS[i],9,bold=True)
        for y in [.5,.75,1.]:
            line(fig,left,Y(y),left+w,Y(y)); txt(fig,left-5,Y(y)-2,f'{y:.2f}',6.7,anchor='end',color=MUTED)
        for x in [100,1000,10000,30000]:
            line(fig,X(x),bottom,X(x),bottom+h,GRID,.4)
            txt(fig,X(x),bottom-12,{100:'100',1000:'1k',10000:'10k',30000:'30k'}[x],6.5,anchor='middle',color=MUTED)
        baseline=T.iloc[i].constant_baseline
        line(fig,left,Y(baseline),left+w,Y(baseline),colors.HexColor('#8A989F'),.7,[2,2])
        color=colors.HexColor(PALETTE[i])
        light=colors.Color(*[.78+.22*c for c in (color.red,color.green,color.blue)])
        raw=np.array(P['accuracies'][n]); smooth=np.array(P['smooth'][n])
        # All raw points, clipped to the visible accuracy range.
        poly(fig,X(xs),Y(np.clip(raw,.3,1)),light,.5)
        poly(fig,X(xs),Y(np.clip(smooth,.3,1)),color,1.45)
        for step,selected in zip(P['diagnostic_steps'],P['selected'][n]):
            if selected:
                a=raw[step//10-1]
                fig.add(Circle(float(X(step)),float(Y(a)),1.6,fillColor=color,strokeColor=colors.white,strokeWidth=.3))
    txt(fig,WIDTH/2,8,'Completed training steps (logarithmic scale)',8,anchor='middle',color=MUTED)
    return fig


def heatmap(mode):
    selected=PAIRS[(PAIRS.variant=='primary_1000_1pp') & (PAIRS['mode']==mode)]
    fig=Drawing(WIDTH,410); x0=84; y0=41; cell=42
    txt(fig,x0+4*cell,397,'TARGET FUNCTION',9,anchor='middle',bold=True)
    for j in range(8):
        txt(fig,x0+(j+.5)*cell,y0+8*cell+9,ABBR[j],8,anchor='middle',bold=True)
    for i in range(8):
        txt(fig,x0-9,y0+(7.5-i)*cell-3,ABBR[i],8,anchor='end',bold=True)
        for j in range(8):
            r=selected[(selected.source==NAMES[i])&(selected.target==NAMES[j])].iloc[0]
            usable=r.n>=10
            val=r['median']
            if usable:
                q=min(abs(val)/.15,1)
                target=np.array([.74,.16,.23]) if val<0 else np.array([.08,.47,.66])
                rgb=(1-q)*np.array([.975,.98,.98])+q*target
                fill=colors.Color(*rgb)
            else: fill=colors.HexColor('#EFF2F3')
            x=x0+j*cell; y=y0+(7-i)*cell
            fig.add(Rect(x,y,cell,cell,fillColor=fill,strokeColor=INK if i==j else colors.white,strokeWidth=.8 if i==j else .5))
            color=colors.white if usable and abs(val)>.1 else INK
            txt(fig,x+cell/2,y+23,f'{val*1000:+.1f}' if usable else '-',8,color,anchor='middle',bold=True)
            txt(fig,x+cell/2,y+10,f'n={int(r.n)}',6.3,color if usable else MUTED,anchor='middle')
    txt(fig,29,213,'SOURCE',8,bold=True)
    txt(fig,x0,22,'Cells: signed median affinity x 1,000 (MSE units)',8,color=MUTED)
    fig.add(Rect(x0,4,12,8,fillColor=colors.HexColor('#BD293B'),strokeColor=None));txt(fig,x0+17,4,'negative = loss increase',7,color=MUTED)
    fig.add(Rect(x0+181,4,12,8,fillColor=colors.HexColor('#1678A8'),strokeColor=None));txt(fig,x0+198,4,'positive = loss reduction',7,color=MUTED)
    return fig


def sharpness_figure():
    fig=Drawing(WIDTH,251); left=38; bottom=50; w=WIDTH-50; h=174
    X=lambda t:left+np.asarray(t)/30000*w
    Y=lambda a:bottom+np.asarray(a)/.65*h
    for y in [0,.2,.4,.6]:
        line(fig,left,Y(y),left+w,Y(y));txt(fig,left-6,Y(y)-3,f'{y:.1f}',7,anchor='end',color=MUTED)
    for x in [0,10000,20000,30000]: txt(fig,X(x),bottom-13,f'{x//1000}k',7,anchor='middle',color=MUTED)
    for i,n in enumerate(NAMES):
        series=pd.Series(P['sharpness'][n]).rolling(10,min_periods=3).median()
        ok=series.notna();poly(fig,X(np.array(P['diagnostic_steps'])[ok]),Y(series[ok]),colors.HexColor(PALETTE[i]),1.5)
        lx=38+(i%4)*114;ly=15-(i//4)*13
        line(fig,lx,ly+3,lx+12,ly+3,colors.HexColor(PALETTE[i]),2)
        txt(fig,lx+17,ly,ABBR[i],7)
    txt(fig,left,237,'Prediction-change sharpness',8,bold=True)
    return fig


def sensitivity_figure():
    fig=Drawing(WIDTH,244)
    panels=[('sustained95',20000,'Steps to 1,000-step median accuracy >= 95%'),('sharpness_late',.50,'Late sharpness (steps 20,000-30,000)')]
    for k,(column,ymax,title) in enumerate(panels):
        left=40+k*250;bottom=42;w=194;h=161
        X=lambda s:left+np.log10(np.asarray(s))/math.log10(30)*w
        Y=lambda y:bottom+np.asarray(y)/ymax*h
        txt(fig,left,222,'Learning time' if k==0 else 'Learned-model sharpness',10,bold=True)
        for y in ([0,5000,10000,15000,20000] if k==0 else [0,.1,.2,.3,.4,.5]):
            line(fig,left,Y(y),left+w,Y(y));txt(fig,left-6,Y(y)-2,f'{int(y/1000)}k' if k==0 else f'{y:.1f}',6.5,anchor='end',color=MUTED)
        for x in [1,2,5,10,29]:txt(fig,X(x),bottom-12,str(x),7,anchor='middle',color=MUTED)
        for i,r in T.iterrows():
            x=float(X(r.sensitivity));y=float(Y(r[column]));color=colors.HexColor(PALETTE[i])
            fig.add(Circle(x,y,3.2,fillColor=color,strokeColor=colors.white,strokeWidth=.6))
            offsets=[(4,5),(4,-10),(-22,-11),(-24,7),(5,5),(-22,5),(5,5),(5,5)]
            dx,dy=offsets[i]
            if k==0:
                offsets0=[(5,5),(5,-10),(4,-10),(-24,5),(5,5),(-22,6),(4,7),(4,5)]
                dx,dy=offsets0[i]
            txt(fig,x+dx,y+dy,ABBR[i],6.8,color)
        txt(fig,left+w/2,15,'Exact sensitivity (log scale)',7,anchor='middle',color=MUTED)
    return fig


BODY=ParagraphStyle('Body',fontName='Helvetica',fontSize=9.3,leading=13.2,textColor=INK,spaceAfter=7)
SMALL=ParagraphStyle('Small',parent=BODY,fontSize=8,leading=11,textColor=MUTED,spaceAfter=6)
HEAD=ParagraphStyle('Head',parent=BODY,fontName='Helvetica-Bold',fontSize=22,leading=26,spaceAfter=11)
SUB=ParagraphStyle('Sub',parent=BODY,fontName='Helvetica-Bold',fontSize=12,leading=16,spaceBefore=6,spaceAfter=6)
KICK=ParagraphStyle('Kick',parent=BODY,fontName='Helvetica-Bold',fontSize=8,leading=11,textColor=colors.HexColor('#227A8D'),spaceAfter=7)
CELL=ParagraphStyle('Cell',parent=BODY,fontSize=8,leading=10,spaceAfter=0)
CELLHEAD=ParagraphStyle('CellHead',parent=CELL,fontName='Helvetica-Bold',textColor=colors.white)

pages=[]
def new(kicker,title):
    page=[('kick',kicker),('title',title)];pages.append(page);return page
def p(page,text,small=False):page.append(('small' if small else 'p',text))
def sub(page,text):page.append(('sub',text))
def table(page,headers,rows,widths=None):page.append(('table',(headers,rows,widths)))
def figure(page,name,drawing):page.append(('figure',(name,drawing)))


page=new('RESEARCH NOTE | 6 OCTOBER 2026','Learning, affinity and sensitivity')
p(page,'<b>Latest extracted run: zz25goxp / job 122207.0.</b> Eight Boolean tasks, 29 payload bits plus 3 selector bits, 30,000 AdamW updates, seed 7. This report uses the saved run and archived source code, not the current metric definitions.')
sub(page,'What the run establishes')
p(page,'<b>Learning speed does not follow sensitivity alone.</b> Majority-29 learns first; the square wave follows quickly. Parity-29 reaches a 95% rolling-median accuracy at step 4,590, while the much less sensitive tribes require 15,310 and 17,800 steps. Their early 88% accuracy mostly reflects class imbalance.')
p(page,'<b>The affinity probe is too aggressive to read as ordinary training transfer.</b> During the selected learning periods, 90.6% of self-task probes and 90.7% of cross-task probes increase loss. Every one of the 56 directed cross-task medians is negative. This persists after excluding perfect-accuracy plateaus.')
p(page,'<b>Useful structure remains visible.</b> The tribes have strongly aligned gradients during joint learning (median cosine 0.80), despite negative finite-step affinities. Parity-to-parity help is weak and changes sign with the selection window. These are diagnostic observations, not evidence that joint training causes faster learning.')
p(page,'<b>Sharpness depends on what has been learned and when.</b> The two parity tasks are sharpest late in training (0.405 and 0.420). The tribes become sharper as they finally learn. The sensitivity-sharpness rank correlation drops from 0.83 in steps 10k-20k to 0.24 in steps 20k-30k.')
table(page,['Function','Sensitivity','95% time*','Final accuracy'],[[r.label,f'{r.sensitivity:.3f}',f'{r.sustained95:,}',f'{100*r.final_accuracy:.2f}%'] for r in T.itertuples()],[151,92,110,WIDTH-353])
p(page,'* First checkpoint at which the trailing 1,000-step median accuracy is at least 95%; evaluation is every 10 steps. This is a rolling-median milestone, not a claim that every subsequent reading stays above 95%.',True)
p(page,'Evidence: 3,000 fixed-set evaluations (320 held-out examples per task), 300 affinity matrices, 300 sharpness measurements. All 300 scalar affinity matrices exactly match the saved labeled tables.',True)

page=new('01 | OBSERVED LEARNING','When the functions are acquired')
p(page,'Faint lines show raw evaluation accuracy; solid lines are trailing 500-step medians. Dashed lines show each task\'s best constant predictor on its fixed evaluation set. Colored points mark checkpoints selected for the primary affinity analysis. The horizontal axis is logarithmic to retain the early transitions.')
figure(page,'learning_curves',learning_figure())
p(page,'<b>Early:</b> Majority-29 and the square wave acquire useful accuracy quickly. <b>Middle:</b> full parity rises before prefix Parity-10, despite sensitivity 29 versus 10. This is consistent with an architectural advantage for global counting over identifying a prefix, but the run does not test that mechanism.')
p(page,'<b>Late:</b> both tribes depart decisively from their constant baselines around 13k-16k steps. Tribe (5,5) continues improving after Tribe (5,4). Weighted voting retains a median accuracy of 98.75% over steps 20k-30k, although its final reading reaches 99.06%.')

page=new('02 | AFFINITY MEASUREMENT','Why negative does not mean incompatible')
p(page,'The archived metric takes a normalized descent direction for source i and moves a fixed Euclidean distance of 0.02 in parameter space:')
p(page,'<b>A(i,j) = L_j(theta) - L_j(theta - 0.02 g_i / ||g_i||).</b><br/>Positive values mean lower target MSE; negative values mean higher target MSE. This is an absolute loss difference, not a percentage. Source gradients and target losses use distinct fixed batches of 64 payloads. All trainable parameters, including positional embeddings, are perturbed.')
p(page,'The current README describes a newer, learning-rate-scaled relative affinity. That definition was <b>not used in this run</b>. The old late-training magnitude report also answers a different question; this report preserves the sign and selects improving, imperfect functions.')
sub(page,'The diagonal is the essential control')
table(page,['Source / target','Learning n','Median self A','Self-help rate'],[[r.label,str(r.active_n),f'{r.self_active_median:+.4f}',f'{r.self_active_positive_fraction*100:.1f}%'] for r in T.itertuples()],[164,79,114,WIDTH-357])
p(page,'Across 265 selected task-checkpoints, just 25 self-steps improve the source\'s evaluation-probe loss. All eight median self-affinities are negative. For Parity-10, the median self-step increases MSE by about 0.139 while accuracy is improving.')
p(page,'<b>Interpretation:</b> finite-step overshoot and differences between gradient and evaluation probes are plausible contributors. They cannot be separated with these logs. A negative cross-task entry therefore does not by itself establish a conflicting local gradient or harmful AdamW training transfer.')
p(page,'Gradient cosines in this report come from a separate diagnostic using 256 fixed payloads shared across tasks. They describe local loss-gradient alignment on that batch; they need not match a finite-step loss change measured with the two 64-payload affinity batches.',True)
normrat=[]
for n in NAMES:
    steps=SEL[(SEL.task==n)&SEL.selected].step
    normrat.append(float((.02/(1e-4*D.loc[steps,f'affinity/source_gradient_norm/{n}'])).median()))
p(page,f'As a scale reference, the median fixed-radius step is {min(normrat):.0f}-{max(normrat):.0f} times the length of a raw SGD step, lr x ||g_i||, at lr = 0.0001. This comparison does not estimate the actual AdamW step, whose moments and adaptive scaling are not logged.')

page=new('03 | DIRECTED AFFINITY DURING ACQUISITION','The target is still learning')
p(page,'Each column uses only checkpoints where that target is imperfect, improving, and clearly above its constant-predictor baseline. A source may already be learned. Values are signed medians; diagonals are outlined. Positive help and negative harm are never pooled as absolute magnitudes.')
figure(page,'target_active_affinity',heatmap('target_active'))
p(page,'<b>All 56 cross-task medians are negative.</b> Only 9.3% of the 1,855 selected cross-task observations show help. Parity-29 to Parity-10 is among the least harmful pairs (median -0.00228 MSE; positive in 28.2% of its 39 observations). “Least harmful” is not evidence of assistance.')
p(page,'Columns use different periods, so a direction and its reverse need not refer to the same model states. The next page uses simultaneous learning for a matched-time comparison.',True)
p(page,'Key: T55/T54 = Tribe (5,5)/(5,4); M29/M10 = Majority-29/10; P29/P10 = Parity-29/10; WV = weighted vote; SW = Hamming-weight square wave, band width 4.',True)

page=new('04 | MATCHED-TIME AFFINITY','When both functions are learning')
p(page,'Both source and target must pass the learning filter at the same checkpoint. Cells with fewer than 10 joint observations are left blank and retain their sample counts. Blank cells mean insufficient simultaneous-learning coverage, not zero interaction.')
figure(page,'joint_active_affinity',heatmap('both_active'))
p(page,'<b>Tribes:</b> 18 joint checkpoints; median gradient cosine 0.796. Affinities remain negative in every observation, with medians -0.0528 (T55 to T54) and -0.0377 (reverse). Shared local directions coexist with an overly damaging finite probe.')
p(page,'<b>Parity:</b> 20 joint checkpoints; P29 to P10 has median +0.00045, but helps in exactly 50%, its mean is negative, and its interquartile range crosses zero. The reverse median is -0.0111. The apparent forward help becomes negative with a 2,000-step learning window.')
p(page,'<b>Majority-29 / weighted vote:</b> 20 joint checkpoints; gradient cosine is mildly negative (-0.138), with harmful finite-step medians in both directions. This is evidence of local tension on the probes, not a quantified cost of multitask training.',True)

page=new('05 | PARAMETER PERTURBATIONS','Sharpness grows differently by task')
p(page,'The logged sharpness is the mean squared change in the model\'s scalar predictions under independent Gaussian noise (standard deviation 0.02 per non-positional parameter). It is not a Hessian eigenvalue or ground-truth input sensitivity. Each checkpoint uses 32 inputs per task and 4 shared noise draws.')
figure(page,'sharpness_history',sharpness_figure())
p(page,'Curves show trailing 10-checkpoint medians (1,000 steps). Summary below uses all 101 checkpoints from step 20,000 through 30,000. The interquartile range describes checkpoint variation, not uncertainty across independent runs.',True)
table(page,['Function','Early median*','Late median','Late IQR'],[[r.label,f'{r.sharpness_early:.3f}',f'{r.sharpness_late:.3f}',f'{r.sharpness_late_q25:.3f}-{r.sharpness_late_q75:.3f}'] for r in T.itertuples()],[147,102,100,WIDTH-349])
p(page,'* Early = steps 100-1,000 (10 checkpoints). Late median sharpness is approximately 7.1 times the early median for Parity-29, 9.8 times for Parity-10, and 5.7 / 5.0 times for the two tribes.',True)
p(page,'The parity tasks settle near 0.41-0.42, roughly 3.4-3.5 times Majority-29\'s 0.119. The low-sensitivity tribes end at 0.193-0.209, above the square wave (0.148) and both majorities. Flatness is therefore not a simple ranking of Boolean function sensitivity.')

page=new('06 | FUNCTION COMPLEXITY','Sensitivity is only part of the picture')
p(page,'Sensitivity here is computed exactly for the <b>target Boolean function</b>: the expected number of payload-bit flips that change its label under uniform inputs. Selector bits are held fixed. It is not measured input sensitivity of the trained network; the latest archive does not contain checkpoints needed for that calculation.')
figure(page,'sensitivity_relationships',sensitivity_figure())
p(page,'Left: learning speed uses the first 1,000-step rolling-median accuracy at 95%. Right: late sharpness. With only eight tasks, these comparisons are descriptive. Sensitivity is confounded with label balance, relevant positions, function family, and training stage.')
table(page,['Comparison across 8 tasks','Spearman rank correlation'],[
    ['Sensitivity vs. 95% learning time','-0.381'],
    ['Sensitivity vs. sharpness, steps 10k-20k','+0.833'],
    ['Sensitivity vs. sharpness, steps 20k-30k','+0.238'],
    ['Sensitivity vs. sharpness, each task\'s selected learning period','-0.238'],
],[344,WIDTH-344])
p(page,'<b>The high correlation at 10k-20k is stage-dependent.</b> The parity functions are already well learned while the tribes are only starting to improve. Later, tribe sharpness rises and several easier tasks become less sharp. Task-specific acquisition periods additionally compare different model states, so their correlation is not stage-controlled.')
p(page,'The square wave makes the same point from another direction: sensitivity 7.25, yet 95% rolling-median accuracy arrives at step 1,710. Parity-10 (sensitivity 10) reaches that milestone at 6,120, and Tribe (5,4), with sensitivity only 1.136, at 15,310.')

page=new('07 | SELECTION AND ROBUSTNESS','A reproducible definition of learning')
p(page,'Primary selection is applied separately to every task at each 100-step diagnostic checkpoint. It uses only evaluation readings at or before that checkpoint:')
p(page,'<b>1.</b> Current raw accuracy is below 1.<br/><b>2.</b> Within the trailing 1,000 steps, the mean of the last third of evaluation readings exceeds the first third by at least 1 percentage point.<br/><b>3.</b> The fitted linear slope over those readings is positive; at least 20 evaluations (200 steps) are available.<br/><b>4.</b> Current accuracy beats that task\'s best constant predictor on the fixed evaluation set by at least 2 percentage points.')
p(page,'The baseline condition prevents the initial all-negative tribe strategy from being presented as function learning. Short recovery episodes may still qualify; points are descriptive acquisition/recovery windows, not independent experiments. Perfect-accuracy points are excluded even if the backward-looking trend is positive.')
rob=[]
variant_labels={'primary_1000_1pp':'1,000 steps / 1 pp (primary)','window_500_gain_1pp':'500 steps / 1 pp','window_1000_gain_2pp':'1,000 steps / 2 pp','window_2000_gain_1pp':'2,000 steps / 1 pp','window_2000_gain_2pp':'2,000 steps / 2 pp','1000_1pp_without_baseline_gate':'1,000 / 1 pp, no baseline filter'}
for variant,label in variant_labels.items():
    g=PAIRS[(PAIRS.variant==variant)&(PAIRS['mode']=='target_active')&(PAIRS.source!=PAIRS.target)]
    positive=(g.positive_fraction.fillna(0)*g.n).sum()/g.n.sum()
    n=int(g.groupby('target').n.first().sum())
    available=int(g['median'].notna().sum())
    rob.append([label,str(n),f'{positive*100:.1f}%',f'{int((g["median"]>0).sum())}/{available}'])
table(page,['Selection window / required gain','Task-points','Help rate*','Positive medians'],rob,[222,68,74,WIDTH-364])
p(page,'* Pooled cross-task positive fraction; every selected target checkpoint contributes seven source observations. For the stricter 1,000-step / 2 pp filter, Tribe (5,5) has no selected points, leaving 49 rather than 56 estimable cross-task medians. Missing data are never treated as zero.',True)
p(page,'The conclusion that the finite-step probe is predominantly harmful survives every filter, including removal of the baseline condition. The parity joint-learning direction does not: its median is mildly positive for 500-1,000-step windows and negative for 2,000-step windows. It should remain a hypothesis for a better-scaled measurement.')
p(page,'<b>Measurement limits:</b> one seed; serially correlated checkpoints; 320 fixed held-out examples per task (accuracy increments of 0.3125 pp); two 64-example affinity batches; noisy sharpness with four perturbations. Accuracy 1 on this set does not establish exact correctness over all 2^29 payloads. Loss-space diagnostics were disabled in this run.')

page=new('08 | EXACT DEFINITIONS','Sensitivity and function balance')
p(page,'For a binary function f of uniform payload x, S(f) = sum over bits k of Pr[f(x) differs from f(x with bit k flipped)]. Ignored payload bits contribute zero. All values below use the saved function definitions, including strict majority ties going to zero.')
table(page,['Function','Exact calculation','S(f)','P(label=1)'],[
    ['Tribe (5,5)','25 x (1/16) x (31/32)^4','1.376154','0.146785'],
    ['Tribe (5,4)','20 x (1/16) x (31/32)^3','1.136436','0.119262'],
    ['Majority-29','29 x C(28,14) / 2^28','4.333933','0.500000'],
    ['Parity-29','All 29 bits are pivotal','29','0.500000'],
    ['Parity-10','First 10 bits are pivotal','10','0.500000'],
    ['Majority-10','10 x C(9,5) / 2^9','2.460938','0.376953'],
    ['Weighted vote','Exact subset-sum counts; see below','3.758801','0.500000'],
    ['Square wave','29 x sum C(28,k) / 2^28;<br/>k = 3,7,11,15,19,23,27','7.25','0.546498'],
],[105,226,73,WIDTH-404])
p(page,'<b>Tribes:</b> a bit matters only when the other four bits in its AND-group are 1 and all other groups are false. The two functions share their first four groups; (5,5) adds a fifth group. This explains their structural similarity without assuming that their learned parameterizations coincide.')
p(page,'<b>Weighted vote:</b> weights are the integers 1 through 29. A label is positive if the weighted subset sum is at least 218 (total weight 435). Bit i is pivotal exactly when the other bits sum to an integer from 218 - i through 217. A dynamic program counts those subsets; summing the 29 exact probabilities gives the reported sensitivity.')
p(page,'<b>Square wave:</b> the label is floor(Hamming weight / 4) modulo 2. A bit is pivotal when the other 28 bits have a count congruent to 3 modulo 4. Its oscillations are functions of a global count, unlike the position-specific subsets needed for prefix parity and tribes.')
p(page,'Theoretical tribe majority-class accuracies are 85.32% and 88.07%. The finite evaluation sets contain 38 and 39 positives out of 320, yielding constant baselines of 88.125% and 87.8125%. Selection uses these actual evaluation baselines; function sensitivity uses the exact uniform distribution.')

page=new('09 | PROVENANCE AND NEXT EXPERIMENTS','What would resolve the remaining ambiguity')
sub(page,'Most informative follow-up measurements')
p(page,'<b>Scale affinity to the update.</b> Re-run learning-phase diagnostics with the newer learning-rate-scaled affinity and a small step-size sweep. Retain absolute loss differences, relative changes, source and target validity, and the diagonal self-step controls. Compare fixed source/evaluation batches with a same-batch version to distinguish overshoot from probe mismatch.')
p(page,'<b>Measure actual progress and transfer.</b> Save checkpoints around early acquisition, the parity transitions, and the tribe transitions near 13k-18k. Measure target loss and accuracy changes after task-specific updates using a cloned optimizer state. Compare joint training with single-task or pair-removal controls across several seeds before claiming helpful or harmful transfer.')
p(page,'<b>Compare sharpness at comparable mastery.</b> Match tasks by accuracy and error reduction above their constant baseline, vary the noise scale, and repeat perturbations. Measure the trained network\'s input-bit sensitivity separately from the exact sensitivity of the target function.')
sub(page,'Source and verification')
p(page,'Primary data: <b>unzipped_results/results/wandb/wandb/run-20261006_123923-zz25goxp/run-zz25goxp.wandb</b>. Configuration and all 300 affinity tables were read from that same run directory. Job context identifies 122207.0, started 6 October 2026 at 12:39:19 UTC; saved exit code is 0.',True)
p(page,'Definitions: <b>unzipped_results/results/code/src/metrics.py</b> (task_affinity_metrics and average_direction_sharpness), <b>src/functions.py</b> within the same archived code directory, and archived training/evaluation code. The current working-tree README and metric implementation were not substituted for the archived definitions.',True)
p(page,'Three older runs are present in the extracted directory: wgrcwba5 and y9er89ad (10,000 steps, total length 24), and oo085cja (30,000 steps, total length 20). They have different task mixes and are not pooled as replications of the latest experiment.',True)
sub(page,'Reproducible companion files')
p(page,'The HTML companion and CSV tables are in <b>results/latest_learning_report/</b>. Tables include full evaluation histories, diagnostics, exact task summaries, all selection decisions, directed affinity summaries for every filter, and fixed-time affinity summaries. The analysis JSON records run configuration and the source-file SHA-256.',True)
p(page,'Recompute analysis with <b>env/bin/python scripts/analyze_latest_learning.py</b>. Build the PDF/HTML from the exported tables with <b>scripts/build_latest_learning_report.py</b> using a Python runtime containing reportlab, numpy and pandas. No network service or retraining is needed.',True)
p(page,'Scientific scope: these are measurements from one trained multitask model and one seed. Window-based IQRs and repeated checkpoints are descriptive; they are not independent-seed confidence intervals or causal transfer estimates.',True)


def footer(canvas,doc):
    canvas.saveState()
    canvas.setStrokeColor(GRID);canvas.setLineWidth(.5);canvas.line(48,40,A4[0]-48,40)
    canvas.setFont('Helvetica',7);canvas.setFillColor(MUTED)
    canvas.drawString(48,27,'SHARPNESS EXPERIMENTS  |  zz25goxp  |  6 October 2026')
    canvas.drawRightString(A4[0]-48,27,f'{doc.page} / {len(pages)}')
    canvas.restoreState()


def main():
    PDF.parent.mkdir(parents=True,exist_ok=True);(OUT/'figures').mkdir(exist_ok=True)
    story=[];htmlpages=[]
    for index,page in enumerate(pages):
        htmlparts=[]
        if index:story.append(PageBreak())
        for kind,content in page:
            if kind in ['kick','title','p','small','sub']:
                style={'kick':KICK,'title':HEAD,'p':BODY,'small':SMALL,'sub':SUB}[kind]
                story.append(Paragraph(content,style))
                tag={'title':'h1','sub':'h2'}.get(kind,'p')
                htmlparts.append(f'<{tag} class="{kind}">{content}</{tag}>')
            elif kind=='figure':
                name,drawing=content
                story.append(drawing);story.append(Spacer(1,7))
                renderSVG.drawToFile(drawing,str(OUT/'figures'/f'{name}.svg'))
                svg=renderSVG.drawToString(drawing)
                if isinstance(svg,bytes):svg=svg.decode()
                svg=svg[svg.index('<svg'):]
                htmlparts.append(f'<figure>{svg}</figure>')
            elif kind=='table':
                headers,rows,widths=content
                vals=[[Paragraph(str(x),CELLHEAD) for x in headers]]+[[Paragraph(str(x),CELL) for x in row] for row in rows]
                t=Table(vals,colWidths=widths or [WIDTH/len(headers)]*len(headers),hAlign='LEFT',repeatRows=1)
                t.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),INK),('ROWBACKGROUNDS',(0,1),(-1,-1),[colors.HexColor('#F1F5F6'),colors.white]),
                                      ('VALIGN',(0,0),(-1,-1),'TOP'),('LEFTPADDING',(0,0),(-1,-1),8),('RIGHTPADDING',(0,0),(-1,-1),7),
                                      ('TOPPADDING',(0,0),(-1,-1),6),('BOTTOMPADDING',(0,0),(-1,-1),6),
                                      ('LINEBELOW',(0,0),(-1,0),.5,colors.white)]))
                story.extend([t,Spacer(1,9)])
                htmlparts.append('<table><thead><tr>'+''.join(f'<th>{x}</th>' for x in headers)+'</tr></thead><tbody>'+''.join('<tr>'+''.join(f'<td>{x}</td>' for x in row)+'</tr>' for row in rows)+'</tbody></table>')
        htmlpages.append('<section>'+''.join(htmlparts)+'</section>')
    doc=SimpleDocTemplate(str(PDF),pagesize=A4,rightMargin=48,leftMargin=48,topMargin=43,bottomMargin=54,
                          title='Learning, affinity and sensitivity - zz25goxp',author='Sharpness experiments analysis')
    doc.build(story,onFirstPage=footer,onLaterPages=footer)
    styles='''body{margin:0;background:#e8eef0;color:#173042;font:16px/1.6 system-ui,sans-serif}main{max-width:920px;margin:auto;padding:28px 14px}section{background:white;margin:0 0 22px;padding:40px 48px;border-radius:10px;box-shadow:0 2px 12px #1730420a}h1{font-size:31px;line-height:1.15;margin:0 0 22px}h2{font-size:20px}.kick{font-size:12px;font-weight:700;color:#227a8d;letter-spacing:1px}.small{font-size:13px;color:#596c77}table{border-collapse:collapse;width:100%;font-size:13px;margin:20px 0}th,td{text-align:left;padding:10px;vertical-align:top}th{background:#173042;color:white}tr:nth-child(odd){background:#f1f5f6}figure{margin:20px 0}svg{width:100%;height:auto}a{color:#166b84}.downloads{padding:22px;background:white;border-radius:10px}@media(max-width:600px){section{padding:24px 18px}h1{font-size:26px}th,td{padding:6px;font-size:11px}}@media print{body{background:white}section{box-shadow:none;break-after:page}}'''
    links=['task_summary.csv','checkpoint_selection.csv','affinity_pairs.csv','evaluation_history.csv','diagnostics.csv','affinity_phases.csv','analysis.json']
    downloads='<div class="downloads"><b>Supporting data</b><ul>'+''.join(f'<li><a href="data/{name}">{name}</a></li>' for name in links)+'</ul></div>'
    (OUT/'report.html').write_text('<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Learning, affinity and sensitivity</title><style>'+styles+'</style></head><body><main>'+''.join(htmlpages)+downloads+'</main></body></html>')
    print(PDF)
    print(OUT/'report.html')


if __name__=='__main__':main()
