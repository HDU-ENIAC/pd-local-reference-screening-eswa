"""Plot experimental trends from replayed numerical data (not typeset artwork)."""
import argparse
from pathlib import Path
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument('--results',type=Path,default=Path('outputs/result_replay'))
    p.add_argument('--output',type=Path,default=Path('outputs/plots'))
    a=p.parse_args()
    a.output.mkdir(parents=True,exist_ok=False)
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    d=pd.read_csv(a.results/'figure_4.csv')
    fig,axes=plt.subplots(2,2,figsize=(8,5.4),layout='constrained')
    for col,sweep in enumerate(['normal_budget','alpha']):
        for row,(metric,ylabel) in enumerate([('retention','Event retention (%)'),('fpr','False-positive rate (%)')]):
            for method,label,marker in [('proposed','Proposed','o'),('pd_cnn','PD CNN','s')]:
                v=d[(d.method==method)&(d.sweep==sweep)].sort_values(sweep)
                axes[row,col].plot(v[sweep],v[metric]*100,marker=marker,label=label)
            axes[row,col].set(xlabel='Calibration windows N' if col==0 else 'Nominal level alpha',ylabel=ylabel)
    axes[0,0].legend(frameon=False)
    fig.savefig(a.output/'figure_4.png',dpi=180);plt.close(fig)
    d=pd.read_csv(a.results/'figure_5.csv')
    wide=d.pivot(index=['site','record'],columns='reference',values='auc')
    fig,ax=plt.subplots(figsize=(4,3.5),layout='constrained')
    for _,v in wide.iterrows():
        ax.plot([0,1],[v['matched'],v['replacement']],'-o',alpha=.6,linewidth=1)
    ax.set(xticks=[0,1],xticklabels=['Matched','Other record'],ylabel='Within-record AUC',ylim=(0,1.02))
    fig.savefig(a.output/'figure_5.png',dpi=180);plt.close(fig)
    d=pd.read_csv(a.results/'figure_6.csv')
    fig,axes=plt.subplots(1,2,figsize=(7,3.2),layout='constrained')
    for ax,metric,label in zip(axes,['retention','fpr'],['Event retention (%)','False-positive rate (%)']):
        ax.plot(d.offset,d[metric]*100,'-o')
        ax.set(xlabel='Reference timing offset (samples)',ylabel=label)
    fig.savefig(a.output/'figure_6.png',dpi=180);plt.close(fig)


if __name__=='__main__':
    main()
