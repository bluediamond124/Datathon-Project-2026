import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys
from wsj_style import (apply_style, finish, pct_axis, comma_axis,
                       ACCENT, ACCENT_2, GRAY, INK, INK_2)

DATA=sys.argv[1] if len(sys.argv)>1 else '/Users/bluediamondsam/PycharmProjects/Datathon/datathon_clean.csv'
df=pd.read_csv(DATA) if DATA.endswith('.csv') else pd.read_pickle(DATA)
SOURCE='Source: Summit Ridge University admissions data, entry years 2022-2026'
apply_style()
# Chart 1
admits=df[df['admitted']==1]
yld=admits.groupby('entry_year')['enrolled'].mean()
fig,ax=plt.subplots(figsize=(8,4.5))
ax.plot(yld.index,yld.values,color=ACCENT,marker='o',markersize=7)
# Direct Label first and last points
for yr in [yld.index[0],yld.index[-1]]:
    ax.annotate(f"{yld[yr]:.1%}",(yr,yld[yr]),xytext=(0,10),textcoords='offset points',
            ha='center',fontsize=11,fontweight='bold',color=INK)
ax.set_xticks(yld.index)
ax.set_ylim(0.30,0.42) # linechart: zero baseline
pct_axis(ax)
ax.spines['bottom'].set_color(GRAY)
finish(fig,ax,title='Fewer admitted students are choosing Summit Ridge',
       subtitle='Share of admitted students who enrolled by entry year',
       source=SOURCE+'. Y-axis does not start at zero.')
fig.savefig('chart1_yield_trend.png')
plt.close(fig)

# Plot 2
yr=df['entry_year'].max()
d=df[df['entry_year']==yr]
stages=pd.Series({
    'Applied':len(d),
    'Completed':d['app_completed'].sum(),
    'Admitted':d['admitted'].sum(),
    'Deposited':d['deposited'].sum(),
    'Enrolled':d['enrolled'].sum(),
})
# Highlight the biggest drop; everything else is context
drops=stages/stages.shift(1)
worst=drops.idxmin()
colors=[ACCENT if s==worst else GRAY for s in stages.index]
fig,ax=plt.subplots(figsize=(8,4.2))
y=range(len(stages))[::-1]
# top to botttom
ax.barh(list(y),stages.values,color=colors,height=0.62)
for yi,(stage,n) in zip(y,stages.items()):
    label=f"{n:,.0f}"
    if stage!='Applied':
        label+=f"   {drops[stage]:.0%} of previous"
    ax.text(n+stages.max()*0.012,yi,label,va='center',fontsize=10.5,color=INK if stage==worst else INK_2,
            fontweight='bold' if stage==worst else "normal")
ax.set_yticks(list(y), stages.index, color=INK, fontsize=11)
ax.set_xlim(0, stages.max() * 1.32)
ax.grid(False)
ax.set_xticks([])
ax.spines["bottom"].set_visible(False)
finish(fig, ax,
       title=f"Admission isn't the bottleneck. Commitment is.",
       subtitle=f"Fall {yr} applicant funnel — only about a third of admits put down a deposit",
       source=f"Source: Summit Ridge University admissions data, entry year {yr}")
fig.savefig("chart2_funnel.png")
plt.close(fig)

# Plot 3
enrolled_gpa=df.loc[df['enrolled']==1,'hs_gpa'].mean()
df['tier']=pd.cut(df['hs_gpa'],bins=[0,enrolled_gpa-0.25,enrolled_gpa+0.25,5],
                  labels=['Below Average','Average','High Achieving'])

admits=df[df['admitted']==1]
table=(admits.groupby(['residency','tier'],observed=True)['enrolled'].mean().unstack())
table=table.loc[['Resident','Contiguous','Non-Contiguous']]
print(table)
print(round(enrolled_gpa,2))
print(df['tier'].value_counts())
fig,ax=plt.subplots(figsize=(8,5))
groups=np.arange(len(table))[::-1]
offsets={'Below Average':0.27,'Average':0,'High Achieving':-0.27}
colors={'Below Average':'#d9d7d1','Average':GRAY,'High Achieving':ACCENT}
for tier in table.columns:
    ax.barh(groups+offsets[tier],table[tier],height=0.25,color=colors[tier],label=tier)
    for g,rate in zip(groups,table[tier]):
        ax.text(rate+0.006,g+offsets[tier],f"{rate:.0%}",va='center',
                fontsize=10,color=INK if tier=='High Achieving' else INK_2,
                fontweight='bold' if tier=='High Achieving' else "normal")
ax.set_yticks(groups,table.index,color=INK,fontsize=11)
pct_axis(ax,axis='x')
ax.set_xlim(0,table.max().max()*1.15)
ax.grid(axis='x')
ax.grid(axis='y',visible=False)
ax.legend(loc='lower right',ncols=3,bbox_to_anchor=(1,1.0),
          fontsize=10,handlelength=1)
finish(fig,ax,title="Where students live matters far more than their grades",
       subtitle="Share of admitted students who enrolled, by residency and high school GPA tier",
       source="Source: Summit Ridge University admissions data, 2022–2026. "
              "Tiers are ±0.25 GPA around the enrolled-student average.")
fig.savefig("chart3_tier_residency.png")




