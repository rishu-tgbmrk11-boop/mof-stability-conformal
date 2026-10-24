"""Review H1: do the SI's test-set counts follow from train_test_split(CSV rows, test_size=0.2, random_state=0)? (SI counts as recorded in notes/stability_inventory.md.)"""
import pandas as pd, numpy as np
from sklearn.model_selection import train_test_split
R="data/raw/optimof/data/"
sv=pd.read_csv(R+"solvent/solvent_all_data.csv"); th=pd.read_csv(R+"thermal/thermal_all_data.csv"); wt=pd.read_csv(R+"water_and_haz/data.csv")
_,te=train_test_split(sv,test_size=0.2,random_state=0); print("solvent test n",len(te),"stable",int((te.flag>0).sum()),"unstable",int((te.flag<=0).sum()),"| SI: 275 stable of 436")
_,te=train_test_split(th,test_size=0.2,random_state=0); print("thermal test n",len(te),"stable(>=350)",int((te['T']>=350).sum()))
_,te=train_test_split(wt,test_size=0.2,random_state=0); print("water (split of data.csv rows) n",len(te),te.water_label.value_counts().sort_index().to_dict(),"| SI 17/68/112/22")
c=np.array([int((train_test_split(sv,test_size=0.2,random_state=rs)[1].flag>0).sum()) for rs in range(200)])
print("solvent stable count in a random 436-MOF test set over 200 seeds: mean %.1f sd %.1f; P(count>=275) = %.3f"%(c.mean(),c.std(),(c>=275).mean()))

# internal consistency of the SI thermal numbers: R2 = 1 - RMSE^2 / var(T_test)  =>  implied sd of the SI test set vs the sd of the CSV seed-0 test set
print("\nSI thermal: R2 0.41, RMSE 67 -> implied sd(T_test) = %.1f" % (67/np.sqrt(1-0.41)))
th_te=train_test_split(th,test_size=0.2,random_state=0)[1]["T"]
print("CSV seed-0 thermal test: n %d, sd(T) = %.1f, mean %.1f"%(len(th_te),th_te.std(),th_te.mean()))
print("SI solvent confusion matrix: accuracy %.3f precision %.3f recall %.3f F1 %.3f; stable in test = TP+FN = %d"%((110+228)/436,228/(228+51),228/(228+47),2*228/(2*228+51+47),228+47))
