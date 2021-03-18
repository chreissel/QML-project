import numpy as np
from sklearn import metrics
import matplotlib.pyplot as plt

# Arguments for functions
import argparse
parser = argparse.ArgumentParser(description="Preselection & Train/Test Split of Samples")
parser.add_argument('--path', help = "path from where to read files")
parser.add_argument("--latent", action="store_true")
args = parser.parse_args()
if not args.path.endswith('/'):
    path = args.path + "/"
else:
    path = args.path

# File to get AUC
if args.latent:
    f = open("Latent_AUC.txt", "w+")
else:
    f = open("Input_AUC.txt","w+")
f.write("Discrimination (AUC) for all input variables individually\n")

# Function to get ROC/AUC
def ROC(pred, true):
    
    fpr, tpr, thresholds = metrics.roc_curve(true, pred)
    auc = metrics.roc_auc_score(true, pred)
    return fpr, tpr, auc

# Plotting starts here
true = np.load(path+"y_train.npy")
pred = np.load(path+"X_train.npy")

# Labels
labels = []
if not args.latent:
    met_feats = ["phi","pt","px","py"]
    lep_feats = ["pt","eta","phi","en","px","py","pz"]
    jet_feats = ["pt","eta","phi","en","px","py","pz","btag"]

    for i in range(7):
        for feat in jet_feats:
            labels.append("jet_" + str(i) + "_" + feat)
    labels += ["lep_" + feat for feat in lep_feats]
    labels += ["met_" + feat for feat in met_feats]
else:
    labels += ["latent_variable_"+str(i) for i in range(pred.shape[1])]

import pdb
pdb.set_trace()

for i in range(pred.shape[1]):
    print("Make plot variable: ", labels[i])
    fig = plt.figure(figsize=(6,6))
    pred_var = pred[:,i]
    fpr, tpr, auc = ROC(pred_var, true)
    plt.title("Variable: "+ labels[i])
    plt.plot(fpr,tpr, label = " AUC: %.2f" %auc)
    plt.plot([0, 1], [0, 1], ls="--", c=".3")

    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.legend()
    fig.savefig("var_"+str(i)+".png")
    plt.close()
    f.write(labels[i]+" "+str(auc)+"\n")

f.close()
