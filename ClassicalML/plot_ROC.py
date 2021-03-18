import numpy as np
from sklearn import metrics
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 22})

# Arguments for functions
import argparse
parser = argparse.ArgumentParser(description="Preselection & Train/Test Split of Samples")
parser.add_argument('--classifiers', nargs="+", help = "classifier to be compared")
parser.add_argument('--path', default="/pnfs/psi.ch/cms/trivcat/store/user/creissel/QML-project/",help = "path from where to read files")
parser.add_argument('--path_latent', default="/pnfs/psi.ch/cms/trivcat/store/user/creissel/QML-project/latent/" ,help = "path from where to read files")
parser.add_argument('--directory', default="/work/creissel/QML-project/ClassicalML/final_results/", help="")
args = parser.parse_args()
if not args.path.endswith('/'):
    path = args.path + "/"
else:
    path = args.path
if not args.path_latent.endswith('/'):
    path_latent = args.path_latent + "/"
else:
    path_latent = args.path_latent

AUC = {
"BDT_full" : (0.6907761981513929,0.0011228653269618272),
"BDT_latent": (0.651642086998313,0.0018538581649366558),
"DNN_full": (0.7036843347697646,0.0011043722547259109),
"DNN_latent": (0.6234998347219496,0.002048972380975242)
}

# Function to get ROC/AUC
def ROC(pred, true):
    
    fpr, tpr, thresholds = metrics.roc_curve(true, pred)
    auc = metrics.roc_auc_score(true, pred)
    return fpr, tpr, auc

# Plotting starts here
true = np.load(path+"y_test.npy")
train = np.load(path+"y_train.npy")
true_latent = np.load(path_latent+"y_test.npy")
train_latent = np.load(path_latent+"y_train.npy")

fig = plt.figure(figsize=(10,10))

for clf in args.classifiers:
    pred = np.load(args.directory+"pred_"+clf+".npy")
    pred_latent = np.load(args.directory+"pred_"+clf+"_latent.npy")
    fpr, tpr, auc = ROC(pred, true)
    fpr_latent, tpr_latent, auc_latent = ROC(pred_latent, true_latent) 
    plt.title("N(train): %i, N(test) %i" % (train.shape[0], true.shape[0]) , loc="left")
    p = plt.plot(fpr,tpr, lw=2, label = clf + u", AUC = %.3f \u00B1 %.3f" %(AUC[clf+"_full"][0], AUC[clf+"_full"][1]))
    plt.plot(fpr_latent,tpr_latent, '--', lw=2, color=p[0].get_color(), label = clf + u"(latent), AUC = %.3f \u00B1 %.3f" %(AUC[clf+"_latent"][0], AUC[clf+"_latent"][1]))
    plt.plot([0, 1], [0, 1], ls=":", c=".3")

plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.xlabel("Background Efficiency (FPR)")
plt.ylabel("Signal Efficiency (TPR)")
plt.legend()
plt.tight_layout()
fig.savefig("Classical_HEPApproaches.pdf")
plt.show()
