# load important libraries
import pandas as pd
import numpy as np

# definition of features/selections
met_feats = ["phi","pt","px","py"]
lep_feats = ["pt","eta","phi","en","px","py","pz"]
jet_feats = ["pt","eta","phi","en","px","py","pz","btag"]

njets = 7
nleps = 1

selection = "nleps == 1"


# produce flat numpy arrays
def flat_numpy(filename):

    # read in dataframes
    df = pd.read_hdf(filename)

    # map btag to useful values
    for i in range(10):
        df["jets_btag_" + str(i)] = (df['jets_btag_{0}'.format(i)]>1) 
        df["jets_btag_" + str(i)] = df["jets_btag_" + str(i)].astype(float)

    # apply any selection to dataframe
    if selection is not None:
        df = df.query(selection)

    flats = []

    # jets
    if jet_feats is not None:
        print('formatting jets...')
        onejet = list(range(njets))
        #for ijet in onejet:
        #    make_p4(df,'jets',ijet)
        njf = len(jet_feats)
        jet_feat_cols = ["jets_%s_%d" % (feat,jet) for jet in onejet for feat in jet_feats  ]
        jetsa = df[jet_feat_cols].values
        flats.append(jetsa)
        jetsa = jetsa.reshape(-1,njets,njf)
        #np.save(args.outdir+"/jets",jetsa)
        print('done',jetsa.shape)

    # leptons
    if lep_feats is not None and nleps>0:
        print('formatting leptons...')
        nlf = len(lep_feats)
        #for ilep in range(nleps):
        #    make_p4(df,'leps',ilep)
        lepsa = df[ ["leps_%s_%d" % (feat,lep) for lep in range(nleps) for feat in lep_feats ]  ].values
        flats.append(lepsa)
        lepsa = lepsa.reshape(-1,nleps,nlf)
        #np.save(args.outdir+"/leps",lepsa)
        print('done',lepsa.shape)

    # met
    if met_feats is not None:
        print('formatting met...')
        df["met_px"] = df["met_"+met_feats[1]]*np.cos(df["met_"+met_feats[0]])
        df["met_py"] = df["met_"+met_feats[1]]*np.sin(df["met_"+met_feats[0]])
        meta = df[ ["met_%s" % feat for feat in met_feats  ]  ].values
        flats.append(meta)
        #np.save(args.outdir+"/met",meta)
        print('done',meta.shape)

    print('making flat features...')
    flata = np.hstack(flats)

    return flata


nevents = 100000

sig = flat_numpy("dataSig.h5")
bkg = flat_numpy("dataBkg.h5")
comb = np.concatenate((sig[:nevents,:],bkg[:nevents,:]), axis=0)
np.save("inputs",comb)

