#  source /p/project/exaww/chatterjee1/vissl_venv_2024/activate.sh
import numpy as np
import os, glob
from sklearn.manifold import TSNE
#from openTSNE import TSNE

import openTSNE
import torch


common_path = '/p/scratch/exaww/chatterjee1/nn_obs/all_nc/features_icon/' 
data = torch.load(os.path.join(common_path, "trainfeat_new_multiyear.pth")).cpu().detach().numpy()


embedding_pca_cosine = openTSNE.TSNE(
    perplexity=30,
    initialization="pca",
    metric="cosine",
    n_jobs=-1,
    random_state=3,
).fit(data)

np.save(common_path+ "tsne_icon_my_pcacosine.npy", embedding_pca_cosine)
print('step 1 complete')

################

embedding_annealing = openTSNE.TSNE(
    perplexity=500, metric="cosine", initialization="pca", n_jobs=-1, random_state=3
).fit(data)

embedding_annealing.affinities.set_perplexities([50])

embedding_annealing = embedding_annealing.optimize(250)

np.save(common_path+"tsne_icon_my_500annealing50.npy", embedding_annealing)
print('step 2 complete')

##########


affinities_multiscale_mixture = openTSNE.affinity.Multiscale(
    data,
    perplexities=[50, 500],
    metric="cosine",
    n_jobs=-1,
    random_state=3,
)



init = openTSNE.initialization.pca(data, random_state=42)

embedding_multiscale = openTSNE.TSNE(n_jobs=-1).fit(
    affinities=affinities_multiscale_mixture,
    initialization=init,
)

np.save(common_path+"tsne_icon_my_500multiscale50.npy", embedding_multiscale)
print('step 3 complete')
