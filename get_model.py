from os import system, makedirs
from os.path import join, dirname, basename

elmo_dir = '../../data/MHCflurry/curated_training_data.with_mass_spec.addAbelinNeg.w_pseudoseq.normalized.sorted.groupby8mer_allallele.alldata/alltrain.epitope.elmo'
outdir = 'data/alltrain.epitope.elmo/best_model'
makedirs(outdir)

system(' '.join(['cp -r',
    join(elmo_dir, 'best_model', 'weight*'),
    outdir
    ]))

system(' '.join(['cp -r',
    join(elmo_dir, 'best_model', 'pred*'),
    outdir
    ]))


system(' '.join(['cp -r',
    join(dirname(elmo_dir), 'alltrain.epitope.vocab'),
    'data',
    ]))

model_name = 'mhccat2pep_pepres_relation_massspec_elmo_novar_v3_normal_noeps_bs1024_init1'
model_realname = 'mhccat2pep_pepres_relation_massspec_elmo_novar_v3_normal_noeps'
model_dir = '../../data/MHCflurry/dataset/curated_training_data.with_mass_spec.addAbelinNeg.w_pseudoseq.normalized.sorted.groupby8mer_allallele.CV5_pluselmo/splitall/trial1'

makedirs(join('data', model_name))

system(' '.join(['cp -r',
    join(model_dir, model_name, 'best*'),
    join('data', model_name)
    ]))

system(' '.join([
    'cp ',
    '../models/'+model_realname+'/model_py36.py',
    'model.py'
    ]))
system(' '.join([
    'cp -r',
    '../models/resnet.py',
    'resnet.py'
    ]))
