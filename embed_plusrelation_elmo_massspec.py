import h5py, argparse
from os.path import join, exists
import numpy as np

def outputHDF5(mhc, pep, peplen, label, masslabel, relation, elmo, filename):
    mhc = np.asarray(mhc)
    pep = np.asarray(pep)
    print('mhc size:', mhc.shape)
    print('pep size:', pep.shape)
    assert(len(mhc) == len(pep))
    assert(len(mhc) == len(label))
    comp_kwargs = {'compression': 'gzip', 'compression_opts': 4}
    with h5py.File(filename, 'w') as f:
        f.create_dataset('mhc', data=mhc, **comp_kwargs)
        f.create_dataset('pep', data=pep, **comp_kwargs)
        f.create_dataset('label', data=label, **comp_kwargs)
        f.create_dataset('masslabel', data=masslabel, **comp_kwargs)
        f.create_dataset('peplen', data=peplen, **comp_kwargs)
        f.create_dataset('relation', data=relation, **comp_kwargs)
        f.create_dataset('elmo', data=elmo, **comp_kwargs)

def embed(seq, mapper):
    return np.asarray([mapper[s] for s in seq]).transpose()

def lenpep_feature(pep):
    lenpep = len(pep) - pep.count('J')
    f1 = 1.0/(1.0 + np.exp((lenpep-args.expected_pep_len)/2.0))
    return f1, 1.0-f1

def embed_all(mhc_f, pep_f, label_f, relation_f, masslabel_f, elmo_dir, elmotag, mapper, outfile_prefix, bs=50000):
    mhc = []
    pep = []
    label = []
    masslabel = []
    peplen = []
    relation = []
    elmo = []

    cnt = 0
    bs_cnt = 0
    relation_mapper = {'=':0, '<':1, '>':2} # here relation is relative to 500 nM. '<' 500nM means higher affinity than 500nM

    elmo_cnt = 0
    elmo_batch = []
    elmo_idx = 0

    for mhc_line, pep_line, label_line, masslabel_line, relation_line in zip(mhc_f, pep_f, label_f, masslabel_f, relation_f):

        if elmo_idx >= len(elmo_batch):
            elmo_idx = 0
            elmo_cnt += 1
            assert(exists(join(elmo_dir, 'batch'+str(elmo_cnt)+'.'+elmotag+'.hdf5')))
            with h5py.File(join(elmo_dir, 'batch'+str(elmo_cnt)+'.'+elmotag+'.hdf5'), 'r') as f:
                elmo_batch = f['embed'][()]
                print(join(elmo_dir, 'batch'+str(elmo_cnt)+'.'+elmotag+'.hdf5'), elmo_batch.shape)

        mhc.append(embed(mhc_line.split()[1], mapper))
        pep.append(embed(pep_line.split()[1], mapper))
        peplen.append(lenpep_feature(pep_line.split()[1]))
        label.append(list(map(float, label_line.split()[1:])))
        masslabel.append(list(map(float, masslabel_line.split()[1:])))
        relation.append([relation_mapper[relation_line.split()[1]]])
        elmo.append(elmo_batch[elmo_idx])
        elmo_idx+=1

        if (cnt+1) % bs == 0:
            bs_cnt += 1
            outputHDF5(mhc, pep, peplen, label, masslabel, relation, elmo, outfile_prefix+str(bs_cnt))
            mhc = []
            pep = []
            label=  []
            peplen = []
            relation = []
            masslabel = []
            elmo = []
        cnt += 1

    if len(mhc) > 0:
        bs_cnt += 1
        outputHDF5(mhc, pep, peplen, label, masslabel, relation, elmo, outfile_prefix+str(bs_cnt))

def parse_args():
    parser = argparse.ArgumentParser(description="Convert sequence and target for Caffe")

    # Positional (unnamed) arguments:
    parser.add_argument("--mhcfile",  type=str, help="Sequence in FASTA/TSV format (with .fa/.fasta or .tsv extension)")
    parser.add_argument("--pepfile",  type=str,help="Label of the sequence. One number per line")
    parser.add_argument("--labelfile",  type=str, help="Output file (example: $MODEL_TOPDIR$/data/train.h5). ")
    parser.add_argument("--masslabelfile",  type=str, help="Output file (example: $MODEL_TOPDIR$/data/train.h5). ")
    parser.add_argument("--relationfile",  type=str, help="Output file (example: $MODEL_TOPDIR$/data/train.h5). ")
    parser.add_argument("--elmodir",  type=str, help="Output file (example: $MODEL_TOPDIR$/data/train.h5). ")
    parser.add_argument("--elmotag",  type=str, help="Output file (example: $MODEL_TOPDIR$/data/train.h5). ")
    parser.add_argument("--outfileprefix",  type=str, help="Output file (example: $MODEL_TOPDIR$/data/train.h5). ")
    parser.add_argument("--mapper",  type=str, help="Output file (example: $MODEL_TOPDIR$/data/train.h5). ")
    parser.add_argument("--expected_pep_len",  type=int, help="Output file (example: $MODEL_TOPDIR$/data/train.h5). ")
    return parser.parse_args()

args = parse_args()

with open(args.mapper) as f:
    mapper = dict()
    for x in f:
        line = x.split()
        mapper[line[0]] = list(map(float, x.split()[1:]))

with open(args.mhcfile) as f1, open(args.pepfile) as f2, open(args.labelfile) as f3, open(args.relationfile) as f4, open(args.masslabelfile) as f5:
    embed_all(f1, f2, f3, f4, f5, args.elmodir, args.elmotag, mapper, args.outfileprefix)
