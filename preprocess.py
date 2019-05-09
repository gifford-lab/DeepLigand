from utils import *
import argparse, shutil
from os import makedirs
from os.path import exists, realpath, join, dirname

def parse_args():
    parser = argparse.ArgumentParser(description="Keras + Hyperband for genomics")
    parser.add_argument("-f", "--file", required=True)
    parser.add_argument("-o", "--outdir", required=True)
    return parser.parse_args()

args = parse_args()

create_dir(args.outdir)

pwd = dirname(realpath(__file__))
args.file = realpath(args.file)
args.outdir = realpath(args.outdir)

# Load pseudo-sequences
pseudo_seq_file = join(pwd, 'data/MHC_pseudo.dat')
pseudo_seq_dict = dict()
with open(pseudo_seq_file) as f:
    for x in f:
        line = x.split()
        pseudo_seq_dict[line[0]] = line[1]

# Map MHC names
print('mhc mapping')
mhc_mapper(args.file, args.outdir, pseudo_seq_dict)

# Pad peptides to 40 AA
print('peptide padding')
padseq(join(args.outdir, 'test.pep'), '.pep', pad2len = {'.pep':40})

# Tokenize
print('tokenizing')
tokenize(join(args.outdir, 'test.pep'), join(args.outdir, 'test.pep.token'))

# Peptide embedding
print('peptide embedding')
system(' '.join(['python {}/elmo_embed.py -d {} -e {} -t {} --trial_num -1'.format(pwd, args.outdir, join(pwd, 'data'), 'test')]))

# Embed
print('data embedding')
embed(args.outdir, args.outdir)
