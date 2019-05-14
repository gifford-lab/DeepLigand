from os.path import join, exists, dirname, realpath
from os import makedirs, system
import shutil

def tokenize(infile, outdir, batchsize=128):
    create_dir(outdir)

    cnt = 1
    fout = open(join(outdir, 'batch'+str(cnt)), 'w')
    with open(infile) as f:
        for idx, x in enumerate(f):
            line = list(x.split()[1])
            fout.write('\t'.join(line)+'\n')
            if (idx+1) % batchsize == 0:
                fout.close()
                cnt += 1
                fout = open(join(outdir, 'batch'+str(cnt)), 'w')

def padseq(file2pad, seq_type, pad2len = {'.mhc':34, '.pep':37}, padding = 'J', padded_suffix='.padded'):
    with open(file2pad) as fin, open(file2pad+padded_suffix, 'w') as fout:
        for idx, x in enumerate(fin):
            line = x.split()
            seq = list(line[1])
            fout.write(line[0] + '\t' + ''.join(seq + [padding]*(pad2len[seq_type] - len(seq))) +'\n')

mhc_dict = {
    'Mamu-B5201': 'Mamu-B52',
    'Mamu-A0201': 'Mamu-A02',
    'Mamu-B0101': 'Mamu-B:00101',
    'Mamu-B0301': 'Mamu-B:00301',
    'Mamu-B0401': 'Mamu-B:00401',
    'Mamu-A070103': 'Mamu-A7:0103',
    'Mamu-B1701': 'Mamu-B17',
    'Mamu-A1101': 'Mamu-A11',
    'Mamu-A020102':'Mamu-A2:0102',
    'Mamu-B0801': 'Mamu-B008:01',
    'Mamu-A0101': 'Mamu-A01',
    'Mamu-B6502': 'Mamu-B:06502',
    'BoLA-HD:00601': 'BoLA-HD6'
}

def mhc_rename(mhc):
    mhc = mhc.replace(':', '')
    if mhc[:4] == 'BoLA':
        mhc = mhc.replace('*', ':0')
    else:
        mhc = mhc.replace('*', '')

    if mhc in mhc_dict:
        mhc = mhc_dict[mhc]

    return mhc


def mhc_mapper(rawfile, outdir, pseudo_seq_dict, dt='test'):
    alleles_not_recognized = set()
    with open(rawfile) as f, \
        open(join(outdir, dt+'.mhc'), 'w') as f1, \
        open(join(outdir, dt+'.pep'), 'w') as f2, \
        open(join(outdir, dt+'.label'), 'w') as f3, \
        open(join(outdir, dt+'.relation'), 'w') as f4, \
        open(join(outdir, dt+'.masslabel'), 'w') as f5, \
        open(join(outdir, dt+'.mhcname'), 'w') as f6:

        for idx, x in enumerate(f):
            line = x.strip().split(',')
            prefix = '>mhc_seq' + str(idx) + '\t'
            #mhc = mhc_rename(line[0])
            mhc = line[0]
            if mhc not in pseudo_seq_dict:
                alleles_not_recognized.add(line[0])
                continue

            f1.write(prefix + pseudo_seq_dict[mhc]+'\n')
            f2.write(prefix + line[1]+'\n')
            f3.write(prefix + '0'+'\n')
            f4.write(prefix + '='+'\n')
            f5.write(prefix + '0'+'\n')
            f6.write(prefix + mhc + '\n')

    if len(alleles_not_recognized)>0:
        raise Exception('The following alleles are not recognized:', alleles_not_recognized)

def embed(datadir, outdir):
    pwd = dirname(realpath(__file__))
    expected_pep_len = 9
    mapper = join(pwd, 'data/onehot_first20BLOSUM50')
    elmotag = 'elmo_embeddingds_alltrain.epitope.elmo'

    template = 'python {}/embed_plusrelation_elmo_massspec.py --mhcfile {} --pepfile {} --labelfile {} \
            --relationfile {} --masslabelfile {} --elmodir {} --elmotag {}\
        --mapper {} --outfileprefix  {} --expected_pep_len {}'


    dt = 'test'
    cmd =  template.format(
        pwd,
        join(outdir, dt+'.mhc'),
        join(outdir, dt+'.pep.padded'),
        join(outdir, dt+'.label'),
        join(outdir, dt+'.relation'),
        join(outdir, dt+'.masslabel'),
        join(outdir, dt+'.pep.token'),
        elmotag,
        mapper,
        join(outdir, dt+'.h5.batch'),
    expected_pep_len)
    print('running', cmd)
    system(cmd)


def create_dir(mydir):
    if exists(mydir):
        print('Output directory', mydir, 'exists! Overwrite? (yes/no)')
        if input().lower() == 'yes':
            shutil.rmtree(mydir)
        else:
            print('Quit !')
            sys.exit(1)

    makedirs(mydir)
