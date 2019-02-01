import os, sys
import tensorflow as tf
from os.path import join, dirname, basename, exists
from tqdm import tqdm
import h5py, argparse
from bilm import Batcher, BidirectionalLanguageModel, weight_layers

def parse_args():
    parser = argparse.ArgumentParser(description="Keras + Hyperband for genomics")
    parser.add_argument("-e", "--exptdir", required=True, help="we will use ${exptdir}/alltrain.epitope.elmo as the model directory")
    parser.add_argument("-d", "--datadir", required=True, help="the data directory, under which we look for token folders for each data types (--dtypes) specified")
    parser.add_argument("-t", "--dtypes", default='train:valid:test', help="data types to embed for")
    parser.add_argument("--trial_num", default=1, type=int, help="set it to 0 when there is no trial splits")
    parser.add_argument("--max_len", default=42, type=int, help="this should be set to 2+max(word_len)")
    return parser.parse_args()


args = parse_args()

dtypes = args.dtypes.split(':')
trial_num = max(1, args.trial_num)

###

# We will use "${args.exptdir}/alltrain.epitope.elmo" as the model directory
model_dir = join(args.exptdir, 'alltrain.epitope.elmo', 'best_model')
vocab_file = join(args.exptdir, 'alltrain.epitope.vocab')
options_file = join(model_dir, 'pred.options.json')
weight_file = join(model_dir, 'weights.h5')

# Create a Batcher to map text to character ids.
batcher = Batcher(vocab_file, 50)

# Input placeholders to the biLM.
context_character_ids = tf.placeholder('int32', shape=(None, None, 50))
bilm = BidirectionalLanguageModel(options_file, weight_file)

context_embeddings_op = bilm(context_character_ids)

elmo_context_input = weight_layers('input', context_embeddings_op, l2_coef=0.0)
elmo_context_output = weight_layers(
    'output', context_embeddings_op, l2_coef=0.0
)

with tf.Session() as sess:
    # It is necessary to initialize variables once before running inference.

    sess.run(tf.global_variables_initializer())

    for trial in range(trial_num):
        t_topdir = join(args.datadir, 'trial'+str(trial+1)) if args.trial_num>=1 else args.datadir

        for dtype in dtypes:
                prefix = join(t_topdir, dtype+'.pep.token/batch')
                print('prefix', prefix)

                allcnt = 1
                while exists(prefix+str(allcnt)):
                    allcnt += 1

                print('num of batches', allcnt)

                for cnt in tqdm(range(1, allcnt)):

                    dataset_file = prefix+str(cnt)
                    embedding_file = dataset_file+'.elmo_embeddingds_{}.hdf5'.format(basename(dirname(model_dir)))

                    with open(dataset_file) as f:
                        tokenized_context = [x.strip().split() for x in f]

                    # Create batches of data.
                    context_ids = batcher.batch_sentences(tokenized_context, max_length=args.max_len)

                    # Compute ELMo representations (here for the input only, for simplicity).

                    elmo_context_input_ = sess.run(
                        elmo_context_input['weighted_op'],
                        feed_dict={context_character_ids: context_ids}
                    )

                    print(elmo_context_input_[0].shape)
                    with h5py.File(embedding_file, 'w') as f:
                        f.create_dataset('embed', data=elmo_context_input_)
