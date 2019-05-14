from os.path import join, exists, dirname, abspath
import subprocess, h5py, numpy as np
from numpy.random import choice
from ray.tune import Trainable, grid_search, TrainingResult
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score
from collections import OrderedDict

from torch.autograd import Variable
import torch
from tqdm import tqdm
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal
from torch.utils.data import DataLoader

from torchsummary import summary
import sys
sys.path.insert(0, dirname(dirname(abspath(__file__))))
from resnet import ResNet, BasicBlock, Bottleneck

num_do_mcmc = 50

class PEPnet(nn.Module):

    def __init__(self, config):
        super(PEPnet, self).__init__()

        block_type = BasicBlock if config['pep_block'] == 'basic' else Bottleneck
        self.resnet = ResNet(
                block_type,
                config['pep_layers'],
                seq_len = config['pep_len'],
                conv_fn = config['pep_conv_fn'],
                embed_size = config['pep_embed_size'] + config['mhc_embed_size']*config['mhc_len'],
                )
        self.outlen =  config['pep_conv_fn']*(2**(len(config['pep_layers'])-1)) * block_type(1, 1).expansion
        self.config = config

    def forward(self, x):
        return self.resnet(x)


class Net(nn.Module):
    def __init__(self, config):
        super(Net, self).__init__()

        self.PEPnet = PEPnet(config)

        self.fc1_alpha = nn.Linear(self.PEPnet.outlen+2, config['dense_size'])
        self.fc1_beta = nn.Linear(self.PEPnet.outlen+2, config['dense_size'])
        self.fc2_alpha = nn.Linear(config['dense_size'], config['class_num'])
        self.fc2_beta = nn.Linear(config['dense_size'], config['class_num'])
        self.fc_mass = nn.Linear(config['mass_embed_size']+2, 1)
        self.nl = nn.Tanh()
        self.config = config

    def forward(self, mhc, pep, lenpep, elmo):
        x = self.embed(mhc, pep, lenpep)
        m = F.sigmoid(self.fc2_alpha(self.nl(self.fc1_alpha(x))))
        v = F.softplus(self.fc2_beta(self.nl(self.fc1_beta(x))))
        input2mass = torch.cat([elmo.view(len(x), -1), m, v], dim=1)
        mass_pred = F.sigmoid(self.fc_mass(input2mass))
        return m, v, mass_pred

    def embed(self, mhc, pep, lenpep):
        mhc_flat = mhc.view(-1, self.config['mhc_embed_size']*self.config['mhc_len'], 1).repeat(1, 1, self.config['pep_len'])
        pep_in = torch.cat((pep, mhc_flat), dim=1)
        pep_out = self.PEPnet(pep_in)
        return torch.cat([pep_out, lenpep], dim=1)
        #x = self.do1(torch.cat([pep_out, lenpep], dim=1))
        #return self.do2(self.nl(self.fc1(x)))

def get_setup():
    return {
               "run": "my_class",
               "repeat": 160,
               "trial_resources": {"cpu": 0, "gpu": 1},
               "config": {
                   "pep_layers": [5],
                   "pep_block": "basic",
                   "pep_conv_fn": lambda spec: choice([512, 1024, 2056]),
                   "adam_lr": lambda spec: choice([1e-03, 1e-4]),
                   "adam_beta1": 0.9,
                   "adam_beta2": lambda spec: choice([0.99, 0.999]),
                   "dense_size": lambda spec: choice([16, 32]),
                   "train_epoch_scale": 1,
                   "class_num": 1,
                   "batch_size": 128,
               },
            }


def cnt_param(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


class MyTrainableClass(Trainable):

    def _setup(self):

        self.device = torch.device("cuda")

        self.mode = 'tune' if 'mode' not in self.config else self.config['mode']

        if self.mode in ['train', 'tune']:
            print('loading trainval data')
            trainset = HDF5_dataset(self.config['trainset_prefix'])
            validset = HDF5_dataset(self.config['validset_prefix'])
            print('train prefix', self.config['trainset_prefix'])
            print('valid prefix', self.config['validset_prefix'])
            self.trainset = DataLoader(trainset, batch_size=self.config['batch_size'], shuffle=True, num_workers=0)
            self.validset = DataLoader(validset, batch_size=self.config['batch_size'], shuffle=False, num_workers=0)
            self.config['pep_embed_size'], self.config['pep_len'] = trainset.pep[0].shape[1:3]
            self.config['mhc_embed_size'], self.config['mhc_len'] = trainset.mhc[0].shape[1:3]
            print(trainset.pep[0].shape)

        if self.mode in ['eval', 'pred']:
            print('loading test data')
            testset = HDF5_dataset(self.config['testset_prefix'])
            self.testset = DataLoader(testset, batch_size=self.config['batch_size'], shuffle=False, num_workers=0)
            self.config['pep_embed_size'], self.config['pep_len'] = testset.pep[0].shape[1:3]
            self.config['mhc_embed_size'], self.config['mhc_len'] = testset.mhc[0].shape[1:3]

        self.net = Net(self.config).to(device=self.device)
        self.bin_loss = torch.nn.BCELoss()
        summary(self.net.PEPnet, (self.config['pep_embed_size']+self.config['mhc_len']*self.config['mhc_embed_size'], self.config['pep_len']))
        print('num of params:', cnt_param(self.net))
        self.optimizer = optim.Adam(self.net.parameters(), lr=self.config['adam_lr'], betas=(self.config['adam_beta1'], self.config['adam_beta2']))

    def _train(self):
        train_loss = 0.0
        train_spearmanr = []
        train_auc = []
        train_mass_auc = []
        self.net.train()

        for i, data in enumerate(tqdm(self.trainset), 0):
            # get the inputs
            mhc_inputs, pep_inputs, lenpep_inputs, elmos, labels, relation, masslabels = data[0].to(self.device), data[1].to(self.device), data[2].to(self.device), data[3].to(self.device), data[4].to(self.device), data[5].to(self.device), data[6].to(self.device)

            # zero the parameter gradients
            self.optimizer.zero_grad()

            # forward + backward + optimize
            m, v, mass_pred = self.net(mhc_inputs, pep_inputs, lenpep_inputs, elmos) # both mbsize x label_dim

            mass_pick = masslabels != -1
            real_relation = (m < labels) + 1  # 2 if mean < label, 1 otherwise
            affinity_pick = (labels!=-1) & (real_relation != relation)  # in relation, 2 if real (normalized) affinity  < label, 1 otherwise

            m, v, labels = m[affinity_pick], v[affinity_pick], labels[affinity_pick]
            if len(m)>0:
                normal_distr = Normal(m, v)
                aff_loss = -torch.mean(normal_distr.log_prob(labels))
            else:
                aff_loss = 0

            mass_pred, masslabels = mass_pred[mass_pick], masslabels[mass_pick]
            mass_loss = self.bin_loss(mass_pred, masslabels)

            loss = aff_loss + mass_loss

            label_np = labels.cpu().detach().numpy()
            label_bin = (label_np > 0.426).astype(float)
            o_mean_np = m.cpu().detach().numpy()
            try:
                t_spearmanr = spearmanr(label_np, o_mean_np)[0]
                train_spearmanr.append(t_spearmanr)
            except ValueError as err:
                print ('fail to calculate train spearmanr:', err)

            try:
                t_auc = roc_auc_score(label_bin, o_mean_np)
                train_auc.append(t_auc)
            except ValueError as err:
                print ('fail to calculate train auc:', err)

            masslabel_np = masslabels.cpu().detach().numpy()
            masspred = mass_pred.cpu().detach().numpy()
            #try:
            #    train_mass_auc.append(roc_auc_score(masslabel_np, masspred))
            #except ValueError as err:
            #    print ('fail to calculate train auc:', err)

            loss.backward()
            self.optimizer.step()
            train_loss += loss.item()
            if i+1 == int(len(self.trainset) * self.config['train_epoch_scale']):
                break

        train_loss_mean = train_loss / float(i+1)
        train_auc_mean = np.mean(train_auc) if len(train_auc) >0 else -1
        train_mass_auc_mean = np.mean(train_mass_auc) if len(train_mass_auc) >0 else -1
        train_spearmanr_mean = np.mean(train_spearmanr) if len(train_spearmanr) >0 else -1

        # predict on valid set
        self.net.eval()
        valid_loss = 0.0
        valid_spearmanr = []
        valid_auc = []
        valid_mass_auc = []

        for i, data in enumerate(self.validset):
            mhc_inputs, pep_inputs, lenpep_inputs, elmos, labels, relation, masslabels = data[0].to(self.device), data[1].to(self.device), data[2].to(self.device), data[3].to(self.device), data[4].to(self.device), data[5].to(self.device), data[6].to(self.device)

            m, v, mass_pred = self.net(mhc_inputs, pep_inputs, lenpep_inputs, elmos) # both mbsize x label_dim

            mass_pick = masslabels != -1
            real_relation = (m < labels) + 1  # 2 if mean < label, 1 otherwise
            affinity_pick = (labels!=-1) & (real_relation != relation)  # in relation, 2 if real (normalized) affinity  < label, 1 otherwise

            m, v, labels = m[affinity_pick], v[affinity_pick], labels[affinity_pick]
            if len(m)>0:
                normal_distr = Normal(m, v)
                aff_loss = -torch.mean(normal_distr.log_prob(labels))
            else:
                aff_loss = 0

            mass_pred, masslabels = mass_pred[mass_pick], masslabels[mass_pick]
            mass_loss = self.bin_loss(mass_pred, masslabels)

            loss = aff_loss + mass_loss

            label_np = labels.cpu().detach().numpy()
            label_bin = (label_np > 0.426).astype(float)
            o_mean_np = m.cpu().detach().numpy()
            try:
                t_spearmanr = spearmanr(label_np, o_mean_np)[0]
                valid_spearmanr.append(t_spearmanr)
            except ValueError as err:
                print ('fail to calculate valid spearmanr:', err)
            try:
                t_auc = roc_auc_score(label_bin, o_mean_np)
                valid_auc.append(t_auc)
            except ValueError as err:
                print ('fail to calculate valid auc:', err)

            masslabel_np = masslabels.cpu().detach().numpy()
            masspred = mass_pred.cpu().detach().numpy()
            #try:
            #    valid_mass_auc.append(roc_auc_score(masslabel_np, masspred))
            #except ValueError as err:
            #    print ('fail to calculate train auc:', err)


            valid_loss +=  loss.item()

        valid_loss_mean = valid_loss / float(i+1)
        valid_auc_mean = np.mean(valid_auc) if len(valid_auc)>0 else -1
        valid_mass_auc_mean = np.mean(valid_mass_auc) if len(valid_mass_auc)>0 else -1
        valid_spearmanr_mean = np.mean(valid_spearmanr) if len(valid_spearmanr)>0 else -1

        if self.mode == 'train':
            return OrderedDict([('loss', train_loss_mean), ('auc', train_auc_mean), ('spearmanr', train_spearmanr_mean), ('mass_auc', train_mass_auc_mean)]), \
                    OrderedDict([('loss', valid_loss_mean), ('auc', valid_auc_mean), ('spearmanr', valid_spearmanr_mean), ('mass_auc', valid_mass_auc_mean)])
        else:
            return TrainingResult(mean_loss=valid_loss_mean, timesteps_this_iter=1)

    def _pred(self, dataset):
        self.net.eval()

        for i, data in enumerate(dataset, 0):
            mhc_inputs, pep_inputs, lenpep_inputs, elmos = data[0].to(self.device), data[1].to(self.device), data[2].to(self.device), data[3].to(self.device)

            m, v, mass_pred = self.net(mhc_inputs, pep_inputs, lenpep_inputs, elmos) # both mbsize x label_dim
            mean_y = m.cpu().detach().numpy()
            var_y = v.cpu().detach().numpy()
            mass_pred_y = mass_pred.cpu().detach().numpy()
            t_out = np.hstack((mean_y, var_y, mass_pred_y))

            out = t_out if i == 0 else np.vstack((out, t_out))

        return out


    def _embed(self, dataset):
        self.net.eval()
        for i, data in enumerate(dataset, 0):
            mhc_inputs, pep_inputs, lenpep_inputs = data[0].to(self.device), data[1].to(self.device), data[2].to(self.device)

            t_out = self.net.embed(mhc_inputs, pep_inputs, lenpep_inputs) # both mbsize x label_dim

            out_mean = t_out if i == 0 else np.vstack((out_mean, t_out))

        return out_mean

    def _save(self, checkpoint_dir):
        path = join(checkpoint_dir, 'checkpoint.pt')
        torch.save(self.net, path)

        optim_path = join(checkpoint_dir, 'optim.pt')
        torch.save({'optimizer': self.optimizer.state_dict()}, optim_path)

        return path

    def _restore(self, checkpoint_path):
        self.net = torch.load(checkpoint_path)
        self.optimizer = optim.Adam(self.net.parameters(), lr=self.config['adam_lr'], betas=(self.config['adam_beta1'], self.config['adam_beta2']))
        self.optimizer.load_state_dict(torch.load(join(dirname(checkpoint_path), 'optim.pt'))['optimizer'])


class HDF5_dataset(torch.utils.data.Dataset):
    def __init__(self, prefix):
        self.prefix = prefix
        cnt = 1
        self.label = []
        self.masslabel = []
        self.mhc = []
        self.pep = []
        self.peplen = []
        self.relation = []
        self.elmo = []
        self.num_sample = 0
        while exists(prefix+str(cnt)):
            print('batch', cnt)

            with h5py.File(prefix+str(cnt),'r') as dataall:
                self.label.append(dataall['label'][()].astype(np.float32))
                self.masslabel.append(dataall['masslabel'][()].astype(np.float32))
                self.mhc.append(dataall['mhc'][()].astype(np.float32))
                self.pep.append(dataall['pep'][()].astype(np.float32))
                self.peplen.append(dataall['peplen'][()].astype(np.float32))
                self.relation.append(dataall['relation'][()].astype(np.uint8))
                self.elmo.append(dataall['elmo'][()].astype(np.float32))
            #if cnt == 1:
            #    label = np.asarray(dataall['label'], dtype=np.float32)
            #    masslabel = np.asarray(dataall['masslabel'], dtype=np.float32)
            #    mhc = np.asarray(dataall['mhc'], dtype=np.float32)
            ##    pep = np.asarray(dataall['pep'], dtype=np.float32)
            #    peplen = np.asarray(dataall['peplen'], dtype=np.float32)
            #    relation = np.asarray(dataall['relation'], dtype=np.uint8)
            #    elmo = np.asarray(dataall['elmo'], dtype=np.float32)
            #else:
            #    label = np.vstack((label, np.asarray(dataall['label'], dtype=np.float32)))
            #    masslabel = np.vstack((masslabel, np.asarray(dataall['masslabel'], dtype=np.float32)))
            #    mhc = np.vstack((mhc, np.asarray(dataall['mhc'], dtype=np.float32)))
            #    pep = np.vstack((pep, np.asarray(dataall['pep'], dtype=np.float32)))
            #    peplen = np.vstack((peplen, np.asarray(dataall['peplen'], dtype=np.float32)))
            #    relation = np.vstack((relation, np.asarray(dataall['relation'], dtype=np.uint8)))
            #    elmo = np.vstack((elmo, np.asarray(dataall['elmo'], dtype=np.float32)))
            cnt += 1
            self.num_sample += len(self.label[-1])

        #print('start conversion to numpy')
        #self.mhc = np.vstack(self.mhc).astype(np.float16)
        #self.mhc = np.asarray(self.mhc, dtype=np.float32).reshape(num_sample, -1)
        #self.pep = np.vstack(self.pep).astype(np.float16)
        #self.elmo = np.vstack(self.elmo).astype(np.float16)
        #self.peplen = np.vstack(self.peplen).astype(np.float16)
        #self.masslabel = np.vstack(self.masslabel).astype(np.float16)
        #self.relation =  np.vstack(self.relation).astype(np.uint8)
        #self.label = np.vstack(self.label).astype(np.float16)
        #self.data = self.mhc
        self.data_len = len(self.mhc)

    def __getitem__(self, index):
        bs = 50000
        cnt = index // bs
        i = index % bs
        return self.mhc[cnt][i], self.pep[cnt][i], self.peplen[cnt][i],self.elmo[cnt][i], self.label[cnt][i], self.relation[cnt][i], self.masslabel[cnt][i]

    def __len__(self):
        return self.num_sample
        #return self.data_len
