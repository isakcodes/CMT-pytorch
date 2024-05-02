import os
import glob
import torch
import random
import numpy as np
from collections import defaultdict
import pickle
from scipy.sparse import csc_matrix

from utils import logger
from dataset import collate_fn

class BaseInferencer:
    def __init__(self, asset_path, model, device, config):
        self.asset_path = asset_path
        self.model = model
        self.device = device
        self.config = config

    def load_model(self, restore_epoch, load_rhythm=False):
        model = self.model.module if isinstance(self.model, torch.nn.DataParallel) else self.model

        restore_ckpt = os.path.join(self.asset_path, 'model', 'checkpoint_%d.pth.tar' % restore_epoch)
        if not (os.path.isfile(restore_ckpt) or load_rhythm):
            logger.info("no checkpoint with %d epoch" % restore_epoch)
        else:
            if os.path.isfile(restore_ckpt):
                checkpoint = torch.load(restore_ckpt, map_location=self.device)
            else:
                rhythm_asset_path = os.path.join('/'.join(self.asset_path.split('/')[:-1]),
                                                 'idx%03d' % self.config['restore_rhythm']['idx'])
                rhythm_ckpt = os.path.join(rhythm_asset_path, 'model',
                                           'checkpoint_%d.pth.tar' % self.config['restore_rhythm']['epoch'])
                checkpoint = torch.load(rhythm_ckpt, map_location=self.device)
            if load_rhythm:
                model_dict = model.state_dict()
                rhythm_state_dict = {k: v for k, v in checkpoint['model'].items() if 'rhythm' in k}
                model_dict.update(rhythm_state_dict)
                model.load_state_dict(model_dict)
                logger.info("restore rhythm model")
            else:
                model.load_state_dict(checkpoint['model'])
                logger.info("restore model with %d epoch" % restore_epoch)

    def infer(self, **kwargs):
        raise NotImplementedError()

class CMTInferencer(BaseInferencer):
    def __init__(self, asset_path, model, device, config):
        super(CMTInferencer, self).__init__(asset_path, model, device, config)

    def infer(self, loader, load_rhythm=False):
        return  # TODO: Understand what should happen below in order to do inference conditioned on a chord progression.
        indices = random.sample(range(len(loader.dataset)), self.config["num_sample"])
        batch = collate_fn([loader.dataset[i] for i in indices])
        for key in batch.keys():
            batch[key] = batch[key].to(self.device)
        # TODO: num prime is the starting moments, which the model continues on...
        prime = batch['pitch'][:, :self.config["num_prime"]]
        if isinstance(self.model, torch.nn.DataParallel):
            model = self.model.module
        else:
            model = self.model
        prime_rhythm = batch['rhythm'][:, :self.config["num_prime"]]
        result_dict = model.sampling(prime_rhythm, prime, batch['chord'],
                                     self.config["topk"], self.config['attention_map'])
        # TODO: this is where the sample is made
        result_key = 'pitch'
        pitch_idx = result_dict[result_key].cpu().numpy()

        exit(1) # TODO: be careful not to overwrite anything

        logger.info("==========sampling result of epoch %03d==========" % restore_epoch)
        os.makedirs(os.path.join(asset_path, 'sampling_results', 'epoch_%03d' % restore_epoch), exist_ok=True)

        for sample_id in range(pitch_idx.shape[0]):
            logger.info(("Sample %02d : " % sample_id) + str(pitch_idx[sample_id][self.config["num_prime"]:self.config["num_prime"]+20]))
            save_path = os.path.join(asset_path, 'sampling_results', 'epoch_%03d' % restore_epoch,
                                     'epoch%03d_sample%02d.mid' % (restore_epoch, sample_id))
            gt_pitch = batch['pitch'].cpu().numpy()
            gt_chord = batch['chord'][:, :-1].cpu().numpy()
            sample_dict = {'pitch': pitch_idx[sample_id],
                           'rhythm': result_dict['rhythm'][sample_id].cpu().numpy(),
                           'chord': csc_matrix(gt_chord[sample_id])}


            with open(save_path.replace('.mid', '.pkl'), 'wb') as f_samp:
                pickle.dump(sample_dict, f_samp)
            instruments = pitch_to_midi(pitch_idx[sample_id], gt_chord[sample_id], model.frame_per_bar, save_path)
            save_instruments_as_image(save_path.replace('.mid', '.jpg'), instruments,
                                      frame_per_bar=model.frame_per_bar,
                                      num_bars=(model.max_len // model.frame_per_bar))

            # save groundtruth
            logger.info(("Groundtruth %02d : " % sample_id) +
                        str(gt_pitch[sample_id, self.config["num_prime"]:self.config["num_prime"] + 20]))
            gt_path = os.path.join(asset_path, 'sampling_results', 'epoch_%03d' % restore_epoch,
                                     'epoch%03d_groundtruth%02d.mid' % (restore_epoch, sample_id))
            gt_dict = {'pitch': gt_pitch[sample_id, :-1],
                       'rhythm': batch['rhythm'][sample_id, :-1].cpu().numpy(),
                       'chord': csc_matrix(gt_chord[sample_id])}
            with open(gt_path.replace('.mid', '.pkl'), 'wb') as f_gt:
                pickle.dump(gt_dict, f_gt)
            gt_instruments = pitch_to_midi(gt_pitch[sample_id, :-1], gt_chord[sample_id], model.frame_per_bar, gt_path)
            save_instruments_as_image(gt_path.replace('.mid', '.jpg'), gt_instruments,
                                      frame_per_bar=model.frame_per_bar,
                                      num_bars=(model.max_len // model.frame_per_bar))

            if self.config['attention_map']:
                os.makedirs(os.path.join(asset_path, 'attention_map', 'epoch_%03d' % restore_epoch,
                                         'RDec-Chord', 'sample_%02d' % sample_id), exist_ok=True)

                for head_num in range(8):
                    for l, w in enumerate(result_dict['weights_bdec']):
                        fig_w = plt.figure(figsize=(8, 8))
                        ax_w = fig_w.add_subplot(1, 1, 1)
                        heatmap_w = ax_w.pcolor(w[sample_id, head_num].cpu().numpy(), cmap='Reds')
                        ax_w.set_xticks(np.arange(0, self.model.module.max_len))
                        ax_w.xaxis.tick_top()
                        ax_w.set_yticks(np.arange(0, self.model.module.max_len))
                        ax_w.set_xticklabels(rhythm_to_symbol_list(result_dict['rhythm'][sample_id].cpu().numpy()),
                                             fontdict=x_fontdict)
                        chord_symbol_list = [''] * pitch_idx.shape[1]
                        for t in sorted(chord_array_to_dict(gt_chord[sample_id]).keys()):
                            chord_symbol_list[t] = chord_array_to_dict(gt_chord[sample_id])[t].tolist()
                        ax_w.set_yticklabels(chord_to_symbol_list(gt_chord[sample_id]), fontdict=y_fontdict)
                        ax_w.invert_yaxis()
                        plt.savefig(os.path.join(asset_path, 'attention_map', 'epoch_%03d' % restore_epoch, 'RDec-Chord',
                                                 'sample_%02d' % sample_id,
                                                 'epoch%03d_RDec-Chord_sample%02d_head%02d_layer%02d.jpg' % (
                                                 restore_epoch, sample_id, head_num, l)))
                        plt.close()
