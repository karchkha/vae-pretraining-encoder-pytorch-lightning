from pytorch_lightning.callbacks import Callback, TQDMProgressBar
from pytorch_lightning.loggers import TensorBoardLogger
from data import MonoTextData
from modules.Lit_vae import VAE
import torch
from utils import calc_iwnll, calc_mi, calc_au
# from tqdm import tqdm
# import sys

from utils import visualize_latent

####################### CALLBACKS ################################



#########################TEXT LOGGER ######################################        
        



class TextLogger(Callback):
    def __init__(self, args):
        super().__init__()
        self.args = args
        # self.batch_frequency = batch_frequency 
        # self.logger_log_images = {
        #     TensorBoardLogger(save_dir = "lightning_logs/"+args.dataset, name = 'TextLoggs')
        # }

    def check_frequency(self, batch_idx):
        if batch_idx % self.args.logging_frequency == 0:
            return True
        return False
        
    def reconstruct(self, model, data, strategy, device):
        data = data[0].unsqueeze(0)   #limit to only one sentance
        sentence = ""
        decoded_batch = model.reconstruct(data, strategy)
        for sent in decoded_batch:
            line = " ".join(sent) + "  \n"
            sentence += line
        return sentence
        
    def batch_to_sentence(self, model, data):
        sentence = ""

        batch_size, sent_size = data.size()
        batch_size = 1 #### limit to only 1 sentance
        
        decoded_batch = [[] for _ in range(batch_size)]
        
        for i in range(batch_size):
            for j in range (1, sent_size-1):
              decoded_batch[i].append(model.vocab.id2word(data[i, j].item()))
        
        for sent in decoded_batch:
            line = " ".join(sent) + "  \n"
            sentence += line
        
        return sentence
        
    def sample_from_prior(self, model, z, strategy):
        sentence = ""
        decoded_batch = model.decode(z, strategy)

        for sent in decoded_batch:
            line = " ".join(sent) + "  \n"
            sentence += line
        
        return sentence
            

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if self.check_frequency(batch_idx):
            self.log_text(pl_module, batch, batch_idx, split='train')

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):
        if self.check_frequency(batch_idx) or batch_idx == pl_module.len_val_data-1:        
            self.log_text(pl_module, batch, batch_idx, split='val')
            

    def log_text(self, pl_module, batch, batch_idx, split='train'):

        logger = type(pl_module.logger)
        
        # print(logger)
       
        is_train = pl_module.training
        if is_train:
            pl_module.eval()

        with torch.no_grad():
            
            original = self.batch_to_sentence(pl_module, batch)
            
            text_reconstructed = self.reconstruct(pl_module, batch, self.args.decoding_strategy, self.args.device)

            # Sampling form prior 
            
            z = pl_module.sample_from_prior(1)
            
            # # print(z)
            sampled_from_prior = self.sample_from_prior(pl_module, z, self.args.decoding_strategy)
            

        
        pl_module.logger.experiment.add_text(f'{split}/epoch-{pl_module.current_epoch}/step-{batch_idx}/original', original, global_step=batch_idx)
        pl_module.logger.experiment.add_text(f'{split}/epoch-{pl_module.current_epoch}/step-{batch_idx}/reconstraction', text_reconstructed, global_step=batch_idx)
        pl_module.logger.experiment.add_text(f'{split}/epoch-{pl_module.current_epoch}/step-{batch_idx}/sampled_from_prior', sampled_from_prior, global_step=batch_idx)

        if is_train:
            pl_module.train()
            
            
            


################################ LR CONTROL + mi, au and ppl logging and PRINTING #############################################


class callbeck_of_my_dreams(Callback):
    def __init__(self):
        self.decay_cnt = 0
        self.not_improved = 0
        self.decay_epoch = 5
        self.lr_decay = 0.5

    def on_validation_end(self, trainer, pl_module):
        
        
        
        with torch.no_grad():
            print("\rcalculating mutual_info", end="",flush=True)
            cur_mi = pl_module.calc_mi(pl_module.val_data)
        

            print("\rCalculating active units", end="",flush=True)    
            au, au_var = pl_module.calc_au(pl_module.val_data)
        
        

        print('\rEpoch: %d - loss: %.4f, kl: %.4f, recon: %.4f, nll: %.4f, ppl: %.4f, active_units: %d, mutual_info: %.4f' % (pl_module.current_epoch, 
                                                                                                                            pl_module.test_loss,
                                                                                                                            pl_module.kl_loss, 
                                                                                                                            pl_module.rec_loss, 
                                                                                                                            pl_module.nll, 
                                                                                                                            pl_module.ppl, 
                                                                                                                            au, cur_mi))
        
        
        pl_module.logger.experiment.add_scalar("metrics/mutual_info",  cur_mi, global_step=pl_module.current_epoch)
        pl_module.logger.experiment.add_scalar("metrics/active_units",  au, global_step=pl_module.current_epoch)
        pl_module.logger.experiment.add_scalar("metrics/ppl",  pl_module.ppl, global_step=pl_module.current_epoch)
        pl_module.logger.experiment.add_scalar("metrics/nll",  pl_module.nll, global_step=pl_module.current_epoch)
        pl_module.logger.experiment.add_scalar("metrics/starting_best_loss",  pl_module.best_loss, global_step=pl_module.current_epoch)
        
        
        if trainer.state.fn=="fit":
            
            pl_module.lr = pl_module.get_lr() # to make sure we have everything in sync and read lr form checkpoints correctly
            
                
            if pl_module.test_loss > pl_module.best_loss :

                self.not_improved += 1
                if self.not_improved >= self.decay_epoch and pl_module.current_epoch >=15:
                    
                    ##############################
                    
                    print("model did't improve for more than %d epochs so we load the best model ckpt %s" % (self.decay_epoch, 
                                                                                                            trainer.checkpoint_callback.best_model_path))
                    
                    # Here we load model to another variable and then attach parts of it to running pl_module. There must be better way to do this but I didn't find it.
                    # just loading checkpoint was breaking trainer rutine and I wasn't able to change lr and access optimisers.
                    pl_module_ckpt = VAE.load_from_checkpoint(trainer.checkpoint_callback.best_model_path, args = pl_module.args).to(pl_module.args.device)
                    pl_module.encoder = pl_module_ckpt.encoder
                    pl_module.decoder = pl_module_ckpt.decoder
                    pl_module.best_loss = pl_module_ckpt.best_loss
                    pl_module.pre_mi = pl_module_ckpt.pre_mi
                    pl_module.kl_weight = pl_module_ckpt.kl_weight
                    
                    
                    ##################################
                    
                    pl_module.lr = pl_module.lr * self.lr_decay
                    pl_module.set_lr(pl_module.lr)
                    
                    print("\rEpoch: %d - Best loss was: %.4f not_improved: %d and new lr to: %.4f\n" % (pl_module.current_epoch, 
                                                                                                  pl_module.best_loss,
                                                                                                  self.not_improved, 
                                                                                                  pl_module.lr, 
                                                                                                  ))
                    self.not_improved = 0
                    # pl_module.best_loss = pl_module.test_loss  # Best loss will be taken from checkpoint!
                else:
                    print("\rEpoch: %d - Best loss: %.4f not_improved: %d and lr : %.4f\n" % (pl_module.current_epoch, 
                                                                                                  pl_module.best_loss,
                                                                                                  self.not_improved, 
                                                                                                  pl_module.lr, 
                                                                                                  ))
                
        
            else:
                
                self.not_improved = 0
                print("\rEpoch: %d - Best loss was: %.4f not_improved: %d lr %.4f setting best_loss %.4f\n" % (pl_module.current_epoch, 
                                                                                                pl_module.best_loss,
                                                                                                self.not_improved, 
                                                                                                pl_module.lr,
                                                                                                pl_module.test_loss 
                                                                                                ))
                
                pl_module.best_loss = pl_module.test_loss
                
            if pl_module.args.save_latent > 0 and pl_module.current_epoch <= pl_module.args.save_latent:
                visualize_latent(args, epoch, vae, "cuda", test_data)
                
                
        else:
            
            print("\rCurrunt epoch: %d - Best loss was: %.4f lr %.4f \n" % (pl_module.current_epoch, 
                                                                                pl_module.best_loss,
                                                                                pl_module.lr
                                                                                ))  #### here put everything you wanna print in time of evaluation and testing
        
        

        
        

        
        