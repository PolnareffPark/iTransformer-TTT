from experiments.exp_long_term_forecasting import Exp_Long_Term_Forecast
from utils.metrics import metric
from data_provider.data_factory import data_provider
from utils.tools import visual
import torch
import torch.nn as nn
from torch import optim
import os
import time
import warnings
import numpy as np
import copy
import datetime
import csv

warnings.filterwarnings('ignore')

class FFTLoss(nn.Module):
    def __init__(self):
        super(FFTLoss, self).__init__()

    def forward(self, pred, target):
        # pred, target: [B, L, D]
        # FFT along time dimension (dim=1)
        pred_fft = torch.fft.rfft(pred, dim=1)
        target_fft = torch.fft.rfft(target, dim=1)
        
        # Amplitude difference
        loss_amp = torch.abs(torch.abs(pred_fft) - torch.abs(target_fft)).mean()
        # Phase difference (optional, might be noisy for TTT)
        # loss_phase = torch.abs(torch.angle(pred_fft) - torch.angle(target_fft)).mean()
        
        return loss_amp

class Exp_TTT(Exp_Long_Term_Forecast):
    def __init__(self, args):
        super(Exp_TTT, self).__init__(args)
        # TTT Hyperparameters
        self.ttt_lr = args.ttt_lr if hasattr(args, 'ttt_lr') else 1e-4
        self.ttt_steps = args.ttt_steps if hasattr(args, 'ttt_steps') else 1
        self.ttt_mode = 'norm_only' # 'all', 'norm_only'
        self.fft_lambda = 0.1 # Weight for FFT loss

    def _get_ttt_optimizer(self, model):
        # Filter parameters based on mode
        if self.ttt_mode == 'norm_only':
            params = []
            for name, param in model.named_parameters():
                if 'norm' in name.lower() or 'ln' in name.lower() or 'bias' in name.lower():
                    params.append(param)
                    param.requires_grad = True
                else:
                    param.requires_grad = False
        else:
            params = model.parameters()
            
        return optim.Adam(params, lr=self.ttt_lr)

    def _reset_model_grad_state(self):
        # Unfreeze all parameters for inference (though inference usually doesn't need grad)
        # But we must restore state for next batch's full functionality if needed
        for param in self.model.parameters():
            param.requires_grad = True

    def test(self, setting, test=0):
        test_data, test_loader = self._get_data(flag='test')
        if test:
            print('loading model')
            self.model.load_state_dict(torch.load(os.path.join('./checkpoints/' + setting, 'checkpoint.pth')))

        preds = []
        trues = []
        folder_path = './test_results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
            
        print(f"Starting TTT V2 (Structural) with LR={self.ttt_lr}, Steps={self.ttt_steps}, Mode={self.ttt_mode}")
        
        fft_criterion = FFTLoss()
        mse_criterion = nn.MSELoss()
        
        self.model.eval() # Always Eval mode to disable dropout noise
        
        for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(test_loader):
            batch_x = batch_x.float().to(self.device)
            batch_y = batch_y.float().to(self.device)
            if 'PEMS' not in self.args.data and 'Solar' not in self.args.data:
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)
            else:
                batch_x_mark = None
                batch_y_mark = None

            # [Step 1] Backup & Setup
            original_state = {k: v.clone() for k, v in self.model.state_dict().items()}
            
            # [Step 2] Configure Optimizer & Partial Updates
            optimizer = self._get_ttt_optimizer(self.model) # This also handles freezing
            if self.args.use_amp:
                scaler = torch.cuda.amp.GradScaler()

            # Handle Norm (Model Agnostic)
            original_use_norm = None
            if hasattr(self.model, 'use_norm'):
                original_use_norm = self.model.use_norm
                self.model.use_norm = False # Disable internal norm for TTT stability

            # [Step 3] Proxy Task Construction (Aligned Shifted Window)
            seq_len = self.args.seq_len # 96
            proxy_len = min(seq_len // 2, self.args.pred_len)
            sub_in_len = seq_len - proxy_len
            
            sub_in = batch_x[:, :sub_in_len, :]     
            sub_target = batch_x[:, sub_in_len:, :] 
            
            # Manual Normalization
            means = sub_in.mean(1, keepdim=True).detach()
            sub_in_norm = sub_in - means
            stdev = torch.sqrt(torch.var(sub_in_norm, dim=1, keepdim=True, unbiased=False) + 1e-5)
            sub_in_norm /= stdev
            sub_target_norm = (sub_target - means) / stdev

            # Input Construction: [Pad, sub_in]
            pad_len = seq_len - sub_in_len
            pad = torch.zeros(batch_x.shape[0], pad_len, batch_x.shape[2], device=batch_x.device)
            proxy_x = torch.cat([pad, sub_in_norm], dim=1)
            
            # Marks Alignment
            if batch_x_mark is not None:
                pad_mark = torch.zeros(batch_x_mark.shape[0], pad_len, batch_x_mark.shape[2], device=batch_x_mark.device)
                sub_mark = batch_x_mark[:, :sub_in_len, :]
                proxy_x_mark = torch.cat([pad_mark, sub_mark], dim=1)
                proxy_y_mark = batch_y_mark 
            else:
                proxy_x_mark = None
                proxy_y_mark = None

            # Dec Construction
            label_len = min(self.args.label_len, sub_in_len)
            dec_zeros = torch.zeros(batch_x.shape[0], self.args.pred_len, batch_x.shape[2]).float().to(self.device)
            proxy_dec_inp = torch.cat([sub_in_norm[:, -label_len:, :], dec_zeros], dim=1).float().to(self.device)

            # [Step 4] Adaptation Loop
            for step in range(self.ttt_steps):
                optimizer.zero_grad()
                
                with torch.cuda.amp.autocast(enabled=self.args.use_amp):
                    if self.args.output_attention:
                        outputs_proxy = self.model(proxy_x, proxy_x_mark, proxy_dec_inp, proxy_y_mark)[0]
                    else:
                        outputs_proxy = self.model(proxy_x, proxy_x_mark, proxy_dec_inp, proxy_y_mark)
                
                # Slicing for Loss
                f_dim = -1 if self.args.features == 'MS' else 0
                slice_len = min(self.args.pred_len, proxy_len)
                
                pred_proxy = outputs_proxy[:, :slice_len, f_dim:]
                true_proxy = sub_target_norm[:, :slice_len, f_dim:]
                
                # Combined Loss
                loss_mse = mse_criterion(pred_proxy, true_proxy)
                loss_fft = fft_criterion(pred_proxy, true_proxy)
                loss = loss_mse + self.fft_lambda * loss_fft
                
                if torch.isnan(loss):
                    break
                    
                if self.args.use_amp:
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()

            # [Step 5] Inference
            if original_use_norm is not None:
                self.model.use_norm = original_use_norm
            self._reset_model_grad_state() # Unfreeze for safety (though loading state overwrites params)

            self.model.eval()
            with torch.no_grad():
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                
                with torch.cuda.amp.autocast(enabled=self.args.use_amp):
                    if self.args.output_attention:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                    else:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                
                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y_inf = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)

                outputs = outputs.detach().cpu().numpy()
                batch_y_inf = batch_y_inf.detach().cpu().numpy()

                if test_data.scale and self.args.inverse:
                    outputs = test_data.inverse_transform(outputs.squeeze(0)).reshape(outputs.shape)
                    batch_y_inf = test_data.inverse_transform(batch_y_inf.squeeze(0)).reshape(batch_y_inf.shape)

                preds.append(outputs)
                trues.append(batch_y_inf)

            # [Step 6] Restore State
            self.model.load_state_dict(original_state)

        # Metrics
        preds = np.array(preds)
        trues = np.array(trues)
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])
        
        mae, mse, rmse, mape, mspe = metric(preds, trues)
        print('mse:{}, mae:{}'.format(mse, mae))
        
        # Logging
        summary_path = './test_results/summary.csv'
        if not os.path.exists('./test_results'):
            os.makedirs('./test_results')
        file_exists = os.path.isfile(summary_path)
        with open(summary_path, 'a', newline='') as csvfile:
            headers = ['timestamp', 'model_id', 'model', 'data', 'mse', 'mae', 'setting', 'ttt_lr', 'ttt_steps', 'ttt_mode']
            writer = csv.DictWriter(csvfile, fieldnames=headers)
            if not file_exists:
                writer.writeheader()
            writer.writerow({
                'timestamp': datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                'model_id': self.args.model_id,
                'model': self.args.model,
                'data': self.args.data,
                'mse': mse,
                'mae': mae,
                'setting': setting,
                'ttt_lr': self.ttt_lr,
                'ttt_steps': self.ttt_steps,
                'ttt_mode': self.ttt_mode
            })
            
        np.save(folder_path + 'pred.npy', preds)
        np.save(folder_path + 'true.npy', trues)
