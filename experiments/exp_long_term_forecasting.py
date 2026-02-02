from data_provider.data_factory import data_provider
from experiments.exp_basic import Exp_Basic
from utils.tools import EarlyStopping, adjust_learning_rate, visual
from utils.metrics import metric
import torch
import torch.nn as nn
from torch import optim
import os
import time
import warnings
import numpy as np
import csv
import datetime

try:
    from thop import profile
except ImportError:
    profile = None

warnings.filterwarnings('ignore')


class Exp_Long_Term_Forecast(Exp_Basic):
    def __init__(self, args):
        super(Exp_Long_Term_Forecast, self).__init__(args)

    def _build_model(self):
        model = self.model_dict[self.args.model].Model(self.args).float()

        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim

    def _select_criterion(self):
        criterion = nn.MSELoss()
        return criterion

    def vali(self, vali_data, vali_loader, criterion):
        total_loss = []
        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(vali_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                if getattr(self.args, 'noise_std', 0.0) > 0:
                    batch_x = batch_x + torch.randn_like(batch_x) * self.args.noise_std

                if 'PEMS' in self.args.data or 'Solar' in self.args.data:
                    batch_x_mark = None
                    batch_y_mark = None
                else:
                    batch_x_mark = batch_x_mark.float().to(self.device)
                    batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    if self.args.output_attention:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                    else:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)

                pred = outputs.detach().cpu()
                true = batch_y.detach().cpu()

                loss = criterion(pred, true)

                total_loss.append(loss)
        total_loss = np.average(total_loss)
        self.model.train()
        return total_loss

    def train(self, setting):
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')
        test_data, test_loader = self._get_data(flag='test')

        if self.args.output_subdir:
            path = os.path.join(self.args.checkpoints, self.args.output_subdir, setting)
        else:
            path = os.path.join(self.args.checkpoints, setting)
            
        if not os.path.exists(path):
            os.makedirs(path)

        time_now = time.time()

        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        model_optim = self._select_optimizer()
        criterion = self._select_criterion()

        time_now_overall = time.time()
        torch.cuda.reset_peak_memory_stats(self.device)

        if self.args.use_amp:
            scaler = torch.cuda.amp.GradScaler()

        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []

            self.model.train()
            epoch_time = time.time()
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):
                iter_count += 1
                model_optim.zero_grad()
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                if getattr(self.args, 'noise_std', 0.0) > 0:
                    batch_x = batch_x + torch.randn_like(batch_x) * self.args.noise_std

                if 'PEMS' in self.args.data or 'Solar' in self.args.data:
                    batch_x_mark = None
                    batch_y_mark = None
                else:
                    batch_x_mark = batch_x_mark.float().to(self.device)
                    batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)

                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                        f_dim = -1 if self.args.features == 'MS' else 0
                        outputs = outputs[:, -self.args.pred_len:, f_dim:]
                        batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                        loss = criterion(outputs, batch_y)
                        train_loss.append(loss.item())
                else:
                    if self.args.output_attention:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                    else:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                    f_dim = -1 if self.args.features == 'MS' else 0
                    outputs = outputs[:, -self.args.pred_len:, f_dim:]
                    batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                    loss = criterion(outputs, batch_y)
                    train_loss.append(loss.item())

                if (i + 1) % 100 == 0:
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()

                if self.args.use_amp:
                    scaler.scale(loss).backward()
                    scaler.step(model_optim)
                    scaler.update()
                else:
                    loss.backward()
                    model_optim.step()

                if self.args.debug and (i + 1) >= 10:
                    print("DEBUG MODE: Breaking epoch early after 10 iterations")
                    break


            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.average(train_loss)
            vali_loss = self.vali(vali_data, vali_loader, criterion)
            test_loss = self.vali(test_data, test_loader, criterion)

            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f}".format(
                epoch + 1, train_steps, train_loss, vali_loss, test_loss))
            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break

            adjust_learning_rate(model_optim, epoch + 1, self.args)

        # After all epochs, capture peak reserved memory (including grads, optimizer states)
        self.train_peak_vram = torch.cuda.max_memory_reserved(self.device) / 1024 / 1024 / 1024  # GB
        self.total_train_time = time.time() - time_now_overall
        print("Final Training Peak VRAM: {:.2f}GB".format(self.train_peak_vram))

        best_model_path = path + '/' + 'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))

        return self.model

    def test(self, setting, test=0):
        test_data, test_loader = self._get_data(flag='test')
        if test:
            print('loading model')
            self.model.load_state_dict(torch.load(os.path.join('./checkpoints/' + setting, 'checkpoint.pth')))

        self.model.eval()
        
    def profile_model(self):
        """Separate profiling run to avoid impact on main experiment timing"""
        if profile is not None:
            from layers.SelfAttention_Family import FullAttention
            from layers.Hierarchical_Attention import HierarchicalAttention
            
            def count_full_attention(m, x, y):
                # x: (queries, keys, values, attn_mask, tau, delta)
                queries, keys, values = x[0], x[1], x[2]
                B, L, H, E = queries.shape
                _, S, _, D = values.shape
                # einsum("blhe,bshe->bhls")
                flops_qk = B * H * L * S * E * 2
                # einsum("bhls,bshd->blhd")
                flops_av = B * H * L * S * D * 2
                m.total_ops += torch.DoubleTensor([flops_qk + flops_av])

            def count_hierarchical_attention(m, x, y):
                # x: (queries, keys, values, attn_mask, tau, delta)
                queries, keys, values = x[0], x[1], x[2]
                B, N, H, D = queries.shape
                G = m.num_groups
                # Pad if N not divisible by G
                pad_len = (G - (N % G)) % G
                N_padded = N + pad_len
                M = N_padded // G
                
                # 1. Local Attention: G groups of (M x M) attention
                # einsum ops: B * G * H * M * M * D * 2 (for QK and AV)
                flops_local = B * G * H * M * M * D * 2 * 2 
                
                # 2. Global Attention: 1 group of (G x G) attention (simplified)
                # or statistical (3*G x 3*G)
                if m.pooling == 'statistical':
                    G_global = G * 3
                else:
                    G_global = G
                flops_global = B * H * G_global * G_global * D * 2 * 2
                
                m.total_ops += torch.DoubleTensor([flops_local + flops_global])

            self.model.eval()
            try:
                # Use a dummy input for profiling
                dummy_x = torch.randn(1, self.args.seq_len, self.args.enc_in).to(self.device)
                dummy_mark = torch.randn(1, self.args.seq_len, 4).to(self.device) if not ('PEMS' in self.args.data or 'Solar' in self.args.data) else None
                dummy_dec = torch.randn(1, self.args.label_len + self.args.pred_len, self.args.dec_in).to(self.device)
                dummy_y_mark = torch.randn(1, self.args.label_len + self.args.pred_len, 4).to(self.device) if not ('PEMS' in self.args.data or 'Solar' in self.args.data) else None
                
                custom_ops = {
                    FullAttention: count_full_attention,
                    HierarchicalAttention: count_hierarchical_attention
                }
                
                flops, params = profile(self.model, inputs=(dummy_x, dummy_mark, dummy_dec, dummy_y_mark), 
                                        custom_ops=custom_ops, verbose=False)
                return flops, params
            except Exception as e:
                print(f"Error profiling model: {e}")
        return 0, 0

    def test(self, setting, test=0):
        test_data, test_loader = self._get_data(flag='test')
        if test:
            print('loading model')
            self.model.load_state_dict(torch.load(os.path.join('./checkpoints/' + setting, 'checkpoint.pth')))

        self.model.eval()
        
        # 1. Metric-Only Run (isolated)
        print("Running independent metric profiling...")
        total_flops, total_params = self.profile_model()
        print("Warming up...")
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(test_loader):
                if i >= 5: break
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                self.model(batch_x, None, dec_inp, None)
        
        # 2. Main Latency & VRAM Tracking
        torch.cuda.synchronize(self.device) # Barrier
        torch.cuda.reset_peak_memory_stats(self.device)
        start_time = time.time()
        
        preds = []
        trues = []
        folder_path = './test_results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(test_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                if getattr(self.args, 'noise_std', 0.0) > 0:
                    batch_x = batch_x + torch.randn_like(batch_x) * self.args.noise_std

                if 'PEMS' in self.args.data or 'Solar' in self.args.data or 'stress' in self.args.data:
                    batch_x_mark = None
                    batch_y_mark = None
                else:
                    batch_x_mark = batch_x_mark.float().to(self.device)
                    batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    if self.args.output_attention:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                    else:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)


                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                outputs = outputs.detach().cpu().numpy()
                batch_y = batch_y.detach().cpu().numpy()
                if test_data.scale and self.args.inverse:
                    shape = outputs.shape
                    outputs = test_data.inverse_transform(outputs.squeeze(0)).reshape(shape)
                    batch_y = test_data.inverse_transform(batch_y.squeeze(0)).reshape(shape)

                pred = outputs
                true = batch_y

                preds.append(pred)
                trues.append(true)
                if i % 20 == 0:
                    input = batch_x.detach().cpu().numpy()
                    if test_data.scale and self.args.inverse:
                        shape = input.shape
                        input = test_data.inverse_transform(input.squeeze(0)).reshape(shape)
                    gt = np.concatenate((input[0, :, -1], true[0, :, -1]), axis=0)
                    pd = np.concatenate((input[0, :, -1], pred[0, :, -1]), axis=0)
                    visual(gt, pd, os.path.join(folder_path, str(i) + '.pdf'))
            
        # Get peak VRAM usage for inference
        inference_peak_vram = torch.cuda.max_memory_reserved(self.device) / 1024 / 1024 / 1024 # GB
        inference_time_total = time.time() - start_time
        inference_latency = inference_time_total / len(test_loader) # s/iter
        inference_speed = (len(test_loader) * self.args.batch_size) / inference_time_total # items/sec

        preds = np.array(preds)
        trues = np.array(trues)
        print('test shape:', preds.shape, trues.shape)
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])
        print('test shape:', preds.shape, trues.shape)

        # result save
        folder_path = './results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        mae, mse, rmse, mape, mspe = metric(preds, trues)
        print('mse:{}, mae:{}'.format(mse, mae))
        # Log BOTH training and inference metrics for transparency
        train_vram = getattr(self, 'train_peak_vram', 0)
        train_speed = getattr(self, 'total_train_time', 0) / self.args.train_epochs
        
        print('FLOPs: {:.2f}G, Params: {:.2f}M'.format(total_flops / 1e9, total_params / 1e6))
        print('Training: Peak VRAM {:.2f}GB, Speed {:.4f}s/epoch'.format(train_vram, train_speed))
        print('Inference: Peak VRAM {:.2f}GB, Latency {:.4f}s/iter, Speed {:.2f} items/sec'.format(inference_peak_vram, inference_latency, inference_speed))
        
        with open("result_long_term_forecast.txt", 'a') as f:
            f.write(setting + "  \n")
            f.write('mse:{}, mae:{}, flops:{:.4f}G, params:{:.4f}M, train_vram:{:.4f}GB, train_time:{:.4f}s, test_vram:{:.4f}GB, infer_latency:{:.4f}s, infer_speed:{:.2f}items/s\n'.format(
                mse, mae, total_flops/1e9, total_params/1e6, train_vram, train_speed, inference_peak_vram, inference_latency, inference_speed))
            f.write('\n')
            f.flush()
            os.fsync(f.fileno())
        print("Logged to result_long_term_forecast.txt")

        # New CSV logging with isolation support
        summary_filename = getattr(self.args, 'summary_file', 'summary.csv')
        summary_path = os.path.join('./test_results', summary_filename)
        
        if not os.path.exists('./test_results'):
            os.makedirs('./test_results')
            
        file_exists = os.path.isfile(summary_path)
        with open(summary_path, 'a', newline='') as csvfile:
            headers = ['timestamp', 'model_id', 'model', 'data', 'mse', 'mae', 'flops_G', 'params_M', 'train_vram_GB', 'train_time_s', 'test_vram_GB', 
                       'infer_latency_s', 'infer_speed_items_s', 'num_groups', 'pooling', 'dynamic_VR', 'dynamic_Bridge', 'dynamic_tokens', 'shuffling', 'seed', 'setting']
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
                'flops_G': f"{total_flops / 1e9:.4f}",
                'params_M': f"{total_params / 1e6:.4f}",
                'train_vram_GB': f"{train_vram:.4f}",
                'train_time_s': f"{train_speed:.4f}",
                'test_vram_GB': f"{inference_peak_vram:.4f}",
                'infer_latency_s': f"{inference_latency:.4f}",
                'infer_speed_items_s': f"{inference_speed:.2f}",
                'num_groups': getattr(self.args, 'num_groups', 0),
                'pooling': getattr(self.args, 'pooling', 'statistical'),
                'dynamic_VR': getattr(self.args, 'use_variable_resolution', 1),
                'dynamic_Bridge': getattr(self.args, 'use_interaction_bridge', 1),
                'dynamic_tokens': getattr(self.args, 'dynamic_tokens_per_group', 1),
                'shuffling': getattr(self.args, 'use_shuffling', 0),
                'seed': getattr(self.args, 'seed', -1),
                'setting': setting
            })
            csvfile.flush()
            os.fsync(csvfile.fileno())
        print("Logged to summary.csv")

        np.save(folder_path + 'metrics.npy', np.array([mae, mse, rmse, mape, mspe, total_flops, total_params, inference_peak_vram]))
        np.save(folder_path + 'pred.npy', preds)
        np.save(folder_path + 'true.npy', trues)

        # Save visualization data (Salience/Interaction gates) if available
        if self.args.output_attention and 'attns' in locals():
            torch.save(attns, folder_path + 'viz_data.pt')
            print(f"Visualization data saved to {folder_path}viz_data.pt")

        return


    def predict(self, setting, load=False):
        pred_data, pred_loader = self._get_data(flag='pred')

        if load:
            path = os.path.join(self.args.checkpoints, setting)
            best_model_path = path + '/' + 'checkpoint.pth'
            self.model.load_state_dict(torch.load(best_model_path))

        preds = []

        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(pred_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    if self.args.output_attention:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                    else:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                outputs = outputs.detach().cpu().numpy()
                if pred_data.scale and self.args.inverse:
                    shape = outputs.shape
                    outputs = pred_data.inverse_transform(outputs.squeeze(0)).reshape(shape)
                preds.append(outputs)

        preds = np.array(preds)
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])

        # result save
        folder_path = './results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        np.save(folder_path + 'real_prediction.npy', preds)

        return