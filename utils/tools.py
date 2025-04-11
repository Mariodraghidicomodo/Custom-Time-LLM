import numpy as np
import torch
import matplotlib.pyplot as plt
import shutil
#----- AGGIUNTE
#from run_main import test_writer #da provare
from torch.utils.tensorboard import SummaryWriter
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import pandas as pd
#-----
from tqdm import tqdm

plt.switch_backend('agg')


def adjust_learning_rate(accelerator, optimizer, scheduler, epoch, args, printout=True):
    if args.lradj == 'type1':
        lr_adjust = {epoch: args.learning_rate * (0.5 ** ((epoch - 1) // 1))}
    elif args.lradj == 'type2':
        lr_adjust = {
            2: 5e-5, 4: 1e-5, 6: 5e-6, 8: 1e-6,
            10: 5e-7, 15: 1e-7, 20: 5e-8
        }
    elif args.lradj == 'type3':
        lr_adjust = {epoch: args.learning_rate if epoch < 3 else args.learning_rate * (0.9 ** ((epoch - 3) // 1))}
    elif args.lradj == 'PEMS':
        lr_adjust = {epoch: args.learning_rate * (0.95 ** (epoch // 1))}
    elif args.lradj == 'TST':
        lr_adjust = {epoch: scheduler.get_last_lr()[0]}
    elif args.lradj == 'constant':
        lr_adjust = {epoch: args.learning_rate}
    if epoch in lr_adjust.keys():
        lr = lr_adjust[epoch]
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        if printout:
            if accelerator is not None:
                accelerator.print('Updating learning rate to {}'.format(lr))
            else:
                print('Updating learning rate to {}'.format(lr))


class EarlyStopping:
    def __init__(self, accelerator=None, patience=7, verbose=False, delta=0, save_mode=True):
        self.accelerator = accelerator
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        #self.val_loss_min = np.Inf
        self.val_loss_min = np.inf #modificato il codice perchè np.Inf non è più supportato
        self.delta = delta
        self.save_mode = save_mode

    def __call__(self, val_loss, model, path):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            if self.save_mode:
                self.save_checkpoint(val_loss, model, path)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.accelerator is None:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            else:
                self.accelerator.print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            if self.save_mode:
                self.save_checkpoint(val_loss, model, path)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, path):
        if self.verbose:
            if self.accelerator is not None:
                self.accelerator.print(
                    f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
            else:
                print(
                    f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')

        if self.accelerator is not None:
            model = self.accelerator.unwrap_model(model)
            torch.save(model.state_dict(), path + '/' + 'checkpoint')
        else:
            torch.save(model.state_dict(), path + '/' + 'checkpoint')
        self.val_loss_min = val_loss


class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


class StandardScaler(): #class/function for standardization of the data
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def transform(self, data):
        return (data - self.mean) / self.std

    def inverse_transform(self, data):
        return (data * self.std) + self.mean

def adjustment(gt, pred):
    anomaly_state = False
    for i in range(len(gt)):
        if gt[i] == 1 and pred[i] == 1 and not anomaly_state:
            anomaly_state = True
            for j in range(i, 0, -1):
                if gt[j] == 0:
                    break
                else:
                    if pred[j] == 0:
                        pred[j] = 1
            for j in range(i, len(gt)):
                if gt[j] == 0:
                    break
                else:
                    if pred[j] == 0:
                        pred[j] = 1
        elif gt[i] == 0:
            anomaly_state = False
        if anomaly_state:
            pred[i] = 1
    return gt, pred


def cal_accuracy(y_pred, y_true):
    return np.mean(y_pred == y_true)


def del_files(dir_path):
    shutil.rmtree(dir_path)


def vali(args, accelerator, model, vali_data, vali_loader, criterion, mae_metric): #esegue la validazione, questa parte è utile per capire la precisione del modello da qua devo ritornare i valori che predice e confrontarli con quelli reali (fare anche un grafico e capire mea e altri errori)
    total_loss = []
    total_mae_loss = []
    model.eval()
#----- AGGIUNTE
    predictions = []  
    actuals = []  
#-----
    with torch.no_grad(): #inference?
        for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in tqdm(enumerate(vali_loader)):
            batch_x = batch_x.float().to(accelerator.device)
            batch_y = batch_y.float()

            batch_x_mark = batch_x_mark.float().to(accelerator.device)
            batch_y_mark = batch_y_mark.float().to(accelerator.device)

            # decoder input
            dec_inp = torch.zeros_like(batch_y[:, -args.pred_len:, :]).float()
            dec_inp = torch.cat([batch_y[:, :args.label_len, :], dec_inp], dim=1).float().to(
                accelerator.device)
            # encoder - decoder
            if args.use_amp:
                with torch.cuda.amp.autocast():
                    if args.output_attention:
                        outputs = model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                    else:
                        outputs = model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
            else:
                if args.output_attention:
                    outputs = model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                else:
                    outputs = model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

            outputs, batch_y = accelerator.gather_for_metrics((outputs, batch_y))

            f_dim = -1 if args.features == 'MS' else 0
            outputs = outputs[:, -args.pred_len:, f_dim:]
            batch_y = batch_y[:, -args.pred_len:, f_dim:].to(accelerator.device)

            pred = outputs.detach() #qua adesso abbiamo i valori predetti
            true = batch_y.detach() #qua abbiamo i valori reali
            #qua potremmo salvare i valori e fare un GRAFICO da mettere su tensor
#----- AGGIUNTE
            predictions.append(pred.cpu().numpy()) 
            actuals.append(true.cpu().numpy()) 
            #print('PREDICTION: ',predictions)
            #print('ACTUALS: ',actuals)
#-----
            loss = criterion(pred, true)

            mae_loss = mae_metric(pred, true)

            total_loss.append(loss.item())
            total_mae_loss.append(mae_loss.item())

    total_loss = np.average(total_loss)
    total_mae_loss = np.average(total_mae_loss)
#----- AGGIUNTE
    predictions = np.concatenate(predictions, axis = 0)
    #predictions = np.concatenate([pred.cpu().numpy() for pred in predictions], axis=0)
    actuals = np.concatenate(actuals, axis = 0)
    #actuals = np.concatenate([true.cpu().numpy() for true in actuals], axis = 0)
    mean,std = vali_data.get_scaler_params()
    scaler = StandardScaler(mean,std)
    predictions_norm = scaler.inverse_transform(predictions)
    actuals_norm = scaler.inverse_transform(actuals)
    #print('PREDICTION: ',predictions)
    #print('ACTUALS: ',actuals)
    #scaler = StandardScaler() #SBAGLIATO
    #predictions_normal = predictions.scaler.inverse_transform(predictions.reshape(-1,1))
    #actuals_normal = actuals.scaler.inverse_transform(actuals.reshape(-1,1))
    #print('PREDICTION NORMAL: ',predictions_normal.reshape(1,-1))
    #print('ACTUALS NORMAL: ',actuals_normal.reshape(1,-1))
    test_writer = SummaryWriter(log_dir='runs/model_small_small') #open writer
    
    for step in range(predictions.shape[0]): #loop samples
      test_writer.add_scalars("Predictions vs Actuals",{"Predicted":predictions[step].mean(), "Actual":actuals[step].mean()}, step) #name, dict, step
    
    for step in range(predictions.shape[0]): #loop samples
      test_writer.add_scalars("Predictions Normal vs Actuals Normal",{"Predicted Normal":predictions_norm[step].mean(), "Actual Normal":actuals_norm[step].mean()}, step) #name, dict, step
    
    #or
    fig,ax = plt.subplots(figsize=(10,5))
    ax.plot(actuals[0], label = 'Actual') #hanno una struttura del tipo 40,1,90
    #ax.plot(actuals, label = 'Actual')
    ax.plot(predictions[0], label = 'Predictions', color='red')
    #ax.plot(predictions, label = 'Predictions', color='red')
    ax.legend()
    ax.set_title('Prediction vs Actual')
    test_writer.add_figure("Prediction vs Actual (simple plot)", fig)

    fig,ax = plt.subplots(figsize=(10,5))
    ax.plot(actuals_norm[0], label = 'Actual')
    #ax.plot(actuals, label = 'Actual Normal')
    ax.plot(predictions_norm[0], label = 'Predictions', color='red')
    #ax.plot(predictions, label = 'Predictions Normal', color='red')
    ax.legend()
    ax.set_title('Prediction vs Actual NORMAL')
    test_writer.add_figure("Prediction Normal vs Actual Normal (simple plot)", fig)

    test_writer.close() #close writer
#-----

    model.train()
    return total_loss, total_mae_loss


def test(args, accelerator, model, train_loader, vali_loader, criterion):
    x, _ = train_loader.dataset.last_insample_window()
    y = vali_loader.dataset.timeseries
    x = torch.tensor(x, dtype=torch.float32).to(accelerator.device)
    x = x.unsqueeze(-1)

    model.eval()
    with torch.no_grad():
        B, _, C = x.shape
        dec_inp = torch.zeros((B, args.pred_len, C)).float().to(accelerator.device)
        dec_inp = torch.cat([x[:, -args.label_len:, :], dec_inp], dim=1)
        outputs = torch.zeros((B, args.pred_len, C)).float().to(accelerator.device)
        id_list = np.arange(0, B, args.eval_batch_size)
        id_list = np.append(id_list, B)
        for i in range(len(id_list) - 1):
            outputs[id_list[i]:id_list[i + 1], :, :] = model(
                x[id_list[i]:id_list[i + 1]],
                None,
                dec_inp[id_list[i]:id_list[i + 1]],
                None
            )
        accelerator.wait_for_everyone()
        outputs = accelerator.gather_for_metrics(outputs)
        f_dim = -1 if args.features == 'MS' else 0
        outputs = outputs[:, -args.pred_len:, f_dim:]
        pred = outputs
        true = torch.from_numpy(np.array(y)).to(accelerator.device)
        batch_y_mark = torch.ones(true.shape).to(accelerator.device)
        true = accelerator.gather_for_metrics(true)
        batch_y_mark = accelerator.gather_for_metrics(batch_y_mark)

        loss = criterion(x[:, :, 0], args.frequency_map, pred[:, :, 0], true, batch_y_mark)

    model.train()
    return loss


def load_content(args):
    if 'ETT' in args.data:
        file = 'ETT'
    else:
        file = args.data
    with open('./dataset/prompt_bank/{0}.txt'.format(file), 'r', encoding="utf-8") as f:
        content = f.read() #attenzion può dare problemi, meglio esplicitare l encoding (è stato aggiunto encoding -> perfetto adesso funziona)
    return content