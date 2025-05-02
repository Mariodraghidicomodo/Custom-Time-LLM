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


def vali(args, accelerator, model, vali_data, vali_loader, criterion, mae_metric, epoch, format): #esegue la validazione, questa parte è utile per capire la precisione del modello da qua devo ritornare i valori che predice e confrontarli con quelli reali (fare anche un grafico e capire mea e altri errori)
    total_loss = []
    total_mae_loss = []
    model.eval()
#----- AGGIUNTE
    predictions = []  
    actuals = []
    #pred_last = [] #questi dovrebbero avere la stessa lunghezza del test, quindi salvo l'ultima iterazione cosi dopo ha la stessa lunghezza del test e confortimao le date
    #true_last = []  
    #date = vali_data.get_date_strings()
    #print('lenght date', len(date['date'])) #dimostriamo che le date e i dati hanno lunghezza uguale quinid cosa succede quando facciamo i batch? perhcè non hanno lunghezza uguale in toools function vali??
    #print('lenght date_x', len(vali_data.data_x))
    all_batch_dates = []
    all_batch_dates_int = []
#-----
    with torch.no_grad(): #inference?
        for i, (batch_x, batch_y, batch_x_mark, batch_y_mark, batch_y_dates, batch_y_dates_int) in tqdm(enumerate(vali_loader)): #tolto
        #for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in tqdm(enumerate(vali_loader)):
            
            batch_x = batch_x.float().to(accelerator.device)
            batch_y = batch_y.float()
            batch_x_mark = batch_x_mark.float().to(accelerator.device)
            batch_y_mark = batch_y_mark.float().to(accelerator.device)
            batch_y_dates_int = batch_y_dates_int.to(accelerator.device) #addddddd

#-----AGGIUNTE 
            '''date_tesnor = torch.tensor([[ord(c) for c in d[0]] for d in batch_y_dates], device=accelerator.device)
            max_len = 32
            if date_tensor.shape[1] < max_len:
                padding = torch.zeros((date_tensor.shape[0], max_len - date_tensor.shape[1]), dtype=torch.long, device=accelerator.device)
                date_tensor = torch.cat((date_tensor, padding), dim=1)'''

#-----

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
            
            #outputs, batch_y = accelerator.gather_for_metrics((outputs, batch_y)) #qua raccoglie i valori distribuiti su più gpu es gpu1 = 8, gpu2 = 8 dopo questo punto batch_y = 16
#----- AGGIUNTE            
            outputs, batch_y, batch_y_dates_int = accelerator.gather_for_metrics((outputs, batch_y, batch_y_dates_int)) #qua raccoglie i valori distribuiti su più gpu es gpu1 = 8, gpu2 = 8 dopo questo punto batch_y = 16
            '''gathered_date_tensor = accelerator.gather_for_metrics(date_tensor)
            
            # Convert back to string
            gathered_dates = [''.join([chr(int(x)) for x in row if x != 0]) for row in gathered_date_tensor.cpu().numpy()]
            wewe.extend(gathered_dates)'''
#-----
            f_dim = -1 if args.features == 'MS' else 0

            #if type == 'test':
            print('output dim: ', outputs.shape)
            print('batch_y: ', batch_y.shape)
#            print('batch_y_date dim: ', batch_y_dates.shape) #se fosse np
            print('batch_y_date_int dim: ', batch_y_dates_int.shape) #se lista
            #print('batch_y_date type: ', type(batch_y_dates))
        
            outputs = outputs[:, -args.pred_len:, f_dim:]
            batch_y = batch_y[:, -args.pred_len:, f_dim:].to(accelerator.device)
            batch_y_dates_int = batch_y_dates_int[:, -args.pred_len:]
            wewe = np.array(batch_y_dates) #funziona
            wewe = wewe.transpose(2,0,1)
            print('type wewe: ', type(wewe))
            #print('wewe: ', wewe)
            print('wewe: ' , wewe.shape)
            print('batch_y_dates_int_cut:', batch_y_dates_int.shape)
            #print('batch_y_date dim transpos: ', np.shape(batch_y_dates)) #se lista
            #batch_y_dates = batch_y_dates[:, -args.pred_len:]
            wewe = wewe[:, -args.pred_len:]

            #if type == 'test':
            print('output dim: ', outputs.shape)
            print('batch_x: ', batch_x.shape)
            print('batch_y: ', batch_y.shape)
            #print('batch_y_dates dim cut: ', np.shape(batch_y_dates))
            print('wewe cut: ', wewe.shape)


            pred = outputs.detach() #qua adesso abbiamo i valori predetti
            true = batch_y.detach() #qua abbiamo i valori reali
            #qua potremmo salvare i valori e fare un GRAFICO da mettere su tensor
#----- AGGIUNTE
            
            #print('batch_y_marker aaaa', batch_y_mark)
            #print('batch_y_dates aaaa', batch_y_dates.shape)
            #print('batch_y_dates aaaa', len(batch_y_dates))
            
            predictions.append(pred.cpu().numpy()) #tensore vuoto, salvo il valore pred/ out loop / save in prediction
            actuals.append(true.cpu().numpy())
            #pred_last.append(pred.cpu().numpy())
            #true_last.append(true.cpu().numpy())
            
            #batch_y_dates = [d[-args.pred_len:]for d in batch_y_dates] #ok funziona ma devo salvare anche la restante parte!!!!!!!AAAAAA PROVARE A SISTEMARE, è L'ULTIMA COSA CHE MANCA ([7, 8, 9, 10])
            #print('batch_y_dates bbbbbb', batch_y_dates.shape) #
            #print('batch_y_dates bbbbbb', len(batch_y_dates))
            #all_batch_dates.append(batch_y_dates) # batch_y_date[: -args.pred_len: f_dim]
            
            #if accelerator.is_main_process:
            #    all_batch_dates.extend(wewe) #test con piu gpu non funziona

            all_batch_dates.append(wewe) #(ATTENZIONE QUESTO FUNZIONA SE USO SOLO UNA CPU)
            all_batch_dates_int.append(batch_y_dates_int.cpu().numpy())
#-----          
            loss = criterion(pred, true)

            mae_loss = mae_metric(pred, true)

            total_loss.append(loss.item())
            total_mae_loss.append(mae_loss.item())

    total_loss = np.average(total_loss)
    total_mae_loss = np.average(total_mae_loss)
#----- AGGIUNTE
    predictions = np.concatenate(predictions, axis = 0)
    actuals = np.concatenate(actuals, axis = 0)
    all_batch_dates = np.concatenate(all_batch_dates, axis = 0 )
    all_batch_dates_int = np.concatenate(all_batch_dates_int, axis=0)
    print('predictions lenght:', predictions.shape)
    print('actuals lenght:', actuals.shape)
    print('dates_from batch:', all_batch_dates.shape)
    print('dates_int_batches:', all_batch_dates_int.shape)
    #print('predictions lenght[0]:', predictions[0].shape)
    #print('actuals lenght[0]:', actuals[0].shape)
    print('dates_from batch[0]:', all_batch_dates[0].shape)


    if(vali_data.scale == True):
        print('RISCALO I DATI')
        mean,std = vali_data.get_scaler_params()
        scaler = StandardScaler(mean,std)
        predictions_norm = scaler.inverse_transform(predictions)
        #pred_last_norm = scaler.inverse_transform(pred_last)
        actuals_norm = scaler.inverse_transform(actuals)
        #true_last_norm = scaler.inverse_transform(true_last)
        print('predictions inv lenght:', predictions_norm.shape)
        print('actuals inv lenght:', actuals_norm.shape)
        #print('actuals inv: ',actuals_norm) #TESTARE
        #print('prediction inv: ', predictions_norm) #TESTARE
        #print('pred_las inv lenght:', pred_last_norm.shape) #in caso provarli a salvare nel df facendo flat!!
        #print('true_last inv lenght:', true_last_norm.shape)

    #provare a fare un df con actuals, predictions
    print('epoch +1: ', epoch+1)
    print('args.train_epochs', args.train_epochs)
    if ((epoch +1) == args.train_epochs) and (format == 'test'): #ultimo testo
        print('TRUE')
        if(vali_data.scale == True):
            np.save('predictions_norm', predictions_norm) 
            np.save('actuals_norm', actuals_norm) 
            np.save('dates', all_batch_dates) 
            np.save('dates_int',all_batch_dates_int)
        else:
            np.save('predictions', predictions) 
            np.save('actuals', actuals) 
            np.save('dates', all_batch_dates) 
            np.save('dates_int',all_batch_dates_int)
    else: 
        print('FALSE')
    
    
    actuals_flat = actuals.squeeze().reshape(-1)
    predictions_flat = predictions.squeeze().reshape(-1)
    actuals_flat_norm = actuals_norm.squeeze().reshape(-1)
    predictions_flat_norm = predictions_norm.squeeze().reshape(-1)
    #all_batch_dates_flates = #all_batch_dates.squeeze().reshape(-1)
    #print('predictions flat lenght:', len(predictions_flat))
    #print('actuals flat lenght:', len(actuals_flat))
    #print('dates_from batch flat:', #all_batch_dates)

    test_writer = SummaryWriter(log_dir=f'runs/{args.model_comment}') #open writer

    #if type == 'vali':
        #plot_vali(predictions, predictions_norm, actuals, actuals_norm, dates, epoch, args)
    #    plot_vali(predictions_flat, predictions_flat_norm, actuals_flat, actuals_flat_norm, dates['date'], epoch, args)
    #else:
        #plot_test(predictions, predictions_norm, actuals, actuals_norm, dates, epoch, args)
    #    plot_test(predictions_flat, predictions_flat_norm, actuals_flat, actuals_flat_norm, dates['date'], epoch, args)

    if format == 'vali':

        #title = f'Predictions vs Actuals Vali Epoch {epoch + 1}'
        #title_normal = f'Predictions Normal vs Actuals Normal Vali Epoch {epoch + 1}'
        #for step in range(predictions.shape[0]): #loop samples
        #    test_writer.add_scalars(title,{"Predicted":predictions[step].mean(), "Actual":actuals[step].mean()}, step) #name, dict, step
        
        #for step in range(predictions.shape[0]): #loop samples
        #    test_writer.add_scalars(title_normal,{"Predicted Normal":predictions_norm[step].mean(), "Actual Normal":actuals_norm[step].mean()}, step) #name, dict, step
        
        #or
        fig,ax = plt.subplots(figsize=(20,15))
        #ax.plot(actuals[0],label = 'Actual') #hanno una struttura del tipo 40,1,90
        #ax.plot(dates, actuals[0],label = 'Actual') #hanno una struttura del tipo 40,1,90
        ax.plot(actuals_flat, label = 'Actual')
        #ax.plot(predictions[0], label = 'Predictions', color='red')
        #ax.plot(dates, predictions[0], label = 'Predictions', color='red')
        ax.plot(predictions_flat, label = 'Predictions', color='red')
        ax.legend()
        ax.set_xlabel('Timestamp')
        ax.set_ylabel('Affluence')
        ax.set_title(f'Prediction vs Actual Vali Epoch {epoch + 1}')
        test_writer.add_figure(f"Prediction vs Actual Vali Epoch{epoch + 1} (simple plot)", fig)

        fig,ax = plt.subplots(figsize=(20,15))
        ax.plot(actuals_norm[5], label = 'Actual')
        #ax.plot(dates, actuals_norm[0], label = 'Actual')
        #ax.plot(actuals_flat_norm, label = 'Actual Normal')
        ax.plot(predictions_norm[5], label = 'Predictions', color='red')
        #ax.plot(dates, predictions_norm[0], label = 'Predictions', color='red')
        #ax.plot(predictions_flat_norm, label = 'Predictions Normal', color='red')
        ax.legend()
        ax.set_xlabel('Timestamp')
        ax.set_ylabel('Affluence')
        ax.set_title(f'Prediction vs Actual NORMAL Vali Epoch{epoch + 1}')
        test_writer.add_figure(f"Prediction Normal vs Actual Normal Vali Epoch{epoch + 1} (simple plot)", fig)

    if format == 'test':

        #title = f'Predictions vs Actuals Test Epoch {epoch + 1}'
        #title_normal = f'Predictions Normal vs Actuals Normal Test Epoch {epoch + 1}'
        #for step in range(predictions.shape[0]): #loop samples
        #    test_writer.add_scalars(title,{"Predicted":predictions[step].mean(), "Actual":actuals[step].mean()}, step) #name, dict, step
        #
        #for step in range(predictions.shape[0]): #loop samples
        #    test_writer.add_scalars(title_normal,{"Predicted Normal":predictions_norm[step].mean(), "Actual Normal":actuals_norm[step].mean()}, step) #name, dict, step
        print('all_batch_dates[0]: ',all_batch_dates[0])
        print('actuals[0]:', actuals[0])
        #or
        fig,ax = plt.subplots(figsize=(30,20))
        ax.plot(all_batch_dates[0].flatten(), actuals[0].flatten(), label = 'Actual') #hanno una struttura del tipo 40,1,90
        #ax.plot(#all_batch_dates_flates, actuals_flat, label = 'Actual') #hanno una struttura del tipo 40,1,90
        #ax.plot(actuals[0], label = 'Actual')
        #ax.plot(all_batch_dates_flates, predictions_flat, label = 'Predictions', color='red')
        ax.plot(all_batch_dates[0].flatten(), predictions[0].flatten(), label = 'Predictions', color='red')
        #ax.plot(predictions[0], label = 'Predictions', color='red')
        ax.legend()
        ax.set_xlabel('Timestamp')
        ax.set_ylabel('Affluence')
        #ax.set_xticklabels(all_batch_dates[0], rotation = 90)
        ax.set_xticklabels(ax.get_xticklabels(), rotation = 90)
        ax.set_title(f'Prediction vs Actual Test Epoch {epoch + 1} blocco 0')
        test_writer.add_figure(f"PvsA Test Epoch{epoch + 1} (simple plot) blocco 0", fig)

        fig,ax = plt.subplots(figsize=(30,20))
        ax.plot(all_batch_dates[1].flatten(), actuals[1].flatten(), label = 'Actual') #hanno una struttura del tipo 40,1,90
        #ax.plot(#all_batch_dates_flates, actuals_flat, label = 'Actual') #hanno una struttura del tipo 40,1,90
        #ax.plot(actuals[1], label = 'Actual')
        #ax.plot(all_batch_dates_flates, predictions_flat, label = 'Predictions', color='red')
        ax.plot(all_batch_dates[1].flatten(), predictions[1].flatten(), label = 'Predictions', color='red')
        #ax.plot(predictions[1], label = 'Predictions', color='red')
        ax.legend()
        ax.set_xlabel('Timestamp')
        ax.set_ylabel('Affluence')
        #ax.set_xticklabels(all_batch_dates[1], rotation = 90)
        ax.set_xticklabels(ax.get_xticklabels(), rotation = 90)
        ax.set_title(f'Prediction vs Actual Test Epoch {epoch + 1} blocco 1')
        test_writer.add_figure(f"PvsA Test Epoch{epoch + 1} (simple plot) blocco 1", fig)


        fig,ax = plt.subplots(figsize=(30,20))
        ax.plot(all_batch_dates[0].flatten(), actuals_norm[0].flatten(), label = 'Actual')
        #ax.plot(#all_batch_dates_flates, actuals_flat_norm, label = 'Actual')
        #ax.plot(actuals_norm[0], label = 'Actual Normal')
        #ax.plot(#all_batch_dates_flates, predictions_flat_norm, label = 'Predictions', color='red')
        ax.plot(all_batch_dates[0].flatten(), predictions_norm[0].flatten(), label = 'Predictions', color='red')
        #ax.plot(predictions_norm[0], label = 'Predictions Normal', color='red')
        #ax.set_xticks(all_batch_dates_flates)
        ax.set_xticklabels(ax.get_xticklabels(), rotation = 90)
        #ax.set_xticklabels(all_batch_dates[0], rotation = 90)
        ax.legend()
        ax.set_xlabel('Timestamp')
        ax.set_ylabel('Affluence')
        ax.set_title(f'Prediction vs Actual NORMAL Test Epoch{epoch + 1} blocco 0')
        test_writer.add_figure(f"PN vs AN Test Epoch{epoch + 1} (simple plot) blocco 0", fig)

        fig,ax = plt.subplots(figsize=(30,20))
        ax.plot(all_batch_dates[1].flatten(), actuals_norm[1].flatten(), label = 'Actual')
        #ax.plot(#all_batch_dates_flates, actuals_flat_norm, label = 'Actual')
        #ax.plot(actuals_norm[1], label = 'Actual Normal')
        #ax.plot(#all_batch_dates_flates, predictions_flat_norm, label = 'Predictions', color='red')
        ax.plot(all_batch_dates[1].flatten(), predictions_norm[1].flatten(), label = 'Predictions', color='red')
        #ax.plot(predictions_norm[1], label = 'Predictions Normal', color='red')
        #ax.set_xticks(all_batch_dates_flates)
        ax.set_xticklabels(ax.get_xticklabels(), rotation = 90)
        ax.legend()
        #ax.set_xticklabels(all_batch_dates[1], rotation = 90)
        ax.set_xlabel('Timestamp')
        ax.set_ylabel('Affluence')
        ax.set_title(f'Prediction vs Actual NORMAL Test Epoch{epoch + 1} blocco 1')
        test_writer.add_figure(f"PN vs AN Test Epoch{epoch + 1} (simple plot) blocco 1", fig)


    test_writer.close() #close writer
#-----

    model.train()
    return total_loss, total_mae_loss

#----- AGGIUNTE
def plot_vali(predictions, predictions_norm, actuals, actuals_norm, dates, epoch, args):
        
        test_writer = SummaryWriter(log_dir=f'runs/{args.model_comment}') #open writer

        title = f'Predictions vs Actuals Vali Epoch {epoch + 1}'
        title_normal = f'Predictions Normal vs Actuals Normal Vali Epoch {epoch + 1}'
        #for step in range(predictions.shape[0]): #loop samples
        #    test_writer.add_scalars(title,{"Predicted":predictions[step].mean(), "Actual":actuals[step].mean()}, step) #name, dict, step
        
        #for step in range(predictions.shape[0]): #loop samples
        #    test_writer.add_scalars(title_normal,{"Predicted Normal":predictions_norm[step].mean(), "Actual Normal":actuals_norm[step].mean()}, step) #name, dict, step
        
        #or
        fig,ax = plt.subplots(figsize=(10,5))
        #ax.plot(dates, actuals[0],label = 'Actual') #hanno una struttura del tipo 40,1,90
        ax.plot(dates, actuals,label = 'Actual') #hanno una struttura del tipo 40,1,90
        #ax.plot(dates, predictions[0], label = 'Predictions', color='red')
        ax.plot(dates, predictions, label = 'Predictions', color='red')
        ax.legend()
        ax.set_xlabel('Timestamp')
        ax.set_ylabel('Affluence')
        ax.set_xticklabels(ax.get_xticklabels(), rotation = 90)
        ax.set_title(f'Prediction vs Actual Vali Epoch {epoch + 1}')
        test_writer.add_figure(f"Prediction vs Actual Vali Epoch{epoch + 1} (simple plot)", fig)

        fig,ax = plt.subplots(figsize=(10,5))
        #ax.plot(dates, actuals_norm[0], label = 'Actual')
        ax.plot(dates, actuals_norm, label = 'Actual')
        #ax.plot(dates, predictions_norm[0], label = 'Predictions', color='red')
        ax.plot(dates, predictions_norm, label = 'Predictions', color='red')
        ax.legend()
        ax.set_xlabel('Timestamp')
        ax.set_ylabel('Affluence')
        ax.set_xticklabels(ax.get_xticklabels(), rotation = 90)
        ax.set_title(f'Prediction vs Actual NORMAL Vali Epoch{epoch + 1}')
        test_writer.add_figure(f"Prediction Normal vs Actual Normal Vali Epoch{epoch + 1} (simple plot)", fig)

        test_writer.close() #close writer

def plot_test(predictions, predictions_norm, actuals, actuals_norm, dates, epoch, args):
        
        test_writer = SummaryWriter(log_dir=f'runs/{args.model_comment}') #open writer

        title = f'Predictions vs Actuals Test Epoch {epoch + 1}'
        title_normal = f'Predictions Normal vs Actuals Normal Test Epoch {epoch + 1}'
        #for step in range(predictions.shape[0]): #loop samples
        #    test_writer.add_scalars(title,{"Predicted":predictions[step].mean(), "Actual":actuals[step].mean()}, step) #name, dict, step
        
        #for step in range(predictions.shape[0]): #loop samples
        #    test_writer.add_scalars(title_normal,{"Predicted Normal":predictions_norm[step].mean(), "Actual Normal":actuals_norm[step].mean()}, step) #name, dict, step
        
        #or
        fig,ax = plt.subplots(figsize=(10,5))
        ax.plot(dates[0:args.batch_size], actuals[0], label = 'Actual') #hanno una struttura del tipo 40,1,90
        #ax.plot(actuals, label = 'Actual')
        ax.plot(dates[0:args.batch_size], predictions[0], label = 'Predictions', color='red')
        #ax.plot(predictions, label = 'Predictions', color='red')
        ax.legend()
        ax.set_xlabel('Timestamp')
        ax.set_ylabel('Affluence')
        ax.set_xticklabels(ax.get_xticklabels(), rotation = 90)
        ax.set_title(f'Prediction vs Actual Test Epoch {epoch + 1}')
        test_writer.add_figure(f"Prediction vs Actual Test Epoch{epoch + 1} (simple plot)", fig)

        fig,ax = plt.subplots(figsize=(10,5))
        ax.plot(dates[0:args.batch_size], actuals_norm[0], label = 'Actual')
        #ax.plot(actuals, label = 'Actual Normal')
        ax.plot(dates[0:args.batch_size], predictions_norm[0], label = 'Predictions', color='red')
        #ax.plot(predictions, label = 'Predictions Normal', color='red')
        ax.legend()
        ax.set_xlabel('Timestamp')
        ax.set_ylabel('Affluence')
        ax.set_xticklabels(ax.get_xticklabels(), rotation = 90)
        ax.set_title(f'Prediction vs Actual NORMAL Test Epoch{epoch + 1}')
        test_writer.add_figure(f"Prediction Normal vs Actual Normal Test Epoch{epoch + 1} (simple plot)", fig)

        test_writer.close() #close writer

def get_batch_dates(batch_y_mark, freq): #test to return real data from the batch (during infrernze) #sbagliato
        #from dataset i assume batch_y_mark has time featurees in order [year, month, day, hour, min]

        if isinstance(batch_y_mark, torch.Tensor):
            batch_y_mark = batch_y_mark.cpu().numpy()
        
        timestamps = []

        if freq == 'h':
            print('ora')
            for row in batch_y_mark:
                for time_point in row:
                    year, month, day, hour= [int(x) for x in time_point[:4]]
                    timestamps.append(pd.Timestamp(year=year, month=month, day=day, hour=hour))
        if freq == '15min':
            print('min')
            for row in batch_y_mark:
                for time_point in row:
                    year, month, day, hour, minute = [int(x) for x in time_point[:5]]
                    timestamps.append(pd.Timestamp(year=year, month=month, day=day, hour=hour, minute=minute))    
        return timestamps
#----- AGGIUNTE

def test(args, accelerator, model, train_loader, vali_loader, criterion):
    x, _ = train_loader.dataset.last_insample_window()
    print("X TEST:", x)
    y = vali_loader.dataset.timeseries
    print("Y Test:", y)
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
    
#----- AGGIUNTE
        predictions = (pred.cpu().numpy()) 
        actuals = (true.cpu().numpy()) 
        #print('PREDICTION: ',predictions)
        #print('ACTUALS: ',actuals)
#-----
#----- AGGIUNTE
        mean,std = y.get_scaler_params()
        scaler = StandardScaler(mean,std)
        predictions_norm = scaler.inverse_transform(predictions)
        actuals_norm = scaler.inverse_transform(actuals)
        
        test_writer = SummaryWriter(log_dir=f'runs/{args.model_comment}') #open writer
        title = f'Predictions vs Actuals Test Epoch'
        title_normal = f'Predictions Normal vs Actuals Normal Test Epoch'
        for step in range(predictions.shape[0]): #loop samples
            test_writer.add_scalars(title,{"Predicted":predictions[step].mean(), "Actual":actuals[step].mean()}, step) #name, dict, step
        
        for step in range(predictions.shape[0]): #loop samples
            test_writer.add_scalars(title_normal,{"Predicted Normal":predictions_norm[step].mean(), "Actual Normal":actuals_norm[step].mean()}, step) #name, dict, step
        
        #or
        fig,ax = plt.subplots(figsize=(10,5))
        ax.plot(actuals[0], label = 'Actual') #hanno una struttura del tipo 40,1,90
        #ax.plot(actuals, label = 'Actual')
        ax.plot(predictions[0], label = 'Predictions', color='red')
        #ax.plot(predictions, label = 'Predictions', color='red')
        ax.legend()
        ax.set_title('Prediction vs Actual Epoch Test')
        test_writer.add_figure(f"Prediction vs Actual Test (simple plot)", fig)

        fig,ax = plt.subplots(figsize=(10,5))
        ax.plot(actuals_norm[0], label = 'Actual')
        #ax.plot(actuals, label = 'Actual Normal')
        ax.plot(predictions_norm[0], label = 'Predictions', color='red')
        #ax.plot(predictions, label = 'Predictions Normal', color='red')
        ax.legend()
        ax.set_title(f'Prediction vs Actual NORMAL Test')
        test_writer.add_figure(f"Prediction Normal vs Actual Normal Test (simple plot)", fig)

        test_writer.close() #close writer
#-----


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