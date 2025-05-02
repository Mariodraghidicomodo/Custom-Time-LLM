import argparse
import torch
from accelerate import Accelerator, DeepSpeedPlugin #Within the PyTorch repo, we define an “Accelerator” as a torch.device that is being used alongside a CPU to speed up computation.
from accelerate import DistributedDataParallelKwargs #powerful module in PyTorch that allows you to parallelize your model across multiple machines, making it perfect for large-scale deep learning applications. To use DDP, you’ll need to spawn multiple processes and create a single instance of DDP per process.
from torch import nn, optim
from torch.optim import lr_scheduler
from tqdm import tqdm

from models import Autoformer, DLinear, TimeLLM

from data_provider.data_factory import data_provider
import time
import random
import numpy as np
import os

#----- AGGIUNTE
import pickle
import json
#from sklearn.preprocessing import StandardScaler
#-----

#os.environ in Python is a mapping object that represents the user’s OS environmental variables. It returns a dictionary having the user’s environmental variable as key and their values as value.
#praticamente aggiunge queste variabili d'ambiente
os.environ['CURL_CA_BUNDLE'] = '' #capire cosa servono
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:64" #capire pure quanto

from utils.tools import del_files, EarlyStopping, adjust_learning_rate, vali, load_content, test

parser = argparse.ArgumentParser(description='Time-LLM')

fix_seed = 2021
random.seed(fix_seed)
torch.manual_seed(fix_seed)
np.random.seed(fix_seed) 

# basic config
parser.add_argument('--task_name', type=str, required=True, default='long_term_forecast',
                    help='task name, options:[long_term_forecast, short_term_forecast, imputation, classification, anomaly_detection]')
parser.add_argument('--is_training', type=int, required=True, default=1, help='status')
parser.add_argument('--model_id', type=str, required=True, default='test', help='model id')
parser.add_argument('--model_comment', type=str, required=True, default='none', help='prefix when saving test results')
parser.add_argument('--model', type=str, required=True, default='Autoformer',
                    help='model name, options: [Autoformer, DLinear]')
parser.add_argument('--seed', type=int, default=2021, help='random seed')

# data loader
parser.add_argument('--data', type=str, required=True, default='ETTm1', help='dataset type') #nome del file da prendere dentro a data factory nel dizionario (data_dict) -> aggiornare data_dict
parser.add_argument('--root_path', type=str, default='./dataset', help='root path of the data file') #doev è salvato il dataset reale (\poi_crowding)
parser.add_argument('--data_path', type=str, default='ETTh1.csv', help='data file') #nome reale del data_set (export_42.csv)
parser.add_argument('--features', type=str, default='M',
                    help='forecasting task, options:[M, S, MS]; '
                         'M:multivariate predict multivariate, S: univariate predict univariate, '
                         'MS:multivariate predict univariate') #nel nostro caso abbiamo multipli attributi da usare come ts e un unico output (presenze) (quindi direi che è MS) (da testare)
parser.add_argument('--target', type=str, default='OT', help='target feature in S or MS task') #target mettere s (un target, un output)
parser.add_argument('--loader', type=str, default='modal', help='dataset type') #???????
parser.add_argument('--freq', type=str, default='h',
                    help='freq for time features encoding, '
                         'options:[s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly], '
                         'you can also use more detailed freq like 15min or 3h') #perfetto -> indica ogni quanto vengono salvati i dati (il nostro caso mi sembra particolare in quanto abbimao i dati salvati solo per tre ore ogni giorno!!!!!) (TESTARE)
parser.add_argument('--checkpoints', type=str, default='./checkpoints/', help='location of model checkpoints') #??? ricontrollare!!
parser.add_argument('--scale', type=bool, default=True, help='Scale the values') #AGGIUNTO

# forecasting task
parser.add_argument('--seq_len', type=int, default=96, help='input sequence length')
parser.add_argument('--label_len', type=int, default=48, help='start token length')
parser.add_argument('--pred_len', type=int, default=96, help='prediction sequence length')
parser.add_argument('--seasonal_patterns', type=str, default='Monthly', help='subset for M4')

# model define
parser.add_argument('--enc_in', type=int, default=7, help='encoder input size') #Number of input features for the encoder (7 time-series variables in this case).
parser.add_argument('--dec_in', type=int, default=7, help='decoder input size') #Number of input features for the decoder (also 7).
parser.add_argument('--c_out', type=int, default=7, help='output size') #Output dimension (7, matching input features).
parser.add_argument('--d_model', type=int, default=16, help='dimension of model') #Model embedding dimension (16). Defines the hidden size of the transformer layers.
parser.add_argument('--n_heads', type=int, default=8, help='num of heads') #Number of attention heads in multi-head self-attention (8).
parser.add_argument('--e_layers', type=int, default=2, help='num of encoder layers') #Number of layers in the encoder (2).
parser.add_argument('--d_layers', type=int, default=1, help='num of decoder layers') #Number of layers in the decoder (1).
parser.add_argument('--d_ff', type=int, default=32, help='dimension of fcn') #Dimension of the feedforward network in the transformer (32).
parser.add_argument('--moving_avg', type=int, default=25, help='window size of moving average') #Window size for moving average (25), likely used for smoothing the input time series.
parser.add_argument('--factor', type=int, default=1, help='attn factor') #Attention scaling factor (1), might control attention computation.
parser.add_argument('--dropout', type=float, default=0.1, help='dropout') #Dropout rate (0.1) to prevent overfitting.
parser.add_argument('--embed', type=str, default='timeF',
                    help='time features encoding, options:[timeF, fixed, learned]') #Type of embedding for time features (timeF: time feature encoding).
parser.add_argument('--activation', type=str, default='gelu', help='activation') #che tipo di activation function usare (gelu: Gaussian error linear unit (GELU) activation function.)
parser.add_argument('--output_attention', action='store_true', help='whether to output attention in encoder') #If set, the model outputs attention weights.
parser.add_argument('--patch_len', type=int, default=16, help='patch length') #Patch size (16), possibly for a patch-based transformer (similar to ViTs).
parser.add_argument('--stride', type=int, default=8, help='stride') #Stride for patch extraction (8)
parser.add_argument('--prompt_domain', type=int, default=0, help='') #BOH???
parser.add_argument('--llm_model', type=str, default='LLAMA', help='LLM model') # LLAMA, GPT2, BERT
parser.add_argument('--llm_dim', type=int, default='4096', help='LLM model dimension')# LLama7b:4096; GPT2-small:768; BERT-base:768


# optimization  #verranno inseriti nel dataloader (in caso vedere dataloader.py)
parser.add_argument('--num_workers', type=int, default=10, help='data loader num workers') #how many subprocesses to use for data loading. ``0`` means that the data will be loaded in the main process.
parser.add_argument('--itr', type=int, default=1, help='experiments times') #Number of experimental runs
parser.add_argument('--train_epochs', type=int, default=10, help='train epochs') #Number of experimental runs
parser.add_argument('--align_epochs', type=int, default=10, help='alignment epochs') #Alignment epochs, possibly for fine-tuning
parser.add_argument('--batch_size', type=int, default=32, help='batch size of train input data') #how many samples per batch to load (default: ``1``).
parser.add_argument('--eval_batch_size', type=int, default=8, help='batch size of model evaluation') #Evaluation batch size
parser.add_argument('--patience', type=int, default=10, help='early stopping patience') 
parser.add_argument('--learning_rate', type=float, default=0.0001, help='optimizer learning rate')
parser.add_argument('--des', type=str, default='test', help='exp description')
parser.add_argument('--loss', type=str, default='MSE', help='loss function')
parser.add_argument('--lradj', type=str, default='type1', help='adjust learning rate')
parser.add_argument('--pct_start', type=float, default=0.2, help='pct_start') #Percentage of the training schedule where the LR increases (0.2).
parser.add_argument('--use_amp', action='store_true', help='use automatic mixed precision training', default=False)
parser.add_argument('--llm_layers', type=int, default=6) #Number of layers used from the LLM
parser.add_argument('--percent', type=int, default=100)

args = parser.parse_args()
#----- AGGIUNTE
#args_dict = vars(args)
#with open("args.json", "w") as f:
#    json.dump(args_dict, f, indent=4) #save args as json
#-----
#----- AGGIUGO CREAZIONE TENSORBOARD
from torch.utils.tensorboard import SummaryWriter
test_writer = SummaryWriter(log_dir=f'runs/{args.model_comment}')

#import sys
#sys.exit()
#-----
ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True) #PERCHè RITORNA QUA DOPO ESSERE ARRIVATO AL LOOP DEI BATCH?????? (sembra una cosa collegate con il numero di num workers)
deepspeed_plugin = DeepSpeedPlugin(hf_ds_config='./ds_config_zero2.json') #ok se uso il multiprocessing devo capire come lavora deepspeed!!
accelerator = Accelerator(kwargs_handlers=[ddp_kwargs], deepspeed_plugin=deepspeed_plugin) #OVVIAMENTE DA ERRORE PERCHè ESSENGO GIA STATO CREATO ACCELELRATE TI DICE CHE LA PORTA è GIà USATA
#alla fine crea più processi per velocizzare tutto (devere anche script per args dell accelerator)
for ii in range(args.itr):
    # setting record of experiments
    setting = '{}_{}_{}_{}_ft{}_sl{}_ll{}_pl{}_dm{}_nh{}_el{}_dl{}_df{}_fc{}_eb{}_{}_{}'.format(
    #setting = '{}_{}_{}'.format(  #test probelem name path to loong
        args.task_name,
        args.model_id,
        args.model,
        args.data,
        args.features,
        args.seq_len,
        args.label_len,
        args.pred_len,
        args.d_model,
        args.n_heads,
        args.e_layers,
        args.d_layers,
        args.d_ff,
        args.factor,
        args.embed,
        args.des, ii)

    #train data sono solo i dati e in che modo li voglio (indico i feature e i target)
    #train loder caric i dati di train_data per essere usato da tensor indicando batch se fare shuffle e altre cose
    train_data, train_loader = data_provider(args, 'train') #usato per creare le parti del data set
    
#----- AGGIUNTE (return scaler)
    #print('raw_data: ', train_data.data_x[:5]) #    PERCHè DATA_X E DATA_Y SONO UGUALI??
    #print('raw_target', train_data.data_y[:5])
    #print('scaler: ', train_data.scaler.mean_) #la media è per ogni colonna
    #mean, std = train_data.get_scaler_params() #la std è per ogni colonna (quinidi è una lista, dove ogni valori è la media di una colonna)
    #print("Mean:", mean)
    #print("Std:", std)
    #print('unscaled_data = ',train_data.inverse_transform(train_data.data_x))
    #print('unscaled_target = ', train_data.inverse_transform(train_data.data_y))
    #print('data_stamp: ',train_data.data_stamp[:5])
    #date = train_data.get_date_strings()
    #print('date: ', date[:5])
    #print('date type:', type(date))
    #print('lenght date', 'date'])) #dimostriamo che le date e i dati hanno lunghezza uguale quinid cosa succede quando facciamo i batch? perhcè non hanno lunghezza uguale in toools function vali??
    #print('lenght date_x', len(train_data.data_x))
    #print('columns:', date.columns)
    #import sys
    #sys.exit()

#-----   
    vali_data, vali_loader = data_provider(args, 'val')
    
#----- AGGIUNTE
    #print('vali_data:',train_data)
    #print('vali_loader:',train_loader)
#-----    
    
    test_data, test_loader = data_provider(args, 'test') #create test
    
#----- AGGIUNTE
    #salvo train test e val
    #print('test date lenght:', len(test_data.data_x))
    #torch.save(train_data,"train_data.pt")
    #torch.save(vali_data,"vali_data.pt")
    #torch.save(test_data,"test_data.pt")
    #salvo anche i loader cosi non devo ricrearli
    #with open("train_loader.pkl", "wb") as f:
    #    pickle.dump(train_loader, f)
    #with open("vali_loader.pkl", "wb") as f:
    #    pickle.dump(vali_loader, f)
    #with open("test_loader.pkl", "wb") as f:
    #    pickle.dump(test_loader, f)
#-----
    if args.model == 'Autoformer': #creazione del modello
        model = Autoformer.Model(args).float()
    elif args.model == 'DLinear':
        model = DLinear.Model(args).float()
    else:
        model = TimeLLM.Model(args).float()

    path = os.path.join(args.checkpoints,
                        setting + '-' + args.model_comment)  # unique checkpoint saving path
    args.content = load_content(args)
    if not os.path.exists(path) and accelerator.is_local_main_process: #dove mi salva il modello??
        os.makedirs(path) #create the directory where save the model if it doesn't exist!!! (path)

    time_now = time.time()

    train_steps = len(train_loader)
    early_stopping = EarlyStopping(accelerator=accelerator, patience=args.patience)

    trained_parameters = [] #parametri dove allenare il modello (credo, dal debugger mi fa vedere che sono solo valori) (praticamente salva ogni linea del dataset?? NON CREDO)
    for p in model.parameters():
        if p.requires_grad is True: #sta usando solo la cpu??
            trained_parameters.append(p)

    model_optim = optim.Adam(trained_parameters, lr=args.learning_rate)

    if args.lradj == 'COS':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(model_optim, T_max=20, eta_min=1e-8)
    else:
        scheduler = lr_scheduler.OneCycleLR(optimizer=model_optim,
                                            steps_per_epoch=train_steps,
                                            pct_start=args.pct_start,
                                            epochs=args.train_epochs,
                                            max_lr=args.learning_rate)

    criterion = nn.MSELoss() #scelta dei criteri di accuratezza
    mae_metric = nn.L1Loss()
#accelerator come device mi da cuda quindi dovrebbe usare la cpu (attenzione no module mpi4py, lo installo(fatto))
#sembra non caricare MPI library, mi dice di usare il percorso completo con constructor syntax
#da libearia The Windows wheels available on PyPI are specially crafted to work with either the Intel MPI or the Microsoft MPI runtime, therefore requiring a separate installation of any one of these packages.
#Intel MPI is under active development and supports recent version of the MPI standard. Intel MPI can be installed with pip (see the impi-rt package on PyPI), being therefore straightforward to get it up and running within a Python environment. Intel MPI can also be installed system-wide as part of the Intel HPC Toolkit for Windows or via standalone online/offline installers.
#(installado anche microsoft mpi + path)
#FUNZIONA!! (RISULTATO VEDERE NOTE)
    train_loader, vali_loader, test_loader, model, model_optim, scheduler = accelerator.prepare(
        train_loader, vali_loader, test_loader, model, model_optim, scheduler)
#master port 29500
    if args.use_amp:
        scaler = torch.cuda.amp.GradScaler() #uso di cuda quindi uso della mia GPU?

#-----AGGIUNTE
    #test_predictions = []
    running_loss = 0.0
#-----
    for epoch in range(args.train_epochs): #loop for each epochs
        iter_count = 0 #imposto train e iter uguali a zero; a ogni epoch si azzerano
        train_loss = []

        model.train() #set model to trining mode
        epoch_time = time.time() #partenza tempo epochs (tenere traccia di quanto ci dta mettendo) #AGGIUNTO BATCH_Y_DATES
        for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in tqdm(enumerate(train_loader)):
        #for i, (batch_x, batch_y, batch_x_mark, batch_y_mark, batch_y_dates) in tqdm(enumerate(train_loader)): #loop in ogni batch (contiene un numero di input e output uguale a quelo inserito in args) #ATTENZIONE ARRIVO FINO A QUA CON IL DEBUGGER POI SI BUGGA E MI RITORNA ALLA RIGA 106!!!! ERRORE CON LA MASTER PORT (batch y sono i 'labels', batch_x sarebbe images (ex:pytorch))
        #for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):
            #tdqm deorazione barra progreso / al posto di avere batch_x, batch_y, batch_x_mark, batch_y_mark = data, sono inseriti ne loop??? 
            # batch_x = ts values that will use as input (shape: batch_size, seq_len, num_features) (numerical values)
            # batch_x_mark = temporal feature day, week, hour etc... (data, timestamp), shape(batch_size, seq_len, num_time_features) (temporal context)
            iter_count += 1
            model_optim.zero_grad()
            #nuovo errore con accelerator vedere note; sposta i dati sulla gpu, se il dataset è troppo grande oppure le batch sono troppo grande la gpu arriva al masssimo e il programma crasha
            batch_x = batch_x.float().to(accelerator.device)
            batch_y = batch_y.float().to(accelerator.device)
            batch_x_mark = batch_x_mark.float().to(accelerator.device) #mark inidica i label/target??
            batch_y_mark = batch_y_mark.float().to(accelerator.device)

            # decoder input
            dec_inp = torch.zeros_like(batch_y[:, -args.pred_len:, :]).float().to(
                accelerator.device)
            dec_inp = torch.cat([batch_y[:, :args.label_len, :], dec_inp], dim=1).float().to(
                accelerator.device)

            # encoder - decoder
            if args.use_amp: #CAPIRE QUANDO SI ATTIVA
                with torch.cuda.amp.autocast(): #arriva qua
                    if args.output_attention:
                        outputs = model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0] #predizione dei valori??
                    else:
                        outputs = model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                    f_dim = -1 if args.features == 'MS' else 0
                    outputs = outputs[:, -args.pred_len:, f_dim:]
                    batch_y = batch_y[:, -args.pred_len:, f_dim:].to(accelerator.device)
                    loss = criterion(outputs, batch_y) #OK PERFETTO, QUA CALCOLA LA LOSS A OGNI ITERAZIONE -> QUI INSERIRE IL CONTROLLO SULLE ITERAZIONI E SALVARE IL PLOT DELLA LOSS!!
                    train_loss.append(loss.item())
#----- AGGIUNTE
                    running_loss += loss.item()
                    if i % 10 == 9: #ogni 10 iter dovrebbe salvare
                      #log the running loss
                      print('running_loss = ', running_loss)
                      test_writer.add_scalar('training loss', running_loss / 10, epoch * len(train_loader) + i)
                      running_loss = 0.0
#-----
            else:
                if args.output_attention:
                    outputs = model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]  #model.forward(), costruisco il modello con una parte dei dati (più o meno) forward
                else:
                    outputs = model(batch_x, batch_x_mark, dec_inp, batch_y_mark) #forward

                f_dim = -1 if args.features == 'MS' else 0
                #print('batch_y: ', batch_y.shape)
                #print('output dim:', outputs.shape)
                outputs = outputs[:, -args.pred_len:, f_dim:]
                batch_y = batch_y[:, -args.pred_len:, f_dim:]
                #print('output dim:', outputs.shape)
                #print('batch_y: ', batch_y.shape)
                loss = criterion(outputs, batch_y) #OK PERFETTO, QUA CALCOLA LA LOSS A OGNI ITERAZIONE -> QUI INSERIRE IL CONTROLLO SULLE ITERAZIONI E SALVARE IL PLOT DELLA LOSS!!
                train_loss.append(loss.item()) #OK PERFETTO DEVO SALVARE QUESTA VARIABILE (EX PYTORCH == running_loss)
                #inserire if sull iter -> inseire nel tensorboard add_scalar (plot della loss)
#----- AGGIUNTE
                running_loss += loss.item()
                if i % 10 == 9: #ogni 10 iter dovrebbe salvare
                  #log the running loss
                  print('running_loss = ', running_loss)
                  test_writer.add_scalar('training loss', running_loss / 10, epoch * len(train_loader) + i)
                  running_loss = 0.0
#-----

            if (i + 1) % 100 == 0: #quando arriva all unltimo batch satmpa i dati
                accelerator.print(
                    "\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                speed = (time.time() - time_now) / iter_count
                left_time = speed * ((args.train_epochs - epoch) * train_steps - i)
                accelerator.print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                iter_count = 0
                time_now = time.time()

            if args.use_amp:
                scaler.scale(loss).backward()
                scaler.step(model_optim)
                scaler.update()
            else:
                accelerator.backward(loss) 
                model_optim.step() #Update model weights

            if args.lradj == 'TST':
                adjust_learning_rate(accelerator, model_optim, scheduler, epoch + 1, args, printout=False)
                scheduler.step()
            #AGGIUNTA
            #test_predictions.append(outputs.detach().cpu().numpy()) #siamo sicuri che non sia GPU??
        
        #DIVERSO DAI NORMALI ESEMPI DOVE ALLA FINE DI TUTTE LE EPOCH FANNO IL MODEL EVAL SUL TEST 
        #QUA MI FA SUBITO UN EVAL() (FA PREDIZIONI SUL TEST!!!) (PRATICAMENTE AD OGNI EPOCH FA UNAM PREDIZIONI SUL TEST PER VEDERE SE CE EFFETTIVAMENTE UN MIGLIORAMENTO ANCHE SUL TEST!!!) 
#----- AGGIUNTE
        #test_writer.close()
#-----
        accelerator.print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
        train_loss = np.average(train_loss) #print average loss
        vali_loss, vali_mae_loss = vali(args, accelerator, model, vali_data, vali_loader, criterion, mae_metric, epoch, 'vali') #ricontrollare questa funzione!!!! ATTENZIONE: qua dentro fa model.eval() QUINDI QUA DOVREI RITORNARE I RISULTATI FINALI QUDO ESEGUIO L'ULTIMA EPOCH!!!!!!!
        test_loss, test_mae_loss = vali(args, accelerator, model, test_data, test_loader, criterion, mae_metric, epoch, 'test') #ATTENZIONE QUA CHIAMA VALI E NON TEST (test -> funzione usata per il test, controllare che sia uguale a test!!!) !!!!!!!!!
        #AGGIUNTE, QUA INSERIRE IL PLOT DEI RISULTAI DEGLI OUTPUT!!!! (DECIDERE SE FARLO PER TUTTI GLI EPOCH PER VEDERE UN MIGLIORAMENTO OPPURE SOLO IL PROBL SULL ULTIMO EPOCH (QUNIDI A TRAIN COMPLETO))
        accelerator.print(
            "Epoch: {0} | Train Loss: {1:.7f} Vali Loss: {2:.7f} Test Loss: {3:.7f} MAE Loss: {4:.7f}".format(
                epoch + 1, train_loss, vali_loss, test_loss, test_mae_loss))

        early_stopping(vali_loss, model, path) #stop trining if validation loss doesn't improve after a few epochs
        if early_stopping.early_stop:
            accelerator.print("Early stopping")
            break

        if args.lradj != 'TST':
            if args.lradj == 'COS':
                scheduler.step()
                accelerator.print("lr = {:.10f}".format(model_optim.param_groups[0]['lr']))
            else:
                if epoch == 0:
                    args.learning_rate = model_optim.param_groups[0]['lr']
                    accelerator.print("lr = {:.10f}".format(model_optim.param_groups[0]['lr']))
                adjust_learning_rate(accelerator, model_optim, scheduler, epoch + 1, args, printout=True)

        else:
            accelerator.print('Updating learning rate to {}'.format(scheduler.get_last_lr()[0]))

#----- AGGIUNTE
test_writer.close()

#final_loss = test(args, accelerator, model, train_loader, test_loader, criterion) #ATTENZIONE QUA CHIAMA VALI E NON TEST (test -> funzione usata per il test, controllare che sia uguale a test!!!) !!!!!!!!!

#AGGIUNTE
#salvo il modello da utilizzare in futuro
#torch.save(model.state_dict(), '/content/drive/MyDrive/Custom-Time-LLM-copia/model_test_small_small_dict.pth')
#torch.save(model, '/content/drive/MyDrive/Custom-Time-LLM-copia/model_test_small_small_all.pth')

      
accelerator.wait_for_everyone() 
#unwrapped_model = accelerator.unwrap_model(model)
#torch.save(unwrapped_model.state_dict(), 'test_model.pth')

#model_pred = TimeLLM.Model(args).float()
#model_pred.load_state_dict(torch.load('test_model.pth', weights_only=True))
#print('load model!')
#model.to(device) #print the model!!

#-----

if accelerator.is_local_main_process:
    path = './checkpoints'  # unique checkpoint saving path
    del_files(path)  # delete checkpoint files
    accelerator.print('success delete checkpoints')