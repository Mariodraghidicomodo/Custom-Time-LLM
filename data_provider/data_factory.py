from data_provider.data_loader import Dataset_ETT_hour, Dataset_ETT_minute, Dataset_Custom, Dataset_M4
from torch.utils.data import DataLoader

data_dict = {
    'ETTh1': Dataset_ETT_hour, #classi
    'ETTh2': Dataset_ETT_hour,
    'ETTm1': Dataset_ETT_minute,
    'ETTm2': Dataset_ETT_minute,
    'ECL': Dataset_Custom,
    'Traffic': Dataset_Custom,
    'Weather': Dataset_Custom,
    'm4': Dataset_M4,
    'Casa_Giulietta' : Dataset_Custom,
    'Casa_Giulietta_2018': Dataset_Custom,
    'Arena_Verona_2019': Dataset_Custom,
    'Arena_Verona_2019_mini': Dataset_Custom, #prompt decrise
    'Arena_Verona_2019_mini2': Dataset_Custom
    #'export_42':Dataset_Custom # capire se devo usare dataset_custom oppure crearmi il mio (vedere data_loader.py)
}   #attenzione inserire il nome del nostro data set


def data_provider(args, flag):
    Data = data_dict[args.data] #imposta la classe
    timeenc = 0 if args.embed != 'timeF' else 1
    #scale = args.scale #AGGIUNTO
    if args.scale == 'False':
        scale = False
    else: 
        scale = True

    percent = args.percent

    if flag == 'test':
        shuffle_flag = False
        drop_last = True
        batch_size = args.batch_size
        freq = args.freq
    else:
        shuffle_flag = True
        drop_last = True
        batch_size = args.batch_size
        freq = args.freq

    if args.data == 'm4':
        drop_last = False
        data_set = Data(
            root_path=args.root_path,
            data_path=args.data_path,
            flag=flag,
            size=[args.seq_len, args.label_len, args.pred_len],
            features=args.features,
            target=args.target,
            timeenc=timeenc,
            freq=freq,
            seasonal_patterns=args.seasonal_patterns
        )
    else:
        data_set = Data(
            root_path=args.root_path,
            data_path=args.data_path,
            flag=flag,
            size=[args.seq_len, args.label_len, args.pred_len],
            features=args.features,
            target=args.target,
            scale = scale, #AGGIUNTO
            timeenc=timeenc,
            freq=freq,
            percent=percent,
            seasonal_patterns=args.seasonal_patterns
        )
    data_loader = DataLoader( #funzione di torch!!  #errori per il val??
        data_set,
        batch_size=batch_size,
        shuffle=shuffle_flag,
        num_workers=args.num_workers,  #number of trets that we use in order to load the date
        drop_last=drop_last)
    return data_set, data_loader #costruisce il dataset
