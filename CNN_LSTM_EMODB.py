import numpy as np
import os
import librosa


datapath = './wav'
classes = ['W','F','T','N'] # 7 classes


seg_len = 16000  # sample rate 檔案中每秒鐘對這個聲音的取樣次數皆為16000
seg_ov = int(seg_len*0.5) # 50% overlap 使這個變量step_size是的一半window_size

def normalize(s): # 使用RMS normalization歸一化(Root Mean Square): 將峰值相加並除以峰值數量，使用新的平均峰值處理信號，具有不同波峰和波谷的長音頻文件上效果最好。
    new_s = s/np.sqrt(np.sum(np.square((np.abs(s))))/len(s))
    return new_s

def countclasses(fnames):
    dict = {classes[0]:0,classes[1]:0,classes[2]:0,classes[3]:0}
    for name in fnames:
        if name[5] in classes:  # 音檔名中第五個字母代表說話者的情感編號，若有對應classes相同的英文字母，則加入dict的類別數
            dict[name[5]]+=1
    return dict

def data1d(path): # 數據處理

    fnames = os.listdir(datapath)
    dict = countclasses(fnames) # 將所有音檔編號對應數量存為dict
    print('Total Data',dict)
 
    num_cl = len(classes) # 7 classes
  # 建立 train,test,val的dictionary
    train_dict = {classes[0]:0,classes[1]:0,classes[2]:0,classes[3]:0}
    test_dict = {classes[0]:0,classes[1]:0,classes[2]:0,classes[3]:0}
    val_dict = {classes[0]:0,classes[1]:0,classes[2]:0,classes[3]:0}

    for i in range(num_cl):
        cname =  list(dict.keys())[i]
        cnum = dict[cname]
        t = round(0.8*cnum) # 將 t 設為各情緒類別數據集的80%
        test_dict[cname] = int(cnum - t) # test data為各情緒類別數據集的 20 %
        val_dict[cname] = int(round(0.2*t)) #  從ｔ提取20%作為 validation data
        train_dict[cname] = int(t - val_dict[cname]) # 剩下的80%的t 作為train data
        print('Class:',cname,'train:',train_dict[cname],'val:',val_dict[cname],'test:',test_dict[cname])

    x_train = []
    y_train = []
    x_test = []
    y_test = []
    x_val = []
    y_val = []

    count = {classes[0]:0,classes[1]:0,classes[2]:0,classes[3]:0}

    for name in fnames:
        # 若音檔名中第五個字母有對應classes相同的英文字母
        if name[5] in classes:
            sig,fs = librosa.load(datapath+'/'+name, sr=16000)  # 將音頻數據加載為浮點時間序列, sr為音檔的sample rate =16000 --> sig為讀取的數據、fs為sample rate 
            # normalize signal
            data = normalize(sig) # 將讀取的數據(sig)歸一化
            
            ## 若讀取的音檔數據長度 小於seg_len(sample rate)
            if(len(data) < seg_len):  
                pad_len = int(seg_len - len(data))  
                pad_rem = int(pad_len % 2) #
                pad_len /= 2
                signal = np.pad(data,(int(pad_len), int(pad_len+pad_rem)),'constant',constant_values=0)
            ## 若讀取的音檔數據長度 大於seg_len(sample rate)
            elif(len(data) > seg_len):
                signal = []
                end = seg_len
                st = 0
                while(end < len(data)):  # 數據長度大於seg len時
                    signal.append(data[st:end]) 
                    st = st + seg_ov   # seg_ov = 8000
                    end = st + seg_len
                signal = np.array(signal)
                if(end >= len(data)):  # 直到數據長度小於seg len時
                    num_zeros = int(end-len(data)) 
                    if(num_zeros > 0):
                        n1 = np.array(data[st:end])
                        n2 = np.zeros([num_zeros])  # 返回一個用0填充的數組
                        s = np.concatenate([n1,n2],0)  # 垂直堆疊矩陣
                    else:
                        s = np.array(data[int(st):int(end)])
                signal = np.vstack([signal,s]) # 垂直堆疊矩陣
                
            ## 若讀取的音檔數據長度 等於seg_len(sample rate)
            else:
                signal = data
            
            
            ''''處理train,val,test數據資料'''
            if(count[name[5]] < train_dict[name[5]]):
                if(signal.ndim>1): #若signal維度大於1
                    for i in range(signal.shape[0]):
                        x_train.append(signal[i])
                        y_train.append(name[5])
                else:
                    x_train.append(signal)
                    y_train.append(name[5])
            else:
                if((count[name[5]]-train_dict[name[5]]) < val_dict[name[5]]):
                    if(signal.ndim>1):
                        for i in range(signal.shape[0]):
                            x_val.append(signal[i])
                            y_val.append(name[5])
                    else:
                        x_val.append(signal)
                        y_val.append(name[5])
                else:
                    if(signal.ndim>1):
                        for i in range(signal.shape[0]):
                            x_test.append(signal[i])
                            y_test.append(name[5])
                    else:
                        x_test.append(signal)
                        y_test.append(name[5])
            count[name[5]]+=1
    return np.float32(x_train),y_train,np.float32(x_test),y_test,np.float32(x_val),y_val

def string2num(y): # 將資料集中對應的情感字母編號轉成數字1-6
    y1 = []
    for i in y:
        if(i == classes[0]):
            y1.append(0)
        elif(i == classes[1]):
            y1.append(1)
        elif(i == classes[2]):
            y1.append(2)
        else:
            y1.append(3)
    y1 = np.float32(np.array(y1)) # list to array
    return y1

def load_data():
    x_tr,y_tr,x_t,y_t,x_v,y_v = data1d(datapath)
    y_tr = string2num(y_tr)
    y_t = string2num(y_t)
    y_v = string2num(y_v)
    return x_tr, y_tr, x_t, y_t, x_v, y_v
           

'''模型訓練:一維 CNN-LSTM'''
def emo1d(input_shape, num_classes, args):
    model = Sequential(name='Emo1D')

    # 第一層CNN (一維度卷基層)
    model.add(Conv1D(filters=64, kernel_size=(3), strides=1, padding='same', data_format='channels_last',
                     input_shape=input_shape))
    model.add(BatchNormalization())
    model.add(Activation('elu'))
    model.add(MaxPooling1D(pool_size=4, strides=4))

    # 第二層CNN
    model.add(Conv1D(filters=64, kernel_size=3, strides=1, padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('elu'))
    model.add(MaxPooling1D(pool_size=4, strides=4))

    # 第三層CNN
    model.add(Conv1D(filters=128, kernel_size=3, strides=1, padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('elu'))
    model.add(MaxPooling1D(pool_size=4, strides=4))

    # 第四層CNN
    model.add(Conv1D(filters=128, kernel_size=3, strides=1, padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('elu'))
    model.add(MaxPooling1D(pool_size=4, strides=4))

    # LSTM
    model.add(LSTM(units=args.num_fc,return_sequences=True))
    model.add(Activation('tanh'))
    model.add(LSTM(units=args.num_fc,return_sequences=False))

    # FC layer
    model.add(Dense(units=num_classes, activation='softmax'))

    # Model compilation
    opt = optimizers.SGD(lr=args.learning_rate, decay=args.decay, momentum=args.momentum, nesterov=True)
    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['categorical_accuracy'])

    return model


def train(model, x_tr, y_tr, x_val, y_val, args):
    es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=8) # 當被監測的數量不再提升，則停止訓練
    mc = ModelCheckpoint('best_model.h5', monitor='val_categorical_accuracy', mode='max', verbose=1,
                         save_best_only=True) # 在每個訓練期之後保存模型
    history = model.fit(x_tr, y_tr, epochs=args.num_epochs, batch_size=args.batch_size, validation_data=(x_val, y_val),
                        callbacks=[es, mc])
    return model


def test(model, x_t, y_t):
    saved_model = load_model('best_model.h5',custom_objects={'SeqSelfAttention':SeqSelfAttention})
    score = saved_model.evaluate(x_t, y_t, batch_size=20)
    print(score)
    return score


def loadData():
    x_tr, y_tr, x_t, y_t, x_val, y_val = load_data()
    x_tr = x_tr.reshape(-1, x_tr.shape[1], 1)
    x_t = x_t.reshape(-1, x_t.shape[1], 1)
    x_val = x_val.reshape(-1, x_val.shape[1], 1)
    y_tr = to_categorical(y_tr)
    y_t = to_categorical(y_t)
    y_val = to_categorical(y_val)
    return x_tr, y_tr, x_t, y_t, x_val, y_val


if __name__ == "__main__":
    import numpy as np
    import matplotlib.pyplot as plt

    parser = argparse.ArgumentParser()
    args = parser.parse_args(args=[])

    # load data
    x_tr, y_tr, x_t, y_t, x_val, y_val = loadData()

    args.num_fc = 64
    args.batch_size = 32
    args.num_epochs = 200 # best model will be saved before number of epochs reach this value
    args.learning_rate = 0.0001
    args.decay = 1e-6
    args.momentum = 0.9
    

    # define model
    model = emo1d(input_shape=x_tr.shape[1:], num_classes=len(np.unique(np.argmax(y_tr, 1))), args=args)
    model.summary()

    # train model
    model = train(model, x_tr, y_tr, x_val, y_val, args=args)

    # test model
    score = test(model, x_t, y_t) 