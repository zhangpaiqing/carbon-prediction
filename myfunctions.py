#!/usr/bin/env python
# coding: utf-8

import os 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from math import sqrt

from PyEMD import EMD, EEMD, CEEMDAN
import ewtpy

from bayes_opt import BayesianOptimization

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error

import warnings
warnings.filterwarnings("ignore")

# In[1]:
# 将一维时间序列转换为监督学习格式的滑动窗口数据
def createXy(data, look_back=1, forecast_steps=1):
    """
    参数：
      data: 一维数组，形状 (n,)
      look_back: 输入窗口大小（历史时间步数）
      forecast_steps: 预测未来步数（默认为1，即单步预测）
    返回：
      X: 输入数据，形状 (n - look_back - forecast_steps + 1, look_back)
      y: 输出数据，形状 (n - look_back - forecast_steps + 1, forecast_steps)
    """
    X, y = [], []
    for i in range(len(data) - look_back - forecast_steps + 1):
        X.append(data[i:i+look_back])
        y.append(data[i+look_back:i+look_back+forecast_steps])
    return np.array(X), np.array(y)


# 按照7:2:1划分训练验证测试集
def get_train_val_test(X, y):
    """  
    参数：
        X: 输入特征数组，形状为 (n_samples, n_features)
        y: 输出标签数组，形状为 (n_samples, forecast_steps)
        
    返回：
        X_train, X_val, X_test, y_train, y_val, y_test，都是二维数组
    """   
    n_samples = len(X)
    # 计算划分索引
    train_end = int(n_samples * 0.7)       # 70% 训练
    val_end = train_end + int(n_samples * 0.2)  # 20% 验证
    # 剩余部分为测试集（约10%）
    # 划分数据集
    X_train, y_train = X[:train_end], y[:train_end]
    X_val, y_val = X[train_end:val_end], y[train_end:val_end]
    X_test, y_test = X[val_end:], y[val_end:]
    return X_train, X_val, X_test, y_train, y_val, y_test

# In[2]:


##SVR

def svr_model(datass,look_back):
    """
    datass：pandas.DataFrame 对象。
    look_back：时间步长，即用于预测下一个时间步的历史数据的长度。
    """
    # 1. 数据提取
    series = datass.iloc[:, 1].values
    
    # 2. 生成滑动窗口数据
    X, y = createXy(series, look_back)
    
    # 3. 划分数据集（训练+验证+测试）
    X_train, X_val, X_test, y_train, y_val, y_test = get_train_val_test(X, y)
    
    # 4. 数据标准化
    sc_X = StandardScaler()
    sc_y = StandardScaler()
    # 训练集拟合标准化
    X_train_scaled = sc_X.fit_transform(X_train)
    y_train_scaled = sc_y.fit_transform(y_train)
    # 验证集和测试集转换
    X_val_scaled = sc_X.transform(X_val)
    y_val_scaled = sc_y.transform(y_val)
    X_test_scaled = sc_X.transform(X_test)
    y_test_scaled = sc_y.transform(y_test)
         
    np.random.seed(1234)
    tf.random.set_seed(1234)
    
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    
    # 5. 构建SVR模型
    grid = SVR(kernel='rbf')
    # 6. 训练模型：
    '''
        SVR中fit(X,y) 用法：X通常为 numpy.ndarray 且必须是二维，也可以是df，scikit-learn 会自动将其转换为 NumPy 数组进行处理。
        # y对于单变量回归，通常是一维array或 Series 类型；对于多变量回归，是二维的array 或 DataFrame 类型。
        # 在这里我统一X，y都是array。
     '''
    grid.fit(X_train_scaled,y_train_scaled.flatten())
    # 7. 预测测试集
    y_pred_test_svr= grid.predict(X_test_scaled)
    y_pred_test_svr=y_pred_test_svr.reshape(-1,1)  #.reshape将一维数组转为二维
    y_pred_test= sc_y.inverse_transform (y_pred_test_svr)
    
    # 8. 计算评估指标
    y_test_true = sc_y.inverse_transform(y_test_scaled)
    mape = np.mean(np.abs((y_test_true - y_pred_test) / y_test_true)) * 100
    rmse= sqrt(mean_squared_error(y_test_true,y_pred_test))
    mae=mean_absolute_error(y_test_true,y_pred_test)
    r2 = 1 - (np.sum((y_test_true - y_pred_test) ** 2) / np.sum((y_test_true - np.mean(y_test_true)) ** 2))

    print(f"MAPE:{mape:.3f}")
    print(f"RMSE:{rmse:.3f}")
    print(f"MAE:{mae:.3f}")
    print(f"r2:{r2:.3f}")
    
    return y_test_true,y_pred_test


# In[3]:


##ANN
def ann_model(datass,look_back):
    """
    datass：输入的数据集，通常是一个 pandas.DataFrame 对象。
    look_back：时间步长，即用于预测下一个时间步的历史数据的长度。
    """
    # 1. 提取数值列
    series = datass.iloc[:, 1].values
    
    # 2. 生成滑动窗口数据
    X, y = createXy(series, look_back)
    
    # 3. 划分数据集（训练+验证+测试）
    X_train, X_val, X_test, y_train, y_val, y_test = get_train_val_test(X, y)
    
    # 4. 数据标准化（特征和标签分开处理）
    sc_X = StandardScaler()
    sc_y = StandardScaler()
    # 训练集拟合标准化器
    X_train_scaled = sc_X.fit_transform(X_train)
    y_train_scaled = sc_y.fit_transform(y_train)
    # 验证集和测试集转换
    X_val_scaled = sc_X.transform(X_val)
    y_val_scaled = sc_y.transform(y_val)
    X_test_scaled = sc_X.transform(X_test)
    y_test_scaled = sc_y.transform(y_test)
         
    np.random.seed(1234)
    tf.random.set_seed(1234)
    
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    
    # 5. 构建ANN模型
    model = Sequential()
    model.add(Dense(128,activation='relu',input_shape=(look_back, )))  #全连接层（Dense层）的input_shape只需要考虑输入特征数量
    model.add(Dense(64, activation='relu'))
    model.add(Dense(1)) # 输出层
    optimizer = Adam(learning_rate=0.001)
    model.compile(loss='mse',optimizer=optimizer)   

    # 6. 训练模型（添加早停机制）
    ''' 在深度学习模型训练过程中，随着训练轮数（epoch）的增加，模型在训练集上的表现通常会越来越好，但在验证集上可能会先提升后下降，
       出现过拟合现象。EarlyStopping 回调函数的作用就是在监测到模型在验证集上的性能不再提升时，提前停止训练，从而避免过拟合，
       同时还可以选择恢复到性能最好时的模型权重。
       monitor含义：指定要监测的指标,'val_loss'（验证集损失）
       patience=10含义：如果在连续 10 个 epoch 中，验证集损失都没有下降（对于 monitor='val_loss' 的情况），则停止训练
       restore_best_weights含义：设置为 True 表示在停止训练时，将模型的权重恢复到监测指标表现最好时的权重
       
    '''
    early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    '''
    ANN中fit(X,y) 用法：X通常为array 类型，形状需要与模型输入层的 input_shape 参数相匹配，model.add(Dense(128, activation='relu', input_shape=(look_back, ))) 表明输入层期望的输入形状为 (look_back,)，所以X形状为 (n_samples, look_back)
    y通常为array 类型，model.add(Dense(1)) 表明输出层只有一个神经元，所以y应该是一个一维数组，形状为 (n_samples,)
    
    '''
    model.fit(
        X_train_scaled, y_train_scaled.flatten(),
        epochs=100,
        batch_size=64,
        validation_data=(X_val_scaled, y_val_scaled.flatten()),
        callbacks=[early_stop],
        verbose=0
    )    
    # 7. 预测测试集
    y_pred_test_ann = model.predict(X_test_scaled, verbose=0)
    y_pred_test_ann =y_pred_test_ann.reshape(-1,1)  #.reshape将一维数组转为二维
    y_pred_test = sc_y.inverse_transform(y_pred_test_ann)
    
    
    # 8. 计算评估指标
    y_test_true = sc_y.inverse_transform(y_test_scaled)
    mape = np.mean(np.abs((y_test_true - y_pred_test) / y_test_true)) * 100
    rmse= sqrt(mean_squared_error(y_test_true,y_pred_test))
    mae=mean_absolute_error(y_test_true,y_pred_test)
    r2 = 1 - (np.sum((y_test_true - y_pred_test) ** 2) / np.sum((y_test_true - np.mean(y_test_true)) ** 2))

    print(f"MAPE:{mape:.3f}")
    print(f"RMSE:{rmse:.3f}")
    print(f"MAE:{mae:.3f}")
    print(f"r2:{r2:.3f}")
    
    return y_test_true,y_pred_test


# In[4]:


##RF 
def rf_model(datass,look_back):
    """
    datass：输入的数据集，通常是一个 pandas.DataFrame 对象。
    look_back：时间步长，即用于预测下一个时间步的历史数据的长度。
    """
    # 1. 提取数值列
    series = datass.iloc[:, 1].values
    
    # 2. 生成滑动窗口数据
    X, y = createXy(series, look_back)
    
    # 3. 划分数据集（训练+验证+测试）
    X_train, X_val, X_test, y_train, y_val, y_test = get_train_val_test(X, y)
    
    # 4. 数据标准化（特征和标签分开处理）
    sc_X = StandardScaler()
    sc_y = StandardScaler()
    # 训练集拟合标准化器
    X_train_scaled = sc_X.fit_transform(X_train)
    y_train_scaled = sc_y.fit_transform(y_train)
    # 验证集和测试集转换
    X_val_scaled = sc_X.transform(X_val)
    y_val_scaled = sc_y.transform(y_val)
    X_test_scaled = sc_X.transform(X_test)
    y_test_scaled = sc_y.transform(y_test)
         
    np.random.seed(1234)
    tf.random.set_seed(1234)
    
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    
    # 5. 构建RF模型
    grid = RandomForestRegressor()
    
    # 6. 训练模型
    grid.fit(X_train_scaled,y_train_scaled.ravel())
  
    # 7. 预测测试集
    y_pred_test_rf = grid.predict(X_test_scaled)
    y_pred_test_rf =y_pred_test_rf.reshape(-1,1)  #.reshape将一维数组转为二维
    y_pred_test = sc_y.inverse_transform(y_pred_test_rf)
    
    
    # 8. 计算评估指标
    y_test_true = sc_y.inverse_transform(y_test_scaled)
    mape = np.mean(np.abs((y_test_true - y_pred_test) / y_test_true)) * 100
    rmse= sqrt(mean_squared_error(y_test_true,y_pred_test))
    mae=mean_absolute_error(y_test_true,y_pred_test)
    r2 = 1 - (np.sum((y_test_true - y_pred_test) ** 2) / np.sum((y_test_true - np.mean(y_test_true)) ** 2))

    print(f"MAPE:{mape:.3f}")
    print(f"RMSE:{rmse:.3f}")
    print(f"MAE:{mae:.3f}")
    print(f"r2:{r2:.3f}")
    
    return y_test_true,y_pred_test


# In[5]:


##LSTM

def lstm_model(datass,look_back):
    """
    datass：输入的数据集，通常是一个 pandas.DataFrame 对象。
    look_back：时间步长，即用于预测下一个时间步的历史数据的长度。
    """
    # 1. 数据提取
    series = datass.iloc[:, 1].values
    
    # 2. 生成滑动窗口数据
    X, y = createXy(series, look_back)
    
    # 3. 划分数据集（训练+验证+测试）
    X_train, X_val, X_test, y_train, y_val, y_test = get_train_val_test(X, y)
    
    # 4. 数据标准化（特征和标签分开处理）
    sc_X = StandardScaler()
    sc_y = StandardScaler()
    # 训练集拟合标准化器
    X_train_scaled = sc_X.fit_transform(X_train)
    y_train_scaled = sc_y.fit_transform(y_train)
    # 验证集和测试集转换
    X_val_scaled = sc_X.transform(X_val)
    y_val_scaled = sc_y.transform(y_val)
    X_test_scaled = sc_X.transform(X_test)
    y_test_scaled = sc_y.transform(y_test)
    
    # 5. 调整数据形状为LSTM输入格式 (samples, time_steps, features)
    X_train_reshaped = X_train_scaled.reshape((X_train_scaled.shape[0], look_back, 1))
    X_val_reshaped = X_val_scaled.reshape((X_val_scaled.shape[0], look_back, 1))
    X_test_reshaped = X_test_scaled.reshape((X_test_scaled.shape[0], look_back, 1))
      
    np.random.seed(1234)
    tf.random.set_seed(1234)
    
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    
    # 6. 构建LSTM模型
    model = Sequential()
    model.add(LSTM(128,activation='relu',input_shape=(look_back,1))) #
    '''
     input_shape=(look_back,1)这个1必须有。
     LSTM 层，input_shape的格式通常是(时间步长, 特征维度)。这里的look_back代表时间步长，意味着 LSTM 层在处理输入数据时，
     会按look_back个时间步长为一组来处理。而1代表特征维度，说明每个时间步上只有一个特征。
     例如，如果我们在做时间序列预测，预测每天的温度，这里look_back可能是过去 10 天，
     即我们以过去 10 天的温度数据作为一个输入序列，而每天的温度就是一个特征，所以特征维度为 1。
     而全连接层（Dense层）的input_shape只需要考虑输入特征数量。
    '''
    model.add(Dense(64,activation='relu')) # 添加全连接层提升非线性能力
    model.add(Dense(1)) 
    optimizer = Adam(learning_rate=0.001)
    model.compile(loss='mse',optimizer=optimizer)
#    model.summary()
    

   # 7. 训练模型（添加早停）
    early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    '''
    model.fit(X,y)用法：X其形状应该与LSTM层定义的input_shape相匹配，即X应该是一个三维数组，形状为(样本数量, look_back, 1)
    y对于单值预测任务，y应该是一个一维数组
    '''
    model.fit(
        X_train_reshaped, y_train_scaled.flatten(),
        epochs=100,
        batch_size=64,
        validation_data=(X_val_reshaped, y_val_scaled.flatten()),
        callbacks=[early_stop],
        verbose=0
    )
    
   # 8. 预测测试集
    y_pred_test_lstm = model.predict(X_test_scaled, verbose=0)
    y_pred_test_lstm =y_pred_test_lstm.reshape(-1,1)  #.reshape将一维数组转为二维
    y_pred_test = sc_y.inverse_transform(y_pred_test_lstm)
    
    
    # 9. 计算评估指标
    y_test_true = sc_y.inverse_transform(y_test_scaled)
    mape = np.mean(np.abs((y_test_true - y_pred_test) / y_test_true)) * 100
    rmse= sqrt(mean_squared_error(y_test_true,y_pred_test))
    mae=mean_absolute_error(y_test_true,y_pred_test)
    r2 = 1 - (np.sum((y_test_true - y_pred_test) ** 2) / np.sum((y_test_true - np.mean(y_test_true)) ** 2))

    print(f"MAPE:{mape:.3f}")
    print(f"RMSE:{rmse:.3f}")
    print(f"MAE:{mae:.3f}")
    print(f"r2:{r2:.3f}")
    
    return y_test_true,y_pred_test


# In[6]:


##HYBRID EMD LSTM
def emd_lstm(datass,look_back):

    # 1. 数据提取
    series = datass.iloc[:, 1].values
    
    # 2. EMD 分解
    emd = EMD()
    imfs = emd(series)
    imfs_df = pd.DataFrame(imfs.T)  # 分量存储为 (n_samples, n_imfs)
    
    # 3. 对各分量进行LSTM预测
    pred_test=[]
    # 对 imfs_df 中的每一列数据进行训练集和测试集的划分
    for idx in range(imfs_df.shape[1]):
        component = imfs_df.iloc[:, idx].values
        
        # 划分数据集（保持时间顺序）
        X, y = createXy(component, look_back)
        X_train, X_val, X_test, y_train, y_val, y_test = get_train_val_test(X, y)
        
        # 标准化
        sc_X = StandardScaler()
        sc_y = StandardScaler()
        X_train_scaled = sc_X.fit_transform(X_train)
        y_train_scaled = sc_y.fit_transform(y_train)
        X_val_scaled = sc_X.transform(X_val)
        y_val_scaled = sc_y.transform(y_val)
        X_test_scaled = sc_X.transform(X_test)
        y_test_scaled = sc_y.transform(y_test)
        
        # 调整输入形状为 LSTM 格式 (samples, time_steps, features)
        X_train_reshaped = X_train_scaled.reshape(X_train_scaled.shape[0], look_back, 1)
        X_val_reshaped = X_val_scaled.reshape(X_val_scaled.shape[0], look_back, 1)
        X_test_reshaped = X_test_scaled.reshape(X_test_scaled.shape[0], look_back, 1)

        np.random.seed(1234)
        tf.random.set_seed(1234)      
    
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

        # 构建LSTM模型
        model = Sequential()
        model.add(LSTM(128,activation='relu',input_shape=(look_back, 1)))
        model.add(Dense(64,activation='relu')) # 添加全连接层提升非线性能力
        model.add(Dense(1)) 
        optimizer = Adam(learning_rate=0.001)
        model.compile(loss='mse',optimizer=optimizer)

        # 训练模型（添加早停）
        early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
        history = model.fit(
            X_train_reshaped, y_train_scaled.flatten(),
            epochs=100,
            batch_size=64,
            validation_data=(X_val_reshaped, y_val_scaled.flatten()),
            callbacks=[early_stop],
            verbose=0
        )

        # 预测并逆标准化
        y_pred = model.predict(X_test_scaled, verbose=0)
        y_pred =y_pred.reshape(-1,1)  #.reshape将一维数组转为二维
        y_pred = sc_y.inverse_transform(y_pred)
        pred_test.append(y_pred) 
        
    # 5. 合并各分量预测结果
    # pred_test是二嵌套列表（列表嵌套列表，再嵌套列表），类似[[[1, 2], [3, 4]], [[5, 6], [7, 8]]]
    pred_test=np.array(pred_test)   #pred_test形状(9, 196, 1)可以将其看作是包含9 个形状为 (196, 1) 的二维数组。
    y_pred_test = np.sum(np.hstack(pred_test), axis=1) #hstack在列方向上拼接后pred_test形状为(196, 9)，xis=1 表示按行求和，返回一维数组
    y_pred_test=y_pred_test.reshape(-1,1)
    #6. 评估
    X, y = createXy(series, look_back)
    X_train, X_val, X_test, y_train, y_val, y_test = get_train_val_test(X, y)
    sc_y = StandardScaler()
    y_train_scaled = sc_y.fit_transform(y_train)
    y_test_scaled = sc_y.transform(y_test)
    y_test_true = sc_y.inverse_transform(y_test_scaled)
    mape = np.mean(np.abs((y_test_true - y_pred_test) / y_test_true)) * 100
    rmse= sqrt(mean_squared_error(y_test_true,y_pred_test))
    mae=mean_absolute_error(y_test_true,y_pred_test)
    r2 = 1 - (np.sum((y_test_true - y_pred_test) ** 2) / np.sum((y_test_true - np.mean(y_test_true)) ** 2))

    print(f"MAPE:{mape:.3f}")
    print(f"RMSE:{rmse:.3f}")
    print(f"MAE:{mae:.3f}")
    print(f"r2:{r2:.3f}")
    
    return y_test_true,y_pred_test



# In[7]:


##HYBRID EEMD LSTM
def eemd_lstm(datass,look_back):

    # 1. 数据提取
    series = datass.iloc[:, 1].values
    
    # 2. EEMD 分解
    emd = EEMD(noise_width=0.05) ##noise_width添加到原始信号中的噪声的幅度，是噪声标准差与原始信号标准差的比值。
    emd.noise_seed(12345)
    imfs = emd(series)
    imfs_df = pd.DataFrame(imfs.T)  # 分量存储为 (n_samples, n_imfs)
    
    # 3. 对各分量进行LSTM预测
    pred_test=[]
    # 对 imfs_df 中的每一列数据进行训练集和测试集的划分
    for idx in range(imfs_df.shape[1]):
        component = imfs_df.iloc[:, idx].values
        
        # 划分数据集（保持时间顺序）
        X, y = createXy(component, look_back)
        X_train, X_val, X_test, y_train, y_val, y_test = get_train_val_test(X, y)
        
        # 标准化
        sc_X = StandardScaler()
        sc_y = StandardScaler()
        X_train_scaled = sc_X.fit_transform(X_train)
        y_train_scaled = sc_y.fit_transform(y_train)
        X_val_scaled = sc_X.transform(X_val)
        y_val_scaled = sc_y.transform(y_val)
        X_test_scaled = sc_X.transform(X_test)
        y_test_scaled = sc_y.transform(y_test)
        
        # 调整输入形状为 LSTM 格式 (samples, time_steps, features)
        X_train_reshaped = X_train_scaled.reshape(X_train_scaled.shape[0], look_back, 1)
        X_val_reshaped = X_val_scaled.reshape(X_val_scaled.shape[0], look_back, 1)
        X_test_reshaped = X_test_scaled.reshape(X_test_scaled.shape[0], look_back, 1)

        np.random.seed(1234)
        tf.random.set_seed(1234)      
    
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

        # 构建LSTM模型
        model = Sequential()
        model.add(LSTM(128,activation='relu',input_shape=(look_back, 1)))
        model.add(Dense(64,activation='relu')) # 添加全连接层提升非线性能力
        model.add(Dense(1)) 
        optimizer = Adam(learning_rate=0.001)
        model.compile(loss='mse',optimizer=optimizer)

        # 训练模型（添加早停）
        early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
        history = model.fit(
            X_train_reshaped, y_train_scaled.flatten(),
            epochs=100,
            batch_size=64,
            validation_data=(X_val_reshaped, y_val_scaled.flatten()),
            callbacks=[early_stop],
            verbose=0
        )

        # 预测并逆标准化
        y_pred = model.predict(X_test_scaled, verbose=0)
        y_pred =y_pred.reshape(-1,1)  #.reshape将一维数组转为二维
        y_pred = sc_y.inverse_transform(y_pred)
        pred_test.append(y_pred) 
        
    # 5. 合并各分量预测结果
    pred_test=np.array(pred_test)   #(9, 196, 1) 可以将其看作是包含9 个形状为 (196, 1) 的二维数组。
    y_pred_test = np.sum(np.hstack(pred_test), axis=1) #hstack后(196, 9)
    y_pred_test=y_pred_test.reshape(-1,1)
    #6. 评估
    X, y = createXy(series, look_back)
    X_train, X_val, X_test, y_train, y_val, y_test = get_train_val_test(X, y)
    sc_y = StandardScaler()
    y_train_scaled = sc_y.fit_transform(y_train)
    y_test_scaled = sc_y.transform(y_test)
    y_test_true = sc_y.inverse_transform(y_test_scaled)
    mape = np.mean(np.abs((y_test_true - y_pred_test) / y_test_true)) * 100
    rmse= sqrt(mean_squared_error(y_test_true,y_pred_test))
    mae=mean_absolute_error(y_test_true,y_pred_test)
    r2 = 1 - (np.sum((y_test_true - y_pred_test) ** 2) / np.sum((y_test_true - np.mean(y_test_true)) ** 2))

    print(f"MAPE:{mape:.3f}")
    print(f"RMSE:{rmse:.3f}")
    print(f"MAE:{mae:.3f}")
    print(f"r2:{r2:.3f}")
    
    return y_test_true,y_pred_test



# In[8]:


##HYBRID CEEMDAN LSTM
def ceemdan_lstm(datass,look_back):

    # 1. 数据提取
    series = datass.iloc[:, 1].values
    
    # 2. EMD 分解
    emd = CEEMDAN(epsilon=0.05)  #epsilon添加到原始信号中的噪声的幅度，是噪声标准差与原始信号标准差的比值。
    emd.noise_seed(12345)
    imfs = emd(series)
    imfs_df = pd.DataFrame(imfs.T)  # 分量存储为 (n_samples, n_imfs)
    
    # 3. 对各分量进行LSTM预测
    pred_test=[]
    # 对 new_ceemdan 中的每一列数据进行训练集和测试集的划分
    for idx in range(imfs_df.shape[1]):
        component = imfs_df.iloc[:, idx].values
        
        # 划分数据集（保持时间顺序）
        X, y = createXy(component, look_back)
        X_train, X_val, X_test, y_train, y_val, y_test = get_train_val_test(X, y)
        
        # 标准化
        sc_X = StandardScaler()
        sc_y = StandardScaler()
        X_train_scaled = sc_X.fit_transform(X_train)
        y_train_scaled = sc_y.fit_transform(y_train)
        X_val_scaled = sc_X.transform(X_val)
        y_val_scaled = sc_y.transform(y_val)
        X_test_scaled = sc_X.transform(X_test)
        y_test_scaled = sc_y.transform(y_test)
        
        # 调整输入形状为 LSTM 格式 (samples, time_steps, features)
        X_train_reshaped = X_train_scaled.reshape(X_train_scaled.shape[0], look_back, 1)
        X_val_reshaped = X_val_scaled.reshape(X_val_scaled.shape[0], look_back, 1)
        X_test_reshaped = X_test_scaled.reshape(X_test_scaled.shape[0], look_back, 1)

        np.random.seed(1234)
        import tensorflow as tf
        tf.random.set_seed(1234)      
    
        import os 
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

        # 构建LSTM模型
        model = Sequential()
        model.add(LSTM(128,activation='relu',input_shape=(look_back, 1)))
        model.add(Dense(64,activation='relu')) # 添加全连接层提升非线性能力
        model.add(Dense(1)) 
        optimizer = Adam(learning_rate=0.001)
        model.compile(loss='mse',optimizer=optimizer)

        # 训练模型（添加早停）
        early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
        history = model.fit(
            X_train_reshaped, y_train_scaled.flatten(),
            epochs=100,
            batch_size=64,
            validation_data=(X_val_reshaped, y_val_scaled.flatten()),
            callbacks=[early_stop],
            verbose=0
        )

        # 预测并逆标准化
        y_pred = model.predict(X_test_scaled, verbose=0)
        y_pred =y_pred.reshape(-1,1)  #.reshape将一维数组转为二维
        y_pred = sc_y.inverse_transform(y_pred)
        pred_test.append(y_pred) 
        
    # 5. 合并各分量预测结果
    pred_test=np.array(pred_test)   #(9, 196, 1) 可以将其看作是包含9 个形状为 (196, 1) 的二维数组。
    y_pred_test = np.sum(np.hstack(pred_test), axis=1) #hstack后(196, 9)
    y_pred_test=y_pred_test.reshape(-1,1)
    #6. 评估
    X, y = createXy(series, look_back)
    X_train, X_val, X_test, y_train, y_val, y_test = get_train_val_test(X, y)
    sc_y = StandardScaler()
    y_train_scaled = sc_y.fit_transform(y_train)
    y_test_scaled = sc_y.transform(y_test)
    y_test_true = sc_y.inverse_transform(y_test_scaled)
    mape = np.mean(np.abs((y_test_true - y_pred_test) / y_test_true)) * 100
    rmse= sqrt(mean_squared_error(y_test_true,y_pred_test))
    mae=mean_absolute_error(y_test_true,y_pred_test)
    r2 = 1 - (np.sum((y_test_true - y_pred_test) ** 2) / np.sum((y_test_true - np.mean(y_test_true)) ** 2))

    print(f"MAPE:{mape:.3f}")
    print(f"RMSE:{rmse:.3f}")
    print(f"MAE:{mae:.3f}")
    print(f"r2:{r2:.3f}")
    
    return y_test_true,y_pred_test


# In[9]:


##Proposed Method Hybrid CEEMDAN-EWT LSTM
def ceemdan_ewt_lstm(datass,look_back):

    # 1. 数据提取
    series = datass.iloc[:, 1].values
    
    # 2. CEEMDAN 分解
    emd = CEEMDAN(epsilon=0.05)
    emd.noise_seed(12345)
    imfs = emd(series)
    imfs_df = pd.DataFrame(imfs.T)  # 分量存储为 (n_samples, n_imfs)
    
    # 3. EWT 去噪第一个 IMF
    imf1 = imfs_df.iloc[:, 0].values   #imf1是一维数组
    ewt_result,  mfb ,boundaries = ewtpy.EWT1D(imf1, N =3)  #imf1 自适应地分解为 3 个分量
    ewt_df = pd.DataFrame(ewt_result)
    # 去除高频噪声分量（假设最后一个分量是噪声）
    denoised_imf1 = ewt_df.iloc[:, :-1].sum(axis=1).values  #:-1 表示选择除了最后一列之外的所有列，axis=1 表示按行进行求和
    # 重构信号：去噪后的IMF1 + 其他IMFs
    reconstructed_imfs = pd.concat([pd.Series(denoised_imf1),imfs_df.iloc[:, 1:]], axis=1)  
    
    # 4. 对各分量进行LSTM预测
    pred_test=[]
    # 对 new_ceemdan 中的每一列数据进行训练集和测试集的划分
    for idx in range(reconstructed_imfs.shape[1]):
        component = reconstructed_imfs.iloc[:, idx].values
        
        # 划分数据集（保持时间顺序）
        X, y = createXy(component, look_back)
        X_train, X_val, X_test, y_train, y_val, y_test = get_train_val_test(X, y)
        
        # 标准化
        sc_X = StandardScaler()
        sc_y = StandardScaler()
        X_train_scaled = sc_X.fit_transform(X_train)
        y_train_scaled = sc_y.fit_transform(y_train)
        X_val_scaled = sc_X.transform(X_val)
        y_val_scaled = sc_y.transform(y_val)
        X_test_scaled = sc_X.transform(X_test)
        y_test_scaled = sc_y.transform(y_test)
        
        # 调整输入形状为 LSTM 格式 (samples, time_steps, features)
        X_train_reshaped = X_train_scaled.reshape(X_train_scaled.shape[0], look_back, 1)
        X_val_reshaped = X_val_scaled.reshape(X_val_scaled.shape[0], look_back, 1)
        X_test_reshaped = X_test_scaled.reshape(X_test_scaled.shape[0], look_back, 1)

        np.random.seed(1234)
        tf.random.set_seed(1234)      
    
        # 构建LSTM模型
        model = Sequential()
        model.add(LSTM(128,activation='relu',input_shape=(look_back, 1)))
        model.add(Dense(64,activation='relu')) # 添加全连接层提升非线性能力
        model.add(Dense(1)) 
        optimizer = Adam(learning_rate=0.001)
        model.compile(loss='mse',optimizer=optimizer)

        # 训练模型（添加早停）
        early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
        history = model.fit(
            X_train_reshaped, y_train_scaled.flatten(),
            epochs=100,
            batch_size=64,
            validation_data=(X_val_reshaped, y_val_scaled.flatten()),
            callbacks=[early_stop],
            verbose=0
        )

        # 预测并逆标准化
        y_pred = model.predict(X_test_scaled, verbose=0)
        y_pred =y_pred.reshape(-1,1)  #.reshape将一维数组转为二维
        y_pred = sc_y.inverse_transform(y_pred)
        pred_test.append(y_pred) 
        
    # 5. 合并各分量预测结果
    pred_test=np.array(pred_test)   #(9, 196, 1) 可以将其看作是包含9 个形状为 (196, 1) 的二维数组。
    y_pred_test = np.sum(np.hstack(pred_test), axis=1) #hstack后(196, 9)
    y_pred_test=y_pred_test.reshape(-1,1)
    #6. 评估
    X, y = createXy(series, look_back)
    X_train, X_val, X_test, y_train, y_val, y_test = get_train_val_test(X, y)
    sc_y = StandardScaler()
    y_train_scaled = sc_y.fit_transform(y_train)
    y_test_scaled = sc_y.transform(y_test)
    y_test_true = sc_y.inverse_transform(y_test_scaled)
    mape = np.mean(np.abs((y_test_true - y_pred_test) / y_test_true)) * 100
    rmse= sqrt(mean_squared_error(y_test_true,y_pred_test))
    mae=mean_absolute_error(y_test_true,y_pred_test)
    r2 = 1 - (np.sum((y_test_true - y_pred_test) ** 2) / np.sum((y_test_true - np.mean(y_test_true)) ** 2))

    print(f"MAPE:{mape:.3f}")
    print(f"RMSE:{rmse:.3f}")
    print(f"MAE:{mae:.3f}")
    print(f"r2:{r2:.3f}")
    
    return y_test_true,y_pred_test


# In[10]:










