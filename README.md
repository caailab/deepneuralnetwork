# Deep neural networks for daily long-term traffic flow forecasting

This is a TensorFlow implementation of Deep Neural Network in the following paper:  
Licheng Qu, Wei Li, Wenjing Li, Dongfang Ma, and Yinhai Wang. [Daily long-term traffic flow forecasting based on a deep neural network](https://doi.org/10.1016/j.eswa.2018.12.031). Expert systems with applications, Volume 121, 1 May 2019, Pages 304-312.

The future trend of traffic flow is closely related to historical data and its contextual factors. These additional contextual factors, such as collection time, day of the week, weather, season, etc., usually make the traffic flow exhibit various periodic and random fluctuations, and play a very important role in the prediction of traffic flow.  

**This is the first time that only contextual factors have been used to predict traffic. Although this sounds mysterious and impossible, it has brought us surprising results.**  

## Optimization
In this article, the optimal number of neurons and hidden layers is $[15, 18, 22, 9, 5]$, you can use the ***traffic_dnn_train_try.py*** program to find your own DNN structure layer by layer. This is a very simple search routine, just trying to find a suitable next layer within a limited range based on the initially defined hidden layer.

## Training
Then, you can train the model by the program ***traffic_dnn_train.py*** which read the settings from the ***conf*** directory and save the model in the disk every ***100*** epochs. This mechanism can also recover the training from sidk in case something goes wrong. The default learning rate is $10^{-2}$ and the default ***epsilon*** is $10^{-6}$ which was used to early stop the trainning when the change of loss of current epoch is no longer decreasing (i.e.,  less than the ***epsilon***) in latest ***100*** epochs.

## Prediction
After the training, you can predict or evaluate the model by program ***traffic_dnn_train_predict.py***. This program use the same configuration file in ***conf*** directory. The Table 1 in the paper only exhibit the traffic between 6:00 and 22:00. If you want to evaluate the traffic of other time point (e.g., midnight), please comment the following code begining from line 123.
```
    # only evaluate the traffic between 6:00 and 22:00
    data_filter = np.logical_and(xtest[:, 3] >= 6, xtest[:, 3] < 22)
    xtest = xtest[data_filter]
    ytest = ytest[data_filter]
    test_time = test_time[data_filter]
```
Because the Table 1 of the paper only only evaluate the traffic between 6:00 and 22:00. 

## Requirements
These programs are developed based on TensorFlow 1.2. Numpy, pandas and Matplotlib are also essential scientific computing libraries. It is strongly recommended to use Anaconda to manage all software packages and use Pycharm IDE for code evaluation. If you have installed the latest TensorFlow 2.x, please use its compatibility mode.
```
    #import tensorflow as tf
    import tensorflow.compat.v1 as tf
    tf.compat.v1.disable_eager_execution()
```

## Citation
If you find this repository, e.g., the code or the datasets, useful in your research, please cite the following paper:
```
@article{qu2019daily,
  title={Daily long-term traffic flow forecasting based on a deep neural network},
  author={Qu, Licheng and Li, Wei and Li, Wenjing and Ma, Dongfang and Wang, Yinhai},
  journal={Expert systems with applications},
  volume={121},
  pages={304--312},
  year={2019},
  publisher={Elsevier}
}
```

***Note: This dataset should only be used for research.***
