
/ddnstor/imu_panjiahui/dataset/SHdomainShcAssisted/generated_data_cir_uniform_9/：是一个t60随机，但是SNR取int的数据集
/ddnstor/imu_panjiahui/dataset/SHdomainShcAssisted/generated_data_cir_uniform_9_random/:是一个t60随机，SNR随机的数据集
数据生成：
1、运行genRIR_*.py 生成对应的rir文件
2、运行gen_mix_*.py 生成对应的混合语音
3、运行tools.py 第8个，随机选择8W条训练，4K条测试

训练：
运行train_igcrn.py，模型保存在：Lock_baseline_random

最后效果都不如mini数据集上的结果
最终数据集，采用的是/ddnstor/imu_panjiahui/dataset/SHdomainNew/TIMIT/generated_data_cir_uniform_9：是一个T60取整，SNR取整的数据集


