我基于神经网络实现了任意位数的加法运算。算法特点如下：1、采用神经网络进行计算，只用了线性层和relu激活函数（包含与这两种等价的计算）2、输入可以是任意位正整数（位数超出浮点数表示范围的除外）3、能根据计算的复杂程度自主调节推理次数，且推理次数没有上限。
模型构建及训练过程：基于已有的二进制加法算法，通过手工填写线性层参数的方式，用神经网络模拟了加法计算的过程。
演示视频如下：【基于神经网络实现任意位数的加法，模拟了人逻辑推理的过程】 https://www.bilibili.com/video/BV1WVBtYpEFG/?share_source=copy_web&vd_source=48cd81ec5e50e5fa03837b97c7354ee8



I build a adder by neural network ,which can work out with any length of number.Features:
    1、using NN(neural network) to calculate,it only uses linear and relu(containing the equivalent operators).
    2、the length of inputs is not limited.
    3、the number of inferences is dynamically adjusted based on input,without upper limited.
Way to build:Based on existing binary addition algorithms,I simulate the process of addition calculation using NN,by artificially filling in the linear layer parameters.
Demo video:【基于神经网络实现任意位数的加法，模拟了人逻辑推理的过程】 https://www.bilibili.com/video/BV1WVBtYpEFG/?share_source=copy_web&vd_source=48cd81ec5e50e5fa03837b97c7354ee8