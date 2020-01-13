# kmeans
k-means算法原理以及应用

简单地说，K 近邻算法采用测量不同特征值之间的距离方法进行分类。它具有的优缺点如下：

优点：精度高、对异常值不敏感、无数据输入假定。
缺点：计算复杂度高、空间复杂度高。
K 近邻算法适用数据范围为：数值型和标称型。

K 近邻算法的工作原理是：存在一个样本数据集合，也称作训练样本集，并且样本集中每个数据都存在标签，即我们知道样本集中每一数据与所属分类的对应关系。

输入没有标签的新数据后，将新数据的每个特征与样本集中数据对应的特征进行比较，然后算法提取样本集中特征最相似数据（最近邻）的分类标签。一般来说，我们只选择样本数据集中前 kk 个最相似的数据，这就是 K 近邻算法中 kk 的出处，通常 kk 是不大于 20 的整数。最后，选择 kk 个最相似数据中出现次数最多的分类，作为新数据的分类。

K 近邻算法的一般流程

收集数据：可以使用任何方法。
准备数据：距离计算所需要的数值，最好是结构化的数据格式。
分析数据：可以使用任何方法。
训练算法：此步骤不适用于 K 近邻算法。
测试算法：计算错误率。
使用算法：首先需要输入样本数据和结构化的输出结果，然后运行K 近邻算法判定输入数据分别属于哪个分类，最后应用对计算出的分类执行后续的处理。

数据集地址：http://labfile.oss.aliyuncs.com/courses/777/digits.zip
解压后digits 目录下有两个文件夹，分别是:
trainingDigits：训练数据，1934 个文件，每个数字大约 200 个文件。
testDigits：测试数据，946 个文件，每个数字大约 100 个文件。
每个文件中存储一个手写的数字，文件的命名类似 0_7.txt，第一个数字 0 表示文件中的手写数字是 0，后面的 7 是个序号。

文件目录
data.py:处理数据,将数据处理成向量
digits:数据集
model.py:算法实现
test:在测试集上测试结果

算法核心部分：计算「距离」
算法实现过程：
计算已知类别数据集中的点与当前点之间的距离；
按照距离递增次序排序；
选取与当前点距离最小的 k 个点；
确定前 k 个点所在类别的出现频率；
返回前 k 个点出现频率最高的类别作为当前点的预测分类。