&emsp;&emsp;前段时间由于项目需要用到蚁群聚类，于是自学了一下LF算法。网上这部分的资料很少，找到的几篇论文中，我觉得最有用的是LF算法发明者写的那一篇，其他的几篇中文论文主要是提到了一些公式上的改进，但是我对于其实验数据中的迭代次数和聚类的准确率表示怀疑。基于以上几点，打算写一篇blog介绍一下比较冷门的这个算法，也希望有大神能指出代码中存在的不足。
## LF算法简介
&emsp;&emsp;蚁群聚类算法主要分为两种，一种是基于模拟蚂蚁觅食的行为发明的蚁群聚类算法，另一种是基于蚂蚁搬运同伴尸体的行为发明的聚类算法，也即这篇文章主要关注的LF算法。第一种算法主要是提出了蚂蚁觅食过程中会在行进路线上留下**信息素**，为后来的蚂蚁提供信息，而新来的蚂蚁沿同样的路线前进，便会加大信息素的浓度，促使更多蚂蚁选择这条道路。基于这一模式，我们可以从中提炼出一个正反馈循环，因此有助于促进算法的收敛。而LF算法的思想则是，蚂蚁在处理目标点时，将以一定的概率**搬起**或**放下**目标点，而这一概率取决于该点与蚂可视范围内的所有样本点的相似度。相似度越高的点越不容易被拿起来，越容易被放下；相似度越低的点越容易被拿起来，越难被放下。基于这样的思路，只要有**表现优秀**的相似度计算公式，在**足够长的迭代次数**后，我们就可以获得有关所有样本点的聚类信息。
## 算法流程及简单改进
&emsp;&emsp;我们首先来看LF算法最最朴素的版本：
 1. 设有$sampleNum$个样本点，$antNum$只蚂蚁，算法一共迭代$iterNum$次，对数据进行归一化处理
 2. 将所有样本点和蚂蚁随机投影到$gridNum \times gridNum$大小的网格上
 3. 对第$iter$次迭代，考虑所有的蚂蚁：
	 3.0 考虑当前为第$i$只蚂蚁
	 3.1 若蚂蚁$i$有负载，将其随机移动到网格上相邻的某一点，计算当前位置上的放下概率$P_d$，生成$[0,1]$上均匀分布的随机数$R$，若$R>P_d$则放下蚂蚁$i$负载的样本点
	 3.2 若蚂蚁$i$无负载，将蚂蚁随机移动到一个位置，若当前位置有样本点，计算当前位置上的拾起概率$P_u$，生成$[0,1]$上均匀分布的随机数$R$，若$R>P_u$则蚂蚁$i$拾起该位置上的样本点
4. 将网格上所有**网格距离**小于蚂蚁可视半径$r$的样本点归为一个聚类，计算聚类中心和半径

&emsp;&emsp;以上就是LF算法最基础的版本，第一眼看到这个算法流程其实就能看出很多可以改进的地方，比如将蚂蚁的初始投放，和空载的蚂蚁的移动规则改为非完全随机的，直接将其移动到某个未被拾起的样本点处，可以减少许多无意义的蚂蚁移动。诸如此类的小优化可以在一定程度上提高程序的效率。
&emsp;&emsp;同样地，这个算法的缺点也很明显。首先，算法随机的部分太多，导致算法在实际执行的时候，时间上不会有一个太理想的表现，因为算法必须要执行足够多的迭代次数，才能体现出答案的收敛趋向。其次，由于随机带来的影响，算法调试起来可能也会有些困难，需要在写的时候足够细心。最重要的一点在于，由于算法捡起、放下样本点都具有随机性，那么我们设计这两样概率的时候需要考虑相似度阈值的设定，相似度低于多少的时候认定一个样本点是应该被捡起的？相似度高于多少的时候认定一个样本点是应该被放下的？这些都涉及到超参数的调整。而在实际运行的过程中，我们也可以感受到LF算法对于超参数相当敏感，参数的调整难以分析。
&emsp;&emsp;最后是关于网格的数量，可以想象的到，如果网格太小或太大，整个地图过于拥挤或空旷时，都会使蚂蚁找到一个合适的位置放下样本点这一过程变得相当困难。因此也需要做出合适的选择。

## 关于相似度和概率$P_u,P_d$
&emsp;&emsp;根据论文，在衡量两个样本点之间的相似度时，我们采用欧几里得距离作为其相异度的衡量，$d(i,j)$越大，表明两个样本点$data_i,data_j$之间差的越远。那么在蚂蚁的可视半径内，某个样本点与其当前所在区域的相似度可以定义为$$f(i)=\frac{1}{S^2}\Sigma_{0 < gridDis(i,j) \leq R}[1-\frac{d(i,j)}{\alpha}]$$其中$gridDis(i,j)$表示两个样本在网格上的距离；$\alpha$为区分度，用于控制聚类的精细程度；$\frac{1}{S^2}$是区域内网格点数，用于控制住相似度数值的大小。
&emsp;&emsp;接下来我们就可以给出$P_u,P_d$的定义
$$P_u(i) = (\frac{K_1}{K_1+f(i)})^2$$
$$P_d(i)=\begin{cases}
2f(i) & f(i)<K_2 \\
1 & otherwise \\
\end{cases}$$
$K_1,K_2$是两个参数，用于控制拾起及放下的难易程度。
## 关于参数取值
&emsp;&emsp;由查阅到的论文中的经验结论，我们需要取网格数约为样本数的10倍，因此$gridNum \approx \sqrt{10\times sampleNum}$。蚂蚁数要比样本数低一个量级，如果数目太少会导致需要的迭代次数增加，如果数目太多会导致一次迭代中移动的样本太多，从而影响样本放置的过程。
&emsp;&emsp;在我做的模拟中，取各参数为$\alpha = 0.2,K_1=0.1,K_2=0.15,R=2$时效果较好。

## 关于优化
&emsp;&emsp;对于蚁群算法的优化，论文中主要给出的几种方法分别为：加权、调整步速、增加记忆。
### 加权
&emsp;&emsp;对样本的每一个维度，取其权值为$$w_i = \frac{\sum^{sampleNum}_{j,k=1}|data[j][i]-data[k][i]|}{\sum_{i=1}^{featureNum}\sum^{sampleNum}_{j,k=1}|data[j][i]-data[k][i]|}$$
由实践，给样本的不同维度赋权后可以很大地加快聚类的收敛，仅1万次迭代就能达到朴素LF算法5万次以上的聚类效果。
### 调整步速及记忆
&emsp;&emsp;原论文主要给出的优化方法是调整蚂蚁的步速和增加短时记忆。为了加快迭代时收敛的速度，可以让有负载的蚂蚁每次以$V$的步长移动，在迭代的过程中不断地调整$V$即可实现在迭代开始时加速聚拢，接近结束时细微调整的效果。
&emsp;&emsp;而增加短期记忆，则是每次让蚂蚁负载着当前的样本点移动到之前放下过的样本点中，与当前样本最相似的点的附近，以求加速收敛过程。这个记忆可以是每只蚂蚁独立的记忆，也可以是所有蚂蚁全局共享记忆，由提出LF算法的原论文，增加记忆这一优化明显提高了聚拢的程度，得到了很高的准确率。

## 效果对比
![原数据](https://img-blog.csdnimg.cn/20190522235229494.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3UwMTAxNjQ3OTE=,size_16,color_FFFFFF,t_70)
![无权重1w次迭代](https://img-blog.csdnimg.cn/20190522235250377.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3UwMTAxNjQ3OTE=,size_16,color_FFFFFF,t_70)
![加权1w次迭代](https://img-blog.csdnimg.cn/20190522235307732.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3UwMTAxNjQ3OTE=,size_16,color_FFFFFF,t_70)
&emsp;&emsp;从上到下分别为原数据，无权重1w次迭代，加权1w次迭代。可以看到，加权之后样本点聚类效果更好，但是依然不够好，这也是LF的算法存在的问题——迭代次数不够时，同一聚类可能被分成多类。要解决这个问题，就需要增加迭代次数，或者利用步速和记忆来调整迭代过程。
[代码和资料在这](https://github.com/YYXKirito/ClusterAlgorithm/tree/master/Ant/LF)
