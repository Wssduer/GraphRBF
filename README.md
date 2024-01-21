# GraphRBF  
使用了等变图神经网络EGNN与径向基神经网络RBFNN的多种蛋白质结合位点预测模型，主要研究了蛋白质-蛋白质/RNA/DNA相互作用位点的预测。  
A variety of protein binding site prediction models using the isovariogram neural network EGNN and radial basis neural network RBFNN were used, focusing on the prediction of protein-protein/RNA/DNA interaction sites.  

![](https://github.com/Wssduer/GraphRBF/blob/main/GraphRBF-main/IMG/GraphRBF_flame.jpg "Overview of GraphRBF")  
## 模型介绍Introduction  
我们设计了GraphRBF，一种端到端可解释的分层几何深度学习模型。GraphRBF基于氨基酸邻居的空间化学排列，构建了目标残基的局部邻居表示，确保了平移和旋转不变性。之后，我们将等变图神经网络和径向基函数神经网络相结合，直接从蛋白质的局部三维结构中提取特征，嵌入氨基酸的隐层特征表示。  
In this paper, we introduce GraphRBF, an end-to-end interpretable hierarchical geometric deep learning model. GraphRBF constructs a local neighborhood representation for target residues, based on the spatial-chemical arrangement of amino acid neighbors, ensuring translational and rotational invariance. We combine equivariant graph neural network and radial basis function neural network to directly extract features from the local three-dimensional structure of proteins, embed the latent representation of amino acid.
