# GraphRBF  
GraphRBF是一个先进的蛋白质-蛋白质/核酸相互作用位点预测模型，结合了等变图神经网络EGNN与径向基神经网络RBFNN。
这个项目服务用户使用打包的软件直接预测蛋白质结合位点或重新训练新数据集。  
GraphRBF is a state-of-the-art protein-protein/nucleic acid interaction site prediction model built by equivariant graph neural networks and radial basis neural networks. 
This project serves users to use our software to directly predict protein binding sites or train our model on a new database.  
![](https://github.com/Wssduer/GraphRBF/blob/main/GraphRBF-main/IMG/GraphRBF_flame.jpg "Overview of GraphRBF")  
## 模型介绍Description 
我们设计了GraphRBF，一种端到端可解释的分层几何深度学习模型。GraphRBF基于氨基酸邻居的空间化学排列，构建了目标残基的局部邻居表示，确保了平移和旋转不变性。之后，我们将等变图神经网络和径向基函数神经网络相结合，直接从蛋白质的局部三维结构中提取特征，嵌入氨基酸的隐层特征表示。  
In this paper, we introduce GraphRBF, an end-to-end interpretable hierarchical geometric deep learning model. GraphRBF constructs a local neighborhood representation for target residues, based on the spatial-chemical arrangement of amino acid neighbors, ensuring translational and rotational invariance. We combine equivariant graph neural network and radial basis function neural network to directly extract features from the local three-dimensional structure of proteins, embed the latent representation of amino acid.  
## 安装方法Installation  
## 使用手则Usage  
### 使用训练好的模型预测蛋白质结合位点  
### Predict functional binding residues from a protein structure(in PDB format) based on trained deep models
我们在GraphRBF-main文件夹中打包了数据提取data_io_guassian.py、模型训练training_guassian.py、GraphRBF模型GN_model_guassian_posemb.py、验证模块valid_metrices.py以及预测代码GraphRBF.py。  
We have packaged data extraction: data_io_guassian.py, model training: training_guassian.py, GraphRBF model: GN_model_guassian_posemb.py, the validation module: valid_metrices.py, and the prediction code: GraphRBF.py.  
首先按照上述要求安装环境，之后使用文件夹中‘预测命令prediction code.log’文件中的代码：  
First, install the environment as described above, and after that, use the code from the prediction command 'prediction code.log' file in the folder:  
  `cd ../GraphRBF-main`  
  `python GraphRBF.py --querypath ../GraphRBF-main/example --filename 1ddl_A --ligands DNA,RNA,P`  
命令列表：  
  --querypath   蛋白质文件路径The path of query structure  
  --filename    蛋白质名称（单链蛋白，需提供对应的pdb、pssm和hmm文件）The file name of the query structure（single chain protein, we need user to upload its pdb,pssm and hmm file）  

  --ligands     预测配体类型，可以从P，RNA，DNA中选择Ligand types. You can choose from DNA,RNA,P.
