# GraphRBF 
GraphRBF is a state-of-the-art protein-protein/nucleic acid interaction site prediction model built by enhanced graph neural networks and prioritized radial basis function neural networks. 
This project serves users to use our software to directly predict protein binding sites or train our model on a new database.  

![](https://github.com/Wssduer/GraphRBF/blob/main/GraphRBF-main/IMG/GraphRBF_flame.jpg "Overview of GraphRBF")  
## Description 
  Identification of protein-protein and protein-nucleic acid binding sites provides insights into biological processes related to protein functions and technical guidance for disease diagnosis and drug design. However, accurate predictions by computational approaches remain highly challenging due to the limited knowledge of residue binding patterns. The binding pattern of a residue should be characterized by the spatial distribution of its neighboring residues combined with their physicochemical information interaction, which yet can not be achieved by previous methods. Here, we design GraphRBF, a hierarchical geometric deep learning model to learn residue binding patterns from big data. To achieve it, GraphRBF describes physicochemical information interactions by designing an enhanced graph neural network and characterizes residue spatial distributions by introducing a prioritized radial basis function neural network. After training and testing, GraphRBF shows great improvements over existing state-of-the-art methods and strong interpretability of its learned representations. 
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
