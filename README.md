# GraphRBF 
GraphRBF is a state-of-the-art protein-protein/nucleic acid interaction site prediction model built by enhanced graph neural networks and prioritized radial basis function neural networks. 
This project serves users to use our software to directly predict protein binding sites or train our model on a new database.  

![](https://github.com/Wssduer/GraphRBF/blob/main/GraphRBF-main/IMG/GraphRBF_flame.jpg "Overview of GraphRBF")  
## Description 
  Identification of protein-protein and protein-nucleic acid binding sites provides insights into biological processes related to protein functions and technical guidance for disease diagnosis and drug design. However, accurate predictions by computational approaches remain highly challenging due to the limited knowledge of residue binding patterns. The binding pattern of a residue should be characterized by the spatial distribution of its neighboring residues combined with their physicochemical information interaction, which yet can not be achieved by previous methods. Here, we design GraphRBF, a hierarchical geometric deep learning model to learn residue binding patterns from big data. To achieve it, GraphRBF describes physicochemical information interactions by designing an enhanced graph neural network and characterizes residue spatial distributions by introducing a prioritized radial basis function neural network. After training and testing, GraphRBF shows great improvements over existing state-of-the-art methods and strong interpretability of its learned representations. 
## Installation  

## Usage   

### Predict functional binding residues from a protein structure(in PDB format) based on trained deep models
We have packaged data extraction: data_io_guassian.py, model training: training_guassian.py, GraphRBF model: GN_model_guassian_posemb.py, the validation module: valid_metrices.py, and the prediction code: GraphRBF.py.  
First, install the environment as described above, and after that, use the code from the prediction command 'prediction code.log' file in the folder:  
  `cd ../GraphRBF-main`  
  `python GraphRBF.py --querypath ../GraphRBF-main/example --filename 1ddl_A --ligands DNA,RNA,P`  
Command list：  
  --querypath   The path of query structure  
  --filename    The file name of the query structure（single chain protein, we need user to upload its pdb,pssm and hmm file）  
  --ligands     Ligand types. You can choose from DNA,RNA,P.
