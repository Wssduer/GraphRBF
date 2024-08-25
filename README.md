# GraphRBF 

GraphRBF is a state-of-the-art protein-protein/nucleic acid interaction site prediction model built by enhanced graph neural networks and prioritized radial basis function neural networks. 
This project serves users to use our software to directly predict protein binding sites or train our model on a new database.  

![](https://github.com/Wssduer/GraphRBF/blob/main/GraphRBF-main/IMG/GraphRBF_flame.jpg "Overview of GraphRBF")  

## 1 Description 

  Identification of protein-protein and protein-nucleic acid binding sites provides insights into biological processes related to protein functions and technical guidance for disease diagnosis and drug design. However, accurate predictions by computational approaches remain highly challenging due to the limited knowledge of residue binding patterns. The binding pattern of a residue should be characterized by the spatial distribution of its neighboring residues combined with their physicochemical information interaction, which yet can not be achieved by previous methods. Here, we design GraphRBF, a hierarchical geometric deep learning model to learn residue binding patterns from big data. To achieve it, GraphRBF describes physicochemical information interactions by designing an enhanced graph neural network and characterizes residue spatial distributions by introducing a prioritized radial basis function neural network. After training and testing, GraphRBF shows great improvements over existing state-of-the-art methods and strong interpretability of its learned representations. 
  
## 2 Installation  

### 2.1 System requirements
For prediction process, you can predict functional binding residues from a protein structure within a few minutes with CPUs only. However, for training a new deep model from scratch, we recommend using a GPU for significantly faster training.
To use GraphRBF with GPUs, you will need: cuda >= 11.7, cuDNN.
### 2.2 Create an environment

We highly recommend to use a virtual environment for the installation of GraphRBF and its dependencies.

A virtual environment can be created and (de)activated as follows by using conda(https://conda.io/docs/):

        # create
        $ conda create -n GraphRBF python=3.8
        # activate
        $ source activate GraphRBF
        # deactivate
        $ source deactivate
        
### 2.3 Install GraphRBF dependencies
Note: Make sure environment is activated before running each command.

#### 2.3.1 Install requirements
Install pytorch 2.0.1 (For more details, please refer to https://pytorch.org/)

        For linux:
        # CUDA 11.7
        $ pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2
        # CPU only
        $ pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cpu
Install torch_geometric 2.3.1 (For more details, please refer to https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html)

        $ pip install torch-scatter -f https://data.pyg.org/whl/torch-2.0.0+cu117.html
        $ pip install torch-scatter torch-sparse -f https://data.pyg.org/whl/torch-2.0.0+cu117.html
        $ pip install torch-cluster -f https://data.pyg.org/whl/torch-2.0.0+cu117.html
        $ pip install torch_geometric==2.3.1
Install other requirements

        $ pip install torchnet==0.0.4
        $ pip install tqdm
        $ pip install prettytable

Note: Typical install requirements time on a "normal" desktop computer is 10 minutes.
    
#### 2.3.2 Install the bioinformatics tools
Install blast+ for extracting PSSM (position-specific scoring matrix) profiles

    To install ncbi-blast-2.8.1+ and download NR database (ftp://ftp.ncbi.nlm.nih.gov/blast/db/) for psiblast, please refer to BLAST® Help (https://www.ncbi.nlm.nih.gov/books/NBK52640/).
    Set the absolute paths of blast+ and NR databases in the script "GraphRBF.py".
Install HHblits for extracting HMM profiles

    To install HHblits and download uniclust30_2018_08 (http://wwwuser.gwdg.de/~compbiol/uniclust/2018_08/uniclust30_2018_08_hhsuite.tar.gz) for HHblits, please refer to https://github.com/soedinglab/hh-suite.
    Set the absolute paths of HHblits and uniclust30_2018_08 databases in the script "scripts/prediction.py".
    
Install DSSP for extracting SS (Secondary structure) profiles

    DSSP is contained in "scripts/dssp", and it should be given executable permission by:
        $ chmod +x scripts/dssp
 Note: Difference versions of blast+, HHblits and their databases may result in slightly different PSSM and HMM profiles, leading to slight different predictions. Typical download databases and install bioinformatics tools time on a "normal" desktop computer is 10 hours.
    
## 3 Usage   

### 3.1 Predict functional binding residues from a protein structure(in PDB format) based on trained deep models
We have packaged data extraction: data_io_guassian.py, model training: training_guassian.py, GraphRBF model: GN_model_guassian_posemb.py, the validation module: valid_metrices.py, and the prediction code: GraphRBF.py.  
First, install the environment as described above, and after that, use the code from the prediction command 'prediction code.log' file in the folder:  


    cd ../GraphRBF-main  
    python GraphRBF.py --querypath ../GraphRBF-main/example --filename 1ddl --chain A --ligands DNA,RNA,P  
  
    Command list： 
    --querypath   The path of query structure  
    --filename    The file name of the query structure（we need user to upload its pdb(1ddl_A.pdb) and pssm and hmm file of each chain(1ddl_A.pssm and 1ddl_A.hmm)）  
    --chain       The name of each protein chain(for single chain A, and more chains AB or ABC)
    --ligands     Ligand types. You can choose from DNA,RNA,P.
  
### 3.2  Train a new deep model from scratch

#### 3.2.1 Download the datasets used in GraphRBF.

Donload the PDB files and the feature files (the PSSM profiles, HMM profiles, and the DSSP profiles) from http: and store the PDB files in the path of the corresponding data.

Example:

	The PDB files of DNA data should be stored in ../Datasets/PDNA/PDB, and the features file shuld be stored in ../Datasets/customed_data/PDNA/feature.


#### 3.2.2 Generate the training, validation and test data sets from original data sets

    Example:
        $ cd ../GraphRBF-main/scripts
        # demo 1
        $ python data_io_guassian.py --ligand P --features RT,PSSM,HMM,SS,AF --context_radius 20
        # demo 2
        $ python data_io_guassian.py --ligand RNA --features PSSM,HMM,SS,AF --context_radius 15

    Output:
    The data sets are saved in ../Datasets/P{ligand}/P{ligand}_dist{context_radius}_{featurecode}.

    Note: {featurecode} is the combination of the first letter of {features}.
    Expected run time for the demo 1 and demo 2 on a "normal" desktop computer are 30 and 40 minutes, respectively.

    The list of commands:
    --ligand            A ligand type. It can be chosen from DNA,RNA,P.
    --features          Feature groups. Multiple features should be separated by commas. You can combine features from RT(residue type), PSSM, HMM, SS(secondary structure) and AF(atom features).(default=RT,PSSM,HMM,SS,AF)
    --context_radius    Radius of structure context.

#### 3.2.3 Train the deep model

    Example:
        $ cd ../GraphRBF-main/scripts
        # demo 1
        $ python training_guassian.py --ligand P --features RT,PSSM,HMM,SS,AF --context_radius 20 --edge_radius 10 --gnn_steps 2
        # demo 2
        $ python training_guassian.py --ligand RNA --features PSSM,HMM,SS,AF --context_radius 15 --edge_radius 10 --gnn_steps 1

    Output:
    The trained model is saved in ../Datasets/P{ligand}/checkpoints/{starttime}.
    The log file of training details is saved in ../Datasets/P{ligand}/checkpoints/{starttime}/training.log.

    Note: {starttime} is the time when training.py started be executed.
    Expected run time for demo 1 and demo 2 on a "normal" desktop computer with a GPU are 30 and 12 hours, respectively.

    The list of commands:
    --ligand            A ligand type. It can be chosen from DNA,RNA,P.
    --features          Feature groups. Multiple features should be separated by commas. You can combine features from RT(residue type), PSSM, HMM, SS(secondary structure) and AF(atom features).(default=RT,PSSM,HMM,SS,AF)
    --context_radius    Radius of structure context.
    --edge_radius       Radius of the neighborhood of a node. It should be smaller than radius of structure context.(default=10)
    --hidden_size       The dimension of encoded edge, node and graph feature vector.(default=256)
    --gnn_steps         The number of GNN-blocks.(default=2)
    --lr                Learning rate for training the deep model.(default=0.0001)
    --batch_size        Batch size for training deep model.(default=256)
    --epoch             Training epochs.(default=60)

### 4 Frequently Asked Questions
(1) If the script is interrupted by "Segmentation fault (core dumped)" when torch of CUDA version is used, it may be raised because the version of gcc (our version of gcc is 5.5.0) and you can try to set CUDA_VISIBLE_DEVICES to CPU before execute the script to avoid it by:
        $ export CUDA_VISIBLE_DEVICES="-1"
(2) If your CUDA version is not 11.7, please refer to the homepages of Pytorch(https://pytorch.org/) and torch_geometric (https://pytorch-geometric.readthedocs.io/en/latest/) to make sure that the installed dependencies match the CUDA version. Otherwise, the environment could be problematic due to the inconsistency.

