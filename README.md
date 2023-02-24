# IGNR

Implementation for our work: Implicit Graphon Neural Representation


To run the single graphon learning task, go to folder IGNR, and run:\
python train_IGNR.py --use-pg 1

To run the parameterized graphon learning task, go to folder cIGNR, and run:\
python train_cIGNR.py --dataset '2ratio_rand' \
python train_cIGNR.py --dataset 'gCircle'



Requirements: \
pytorch >= 1.7 \
torch_geometric >= 2.1.0 \
networkx >=2.8.4 \
pot >=0.8.2 \
cython >= 0.29.32 \
joblib==0.15.1 \
einops >=0.6.0 
