# TIE-DGNN
## Paper and code
This is the code for the Paper: [Transition Information Enhanced Disentangled Graph Neural Networks for Session-based Recommendation](). We have implemented our methods in **Pytorch**.
## Usage
The code contains the datasets that has been processed, i.e., Tmall,Nowplaying,Last.fm etc.
Then you can run the file ````TIE-DGNN\main.py```` to train the model.  
For example:
First, ```` python build_graph.py --dataset Tmall````, then
```` python main.py --dataset Tmall````
The datasets we have uploaded have completed the build_graph steps.
## Requirements
Python 3 ；PyTorch 1.9.1
