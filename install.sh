conda=/usr/local/anaconda3/bin/conda
activate=/usr/local/anaconda3/bin/activate
deactivate=/usr/local/anaconda3/bin/deactivate

$conda create -y --no-default-packages -n ML_py37 python=3.7
source $activate  ML_py37 
pip install -r requirements.txt
conda $deactivate





