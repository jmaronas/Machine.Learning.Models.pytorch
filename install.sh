conda=/usr/local/anaconda3/bin/conda
activate=/usr/local/anaconda3/bin/activate
deactivate=/usr/local/anaconda3/bin/deactivate

$conda create -y --no-default-packages -n ML_py37 python=3.7
source $activate  ML_py37 
pip install -r requirements.txt
source $deactivate

cd ~
git clone https://github.com/jmaronas/pytorch_library.git
cd pytorch_library
git checkout b71b16c79a945064793fa53bf7b05f3f6760dbb2




