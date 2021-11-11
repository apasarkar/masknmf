$PYTHON setup.py install --single-version-externally-managed --record record.txt

python -m pip install 'git+https://github.com/j-friedrich/OASIS.git@f3ae85e1225bfa4bfe098a3f119246ac1e4f8481#egg=oasis'

python -m pip install 'git+https://github.com/facebookresearch/detectron2.git'