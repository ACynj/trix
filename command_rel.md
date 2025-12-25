
python ./src/run_relation.py -c ./config/run_relation_transductive_mech.yaml --dataset FB15k237_10 --epochs 0 --bpe null --ckpt /T20030104/ynj/TRIX/ckpts/rel_5.pth --gpus [0]

python ./src/run_relation.py -c ./config/run_relation_transductive_mech.yaml --dataset FB15k237_20 --epochs 0 --bpe null --ckpt /T20030104/ynj/TRIX/ckpts/rel_5.pth --gpus [0]

python ./src/run_relation.py -c ./config/run_relation_transductive_mech.yaml --dataset FB15k237_50 --epochs 0 --bpe null --ckpt /T20030104/ynj/TRIX/ckpts/rel_5.pth --gpus [0]
