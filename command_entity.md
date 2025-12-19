python src/run_entity.py -c ./config/run_entity_transductive.yaml --dataset CoDExSmall --epochs 0 --bpe null --ckpt /T20030104/ynj/TRIX/ckpts/trix.pth --gpus [0]

python src/run_entity.py -c ./config/run_entity_transductive.yaml --dataset CoDExLarge --epochs 0 --bpe null --ckpt /T20030104/ynj/TRIX/ckpts/trix.pth --gpus [0]

python src/run_entity.py -c ./config/run_entity_transductive.yaml --dataset NELL995 --epochs 0 --bpe null --ckpt /T20030104/ynj/TRIX/ckpts/trix.pth --gpus [0]

python src/run_entity.py -c ./config/run_entity_transductive.yaml --dataset DBpedia100k --epochs 0 --bpe null --ckpt /T20030104/ynj/TRIX/ckpts/trix.pth --gpus [0]

python src/run_entity.py -c ./config/run_entity_transductive.yaml --dataset ConceptNet100k --epochs 0 --bpe null --ckpt /T20030104/ynj/TRIX/ckpts/trix.pth --gpus [0]

python src/run_entity.py -c ./config/run_entity_transductive.yaml --dataset NELL23k --epochs 0 --bpe null --ckpt /T20030104/ynj/TRIX/ckpts/trix.pth --gpus [0]

python src/run_entity.py -c ./config/run_entity_transductive.yaml --dataset YAGO310 --epochs 0 --bpe null --ckpt /T20030104/ynj/TRIX/ckpts/trix.pth --gpus [0]

python src/run_entity.py -c ./config/run_entity_transductive.yaml --dataset Hetionet --epochs 0 --bpe null --ckpt /T20030104/ynj/TRIX/ckpts/trix.pth --gpus [0]

python src/run_entity.py -c ./config/run_entity_transductive.yaml --dataset WDsinger --epochs 0 --bpe null --ckpt /T20030104/ynj/TRIX/ckpts/trix.pth --gpus [0]

python src/run_entity.py -c ./config/run_entity_transductive.yaml --dataset AristoV4 --epochs 0 --bpe null --ckpt /T20030104/ynj/TRIX/ckpts/trix.pth --gpus [0]

python src/run_entity.py -c ./config/run_entity_transductive_tail.yaml --dataset FB15k237_10 --epochs 0 --bpe null --ckpt /T20030104/ynj/TRIX/ckpts/trix.pth --gpus [0]

python src/run_entity.py -c ./config/run_entity_transductive_tail.yaml --dataset FB15k237_20 --epochs 0 --bpe null --ckpt /T20030104/ynj/TRIX/ckpts/trix.pth --gpus [0]

python src/run_entity.py -c ./config/run_entity_transductive_tail.yaml --dataset FB15k237_50 --epochs 0 --bpe null --ckpt /T20030104/ynj/TRIX/ckpts/trix.pth --gpus [0]

=================================================================================================

python src/run_entity.py -c ./config/run_entity_inductive.yaml --dataset FB15k237Inductive --version v1 --ckpt /T20030104/ynj/TRIX/ckpts/trix.pth --gpus [0] --epochs 0 --bpe null

python src/run_entity.py -c ./config/run_entity_inductive.yaml --dataset FB15k237Inductive --version v2 --ckpt /T20030104/ynj/TRIX/ckpts/trix.pth --gpus [0] --epochs 0 --bpe null

python src/run_entity.py -c ./config/run_entity_inductive.yaml --dataset FB15k237Inductive --version v3 --ckpt /T20030104/ynj/TRIX/ckpts/trix.pth --gpus [0] --epochs 0 --bpe null

python src/run_entity.py -c ./config/run_entity_inductive.yaml --dataset FB15k237Inductive --version v4 --ckpt /T20030104/ynj/TRIX/ckpts/trix.pth --gpus [0] --epochs 0 --bpe null

python src/run_entity.py -c ./config/run_entity_inductive.yaml --dataset WN18RRInductive --version v1 --ckpt /T20030104/ynj/TRIX/ckpts/trix.pth --gpus [0] --epochs 0 --bpe null

python src/run_entity.py -c ./config/run_entity_inductive.yaml --dataset WN18RRInductive --version v2 --ckpt /T20030104/ynj/TRIX/ckpts/trix.pth --gpus [0] --epochs 0 --bpe null

python src/run_entity.py -c ./config/run_entity_inductive.yaml --dataset WN18RRInductive --version v3 --ckpt /T20030104/ynj/TRIX/ckpts/trix.pth --gpus [0] --epochs 0 --bpe null

python src/run_entity.py -c ./config/run_entity_inductive.yaml --dataset WN18RRInductive --version v4 --ckpt /T20030104/ynj/TRIX/ckpts/trix.pth --gpus [0] --epochs 0 --bpe null

python src/run_entity.py -c ./config/run_entity_inductive.yaml --dataset NELLInductive --version v1 --ckpt /T20030104/ynj/TRIX/ckpts/trix.pth --gpus [0] --epochs 0 --bpe null

python src/run_entity.py -c ./config/run_entity_inductive.yaml --dataset NELLInductive --version v2 --ckpt /T20030104/ynj/TRIX/ckpts/trix.pth --gpus [0] --epochs 0 --bpe null

python src/run_entity.py -c ./config/run_entity_inductive.yaml --dataset NELLInductive --version v3 --ckpt /T20030104/ynj/TRIX/ckpts/trix.pth --gpus [0] --epochs 0 --bpe null

python src/run_entity.py -c ./config/run_entity_inductive.yaml --dataset NELLInductive --version v4 --ckpt /T20030104/ynj/TRIX/ckpts/trix.pth --gpus [0] --epochs 0 --bpe null

python src/run_entity.py -c ./config/run_entity_inductive.yaml --dataset ILPC2022 --version small --ckpt /T20030104/ynj/TRIX/ckpts/trix.pth --gpus [0] --epochs 0 --bpe null

python src/run_entity.py -c ./config/run_entity_inductive.yaml --dataset ILPC2022 --version large --ckpt /T20030104/ynj/TRIX/ckpts/trix.pth --gpus [0] --epochs 0 --bpe null

python src/run_entity.py -c ./config/run_entity_inductive.yaml --dataset HM --version 1k --ckpt /T20030104/ynj/TRIX/ckpts/trix.pth --gpus [0] --epochs 0 --bpe null

python src/run_entity.py -c ./config/run_entity_inductive.yaml --dataset HM --version 3k --ckpt /T20030104/ynj/TRIX/ckpts/trix.pth --gpus [0] --epochs 0 --bpe null

python src/run_entity.py -c ./config/run_entity_inductive.yaml --dataset HM --version 5k --ckpt /T20030104/ynj/TRIX/ckpts/trix.pth --gpus [0] --epochs 0 --bpe null

python src/run_entity.py -c ./config/run_entity_inductive.yaml --dataset HM --version indigo --ckpt /T20030104/ynj/TRIX/ckpts/trix.pth --gpus [0] --epochs 0 --bpe null

=======================================================================================================

python src/run_entity.py -c ./config/run_entity_inductive.yaml --dataset FBIngram --version 25 --ckpt /T20030104/ynj/TRIX/ckpts/trix.pth --gpus [0] --epochs 0 --bpe null

python src/run_entity.py -c ./config/run_entity_inductive.yaml --dataset FBIngram --version 50 --ckpt /T20030104/ynj/TRIX/ckpts/trix.pth --gpus [0] --epochs 0 --bpe null

python src/run_entity.py -c ./config/run_entity_inductive.yaml --dataset FBIngram --version 75 --ckpt /T20030104/ynj/TRIX/ckpts/trix.pth --gpus [0] --epochs 0 --bpe null

python src/run_entity.py -c ./config/run_entity_inductive.yaml --dataset FBIngram --version 100 --ckpt /T20030104/ynj/TRIX/ckpts/trix.pth --gpus [0] --epochs 0 --bpe null

python src/run_entity.py -c ./config/run_entity_inductive.yaml --dataset WKIngram --version 25 --ckpt /T20030104/ynj/TRIX/ckpts/trix.pth --gpus [0] --epochs 0 --bpe null

python src/run_entity.py -c ./config/run_entity_inductive.yaml --dataset WKIngram --version 50 --ckpt /T20030104/ynj/TRIX/ckpts/trix.pth --gpus [0] --epochs 0 --bpe null

python src/run_entity.py -c ./config/run_entity_inductive.yaml --dataset WKIngram --version 75 --ckpt /T20030104/ynj/TRIX/ckpts/trix.pth --gpus [0] --epochs 0 --bpe null

python src/run_entity.py -c ./config/run_entity_inductive.yaml --dataset WKIngram --version 100 --ckpt /T20030104/ynj/TRIX/ckpts/trix.pth --gpus [0] --epochs 0 --bpe null

python src/run_entity.py -c ./config/run_entity_inductive.yaml --dataset NLIngram --version 0 --ckpt /T20030104/ynj/TRIX/ckpts/trix.pth --gpus [0] --epochs 0 --bpe null

python src/run_entity.py -c ./config/run_entity_inductive.yaml --dataset NLIngram --version 25 --ckpt /T20030104/ynj/TRIX/ckpts/trix.pth --gpus [0] --epochs 0 --bpe null

python src/run_entity.py -c ./config/run_entity_inductive.yaml --dataset NLIngram --version 50 --ckpt /T20030104/ynj/TRIX/ckpts/trix.pth --gpus [0] --epochs 0 --bpe null

python src/run_entity.py -c ./config/run_entity_inductive.yaml --dataset NLIngram --version 75 --ckpt /T20030104/ynj/TRIX/ckpts/trix.pth --gpus [0] --epochs 0 --bpe null

python src/run_entity.py -c ./config/run_entity_inductive.yaml --dataset NLIngram --version 100 --ckpt /T20030104/ynj/TRIX/ckpts/trix.pth --gpus [0] --epochs 0 --bpe null

python src/run_entity.py -c ./config/run_entity_inductive.yaml --dataset WikiTopicsMT1 --version tax --ckpt /T20030104/ynj/TRIX/ckpts/trix.pth --gpus [0] --epochs 0 --bpe null

python src/run_entity.py -c ./config/run_entity_inductive.yaml --dataset WikiTopicsMT1 --version health --ckpt /T20030104/ynj/TRIX/ckpts/trix.pth --gpus [0] --epochs 0 --bpe null

python src/run_entity.py -c ./config/run_entity_inductive.yaml --dataset WikiTopicsMT2 --version org --ckpt /T20030104/ynj/TRIX/ckpts/trix.pth --gpus [0] --epochs 0 --bpe null

python src/run_entity.py -c ./config/run_entity_inductive.yaml --dataset WikiTopicsMT2 --version sci --ckpt /T20030104/ynj/TRIX/ckpts/trix.pth --gpus [0] --epochs 0 --bpe null

python src/run_entity.py -c ./config/run_entity_inductive.yaml --dataset WikiTopicsMT3 --version art --ckpt /T20030104/ynj/TRIX/ckpts/trix.pth --gpus [0] --epochs 0 --bpe null

python src/run_entity.py -c ./config/run_entity_inductive.yaml --dataset WikiTopicsMT3 --version infra --ckpt /T20030104/ynj/TRIX/ckpts/trix.pth --gpus [0] --epochs 0 --bpe null

python src/run_entity.py -c ./config/run_entity_inductive.yaml --dataset WikiTopicsMT4 --version sci --ckpt /T20030104/ynj/TRIX/ckpts/trix.pth --gpus [0] --epochs 0 --bpe null

python src/run_entity.py -c ./config/run_entity_inductive.yaml --dataset WikiTopicsMT4 --version health --ckpt /T20030104/ynj/TRIX/ckpts/trix.pth --gpus [0] --epochs 0 --bpe null

python src/run_entity.py -c ./config/run_entity_inductive.yaml --dataset Metafam --version null --ckpt /T20030104/ynj/TRIX/ckpts/trix.pth --gpus [0] --epochs 0 --bpe null

python src/run_entity.py -c ./config/run_entity_inductive.yaml --dataset FBNELL --version null --ckpt /T20030104/ynj/TRIX/ckpts/trix.pth --gpus [0] --epochs 0 --bpe null