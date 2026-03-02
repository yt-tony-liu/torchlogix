python train.py \
    --dataset cifar-10-3-thresholds \
    --architecture ClgnCifar10Small \
    --parametrization raw \
    --device cuda \
    --batch-size 128 \
    --eval-freq 500 \
    --num-iterations 20_000 \
    --seed 0 \
    --forward-sampling soft \
    --weight-init residual \
    --adaptive-discretization \
    --output results/clgnS_raw_soft/

python train.py \
    --dataset cifar-10-3-thresholds \
    --architecture ClgnCifar10Medium \
    --parametrization raw \
    --device cuda \
    --batch-size 128 \
    --eval-freq 500 \
    --num-iterations 20_000 \
    --seed 0 \
    --forward-sampling soft \
    --weight-init residual \
    --adaptive-discretization \
    --output results/clgnM_raw_soft/

python train.py \
    --dataset cifar-10-5-thresholds \
    --architecture ClgnCifar10Large \
    --parametrization raw \
    --device cuda \
    --batch-size 64 \
    --eval-freq 500 \
    --num-iterations 30_000 \
    --seed 0 \
    --forward-sampling soft \
    --weight-init residual \
    --adaptive-discretization \
    --output results/clgnL_raw_soft/