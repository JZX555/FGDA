# ----- training base model ------
# gpu='0'
# name='base'
# mode='base'
# model='adapter'
# nohup python train.py \
#     --name=${name} \
#     --GPU=${gpu} \
#     --mode=${mode} \
#     --model=${model} \
#     > ./log/${name}.log 2>&1 &

# ----- training domain-mixed model ------
gpu='0'
name='base'
mode='base'
model='adapter'

nohup python domain_train.py \
    --name=${name} \
    --GPU=${gpu} \
    --mode=${mode} \
    --model=${model} \
    > ./log/${name}.log 2>&1 &