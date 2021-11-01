# evaluation the base model
# gpu='0'
# name='base'
# mode='base'
# model='adapter'

# evaluation the domain-mixed model
gpu='0'
name='base'
mode='base'
model='domain'

python evaluation.py --restore \
    --name=${name} \
    --GPU=${gpu} \
    --mode=${mode} \
    --model=${model}