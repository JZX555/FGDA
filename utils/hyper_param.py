# encoding=utf-8
import os
cwd = os.getcwd()


class HyperParam:
    def __init__(self, mode, gpu=0, vocab=25000):
        self.gpu = gpu

        self.vocabulary_type = 'word'
        self.vocabulary_size = vocab
        self.general_path = None

        self.zhuxian_train = '/data_path/train.zhuxian.wordpos'
        self.zhuxian_dev = '/data_path/dev.zhuxian.wordpos'
        self.zhuxian_test = '/data_path/test.zhuxian.wordpos'
        self.zhuxian = '/data_path/ZX.unlabeled_corpus'

        if mode == 'debug':
            self.debug()
        elif mode == 'base':
            self.base()
        elif mode == 'subdomain-i':
            self.subdomain_i()
        else:
            raise NameError("choose the right mode.")

    def _print_items(self):
        print('\n'.join(['%s:%s' % item for item in self.__dict__.items()]))

    def base(self,
             batch_size=16,
             bert_size=768,
             buffer_size=640,
             d_word_vec=512,
             d_model=512,
             vae_size=128,
             d_inner_hid=2024,
             max_len=150,
             lr=2e-5,
             trainsets_len=1,
             lambda_edit=1,
             label_smoothing=0.1,
             use_bucket=True,
             shuffle=True,
             proj_share_weight=True,
             bridge_type='mlp',
             batching_key='samples',
             schedule_method='loss',
             finetune=True,
             warmup_steps=5000,
             update_cycle=1,
             dropout=0.1,
             src_drop=0.2,
             valid_freq=500,
             valid_threshold=1400,
             optim='adam'):
        self.batch_size = batch_size
        self.buffer_size = batch_size * 10

        self.bert_size = bert_size
        self.d_word_vec = d_word_vec
        self.d_model = d_model
        self.vae_size = vae_size
        self.d_inner_hid = d_inner_hid
        self.max_len = max_len
        self.lr = lr
        self.trainsets_len = trainsets_len
        self.lambda_edit = lambda_edit
        self.dropout = dropout
        self.src_drop = src_drop
        self.valid_freq = valid_freq
        self.valid_threshold = valid_threshold

        self.optim = optim

        self.label_smoothing = label_smoothing
        self.use_bucket = use_bucket
        self.shuffle = shuffle
        self.proj_share_weight = proj_share_weight
        self.batching_key = batching_key
        self.schedule_method = schedule_method
        self.finetune = finetune
        self.warmup_steps = warmup_steps
        self.bridge_type = bridge_type
        self.update_cycle = update_cycle

        self.label_vocab = '/data_path/cws_joint_pos_label_vocab.json'
        self.bert_path = '/bert_path/bert-base-chinese'
        self.pos_path = None

        self.train_data = ['/data_path/cws_joint_pos.train']
        self.dev_data = '/data_path/cws_joint_pos.dev'
        self.test_data = '/data_path/cws_joint_pos.test'

    def subdomain_i(self, trainsets_len='the value of i + 1 or the lengths of train_data'):
        self.base()
        self.trainsets_len = trainsets_len
        self.train_data = [
            '/data_path/subdomain_i.train', '/data_path/subdomain_(i-1).train', '...',
            '/data_path/subdomain_1.train', '/data_path/cws_joint_pos.train'
        ]
        self.ZX_corpus = 'the remaining data after obtaining subdomain_i.train by use the method vocab.apply_vocab() in prepocessing.py'
