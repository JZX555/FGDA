from data.vocabulary import VocabDict


vocab = VocabDict('.', '.')
# vocab.get_oov_rate('./debug/ctb.json', '../data/cws/zhuxian/test.zhuxian.wordpos')
# vocab.get_topk('./save/evaluation/share-adapter-zhuxian.sentences.0', './debug/domain.topfull.train', k=40000)

# generate a json file of source data
vocab._new_cws_vocab(
    '/data_path/CTB-train-data',
    '/data_path/CTB.json',
    max_num=5000000,
    pre_vocab_path=None
)

# generate a json file of OOV word
vocab._new_cws_vocab(
    '/data_path/unlabled_target_corpus',
    '/data_path/OOV_word.json',
    max_num=2000,
    threshold=0.8,
    pre_vocab_path=[
        # training vocab list
        '/data_path/CTB.json',
    ]
)

# generate a high-quality target domain pseudo corpus
vocab.apply_vocab(
    '/data_path/unlabled_target_corpus',
    # OOV vocab generated before
    '/data_path/OOV_word.json',
    train_vocab=[
        # training vocab list
        '/data_path/CTB.json',
    ],
    out_path='/data_path/next_rount_fine-grained_corpus',
    threshold=0.8,
    remain='/data/remain_unlabled_target_corpus'
)