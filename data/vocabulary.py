import json
from .tokenizer import Tokenizer, _Tokenizer


class VocabDict(object):
    def __init__(self, data_path, out_path, max_n_words=35000):
        self.data_path = data_path
        self.out_path = out_path

        self.max_n_words = max_n_words

    def generate_vocabfile(self, data_path=None, out_path=None):
        dic = dict()
        idx = 0

        if data_path:
            self.data_path = data_path
        if out_path:
            self.out_path = out_path

        with open(self.data_path, 'r', encoding='utf-8') as f:
            for line in f.readlines():
                words = line.strip().split()
                for w in words:
                    if w in dic:
                        dic[w] += 1
                    else:
                        dic[w] = 1

        items = sorted(dic.items(), key=lambda d: d[1], reverse=True)[:self.max_n_words]

        dic.clear()        
        for word, num in items:
            dic[word] = (idx, num)
            idx += 1

        with open(self.out_path + '.json', 'w', encoding='utf-8') as f:
            json.dump(dic, f, indent=4, ensure_ascii=False)

    def to_json(self, vocab_path, out_path):
        dic = {}

        print('begin reading the file')
        with open(vocab_path, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f.readlines()):
                words = line.strip().split()
                dic[words[0]] = (i, words[1])

        print('read successful!')
        print('begin trans')

        with open(out_path + '.json', 'w', encoding='utf-8') as f:
            json.dump(dic, f, indent=4, ensure_ascii=False)

        print('trans successful')

    def merge_vocab(self, vocab_paths, out_path, type='dic'):
        dic = {}

        print('begin reading the file')
        for vocab_path in vocab_paths:
            with open(vocab_path, 'r', encoding='utf-8') as f:
                if type == 'dic':
                    for i, line in enumerate(f.readlines()):
                        words = line.strip().split()
                        if words[0] in dic:
                            dic[words[0]] += int(words[1])
                        else:
                            dic[words[0]] = int(words[1])
                elif type == 'json':
                    datas = json.load(f)
                    for w, t in datas.items():
                        if w in dic:
                            dic[w] += int(t[1])
                        else:
                            dic[w] = int(t[1])                        

        print('read successful!')
        print('begin trans')

        items = sorted(dic.items(), key=lambda d: d[1], reverse=True)[:self.max_n_words]

        dic.clear()
        for idx, (word, num) in enumerate(items):
            dic[word] = (idx, num)

        with open(out_path + '.json', 'w', encoding='utf-8') as f:
            json.dump(dic, f, indent=4, ensure_ascii=False)

        print('trans successful')

    def _new_cws_vocab(self, file_path, out_path, max_num=100, threshold=0, pre_vocab_path=None):
        vocab_dic = {}
        pre_vocab = None

        if isinstance(pre_vocab_path, str):
            with open(pre_vocab_path, 'r', encoding='utf-8') as f:
                pre_vocab = json.load(f)
                print(f'load last vocab: {pre_vocab_path}')
            for key in pre_vocab.keys():
                pre_vocab[key] = pre_vocab[key][1]

        if isinstance(pre_vocab_path, list):
            pre_vocab = {}
            for path in pre_vocab_path:
                with open(path, 'r', encoding='utf-8') as f:
                    pre_vocab.update(json.load(f))
            for key in pre_vocab.keys():
                pre_vocab[key] = pre_vocab[key][1]

        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f.readlines():
                segs = line.strip().split()

                for word_pos in segs:
                    word, poss, scores = word_pos.split('_')
                    # word, poss = word_pos.split('_')

                    poss = poss.split(',')
                    scores = [float(score) >= threshold for score in scores.split(',')]
                    if len(set(poss)) != 1 or sum(scores) != len(scores):
                    # if len(set(poss)) != 1:
                        continue

                    item = word + '_' + poss[0]
                    if item not in vocab_dic:
                        vocab_dic[item] = 1
                    else:
                        vocab_dic[item] += 1
        print('vocab size: ', len(vocab_dic))

        rep_len = 0
        if pre_vocab is not None:
            rep_list = []
            for word, num in vocab_dic.items():
                if word in pre_vocab:
                    rep_list += [word]

            for word in rep_list:
                vocab_dic.pop(word)

            rep_len = len(rep_list)
            print(f'have {rep_len} repeat words')

        items = sorted(vocab_dic.items(), key=lambda d: d[1], reverse=True)[:max_num]
        # if pre_vocab is not None:
        #     items += pre_vocab.items()

        vocab_dic.clear()
        for i, (word, num) in enumerate(items):
            vocab_dic[word] = (i, num)

        with open(out_path, 'w', encoding='utf-8') as f:
            json.dump(vocab_dic, f, indent=4, ensure_ascii=False)

    def apply_vocab(self, file_path, oov_vocab, train_vocab, out_path, threshold=0.8, remain=None, max_oov=1, max_conf=1):
        r = None
        if remain is not None:
            r = open(remain, 'w', encoding='utf-8')

        with open(file_path, 'r', encoding='utf-8') as f, open(out_path, 'w', encoding='utf-8') as o:
            if isinstance(oov_vocab, str):
                with open(oov_vocab, 'r', encoding='utf-8') as v:
                    ov = json.load(v)
            else:
                ov = {}
                for path in oov_vocab:
                    with open(path, 'r', encoding='utf-8') as t:
                        ov.update(json.load(t))

            if isinstance(train_vocab, list):
                tv = {}
                for path in train_vocab:
                    with open(path, 'r', encoding='utf-8') as t:
                        tv.update(json.load(t))
            else:
                with open(train_vocab, 'r', encoding='utf-8') as t:
                    tv = json.load(t)

            for line in f.readlines():
                segs = line.strip().split()
                res = []
                legal = True
                oov_num = 0
                conf_num = 0

                for word_pos in segs:
                    add_num = 0
                    word, poss, scores = word_pos.split('_')

                    poss = poss.split(',')
                    scores = [float(score) >= threshold for score in scores.split(',')]

                    if len(set(poss)) != 1:
                        legal = False
                    if sum(scores) < len(scores):
                        conf_num += 1

                    item = word + '_' + poss[0]
                    if item in tv:
                        oov_num += add_num
                        res.append(item)

                    elif item in ov:
                        add_num = 1
                        res.append(item)
                    else:
                        legal = False
                        add_num = 1
                        res.append(item)
                    oov_num += add_num

                if legal is True and oov_num <= max_oov and conf_num <= max_conf:
                    o.write(' '.join(res) + '\n')
                elif r is not None:
                    r.write(' '.join(res) + '\n')

        if r is not None:
            r.close()

    def compare_vocab(self, path1, path2):
        with open(path1, 'r', encoding='utf-8') as f1, open(path2, 'r', encoding='utf-8') as f2:
            vocab1 = json.load(f1)
            vocab2 = json.load(f2)

            keys1 = set(vocab1.keys())
            keys2 = set(vocab2.keys())

            v1_only = keys1 - keys2
            v2_only = keys2 - keys1

            print('{} words only in vocab1: {}'.format(len(v1_only), v1_only))
            print('{} words only in vocab2: {}'.format(len(v2_only), v2_only))

    def filte_corpus(self, file_path, out_path, threshold=0.8):

        with open(file_path, 'r', encoding='utf-8') as f, open(out_path, 'w', encoding='utf-8') as o:
            for line in f.readlines():
                segs = line.strip().split()
                res = []
                legal = False

                for word_pos in segs:
                    word, poss, scores = word_pos.split('_')

                    poss = poss.split(',')
                    scores = [float(score) >= threshold for score in scores.split(',')]

                    if len(set(poss)) != 1 or sum(scores) < len(scores):
                        res.append(word + '_O')
                        continue

                    item = word + '_' + poss[0]

                    legal = True
                    res.append(item)

                if legal is True:
                    o.write(' '.join(res) + '\n')

    def clear_repeat(self, path):
        with open(path, 'r', encoding='utf-8') as f, open(path + '.clr', 'w', encoding='utf-8') as o:
            first = True
            for line in f:
                line = line.strip()
                if line == '」_PU':
                    if first is True:
                        first = False
                    else:
                        continue
                o.write(line + '\n')

    def get_topk(self, path, out_path, k=10000):
        with open(path, 'r', encoding='utf-8') as f, open(out_path, 'w', encoding='utf-8') as o:
            for i, line in enumerate(f.readlines()):
                if i == k:
                    break
                segs = line.strip().split()
                res = []

                for word_pos in segs:
                    word, poss, scores = word_pos.split('_')

                    poss = poss.split(',')

                    item = word + '_' + poss[0]
                    res.append(item)

                o.write(' '.join(res) + '\n')

    def split_with_oov(self, vocab_path, data_path, pre_out_path, max_oov=10):
        with open(vocab_path, 'r', encoding='utf-8') as v, open(data_path, 'r', encoding='utf-8') as f:
            vocab = json.load(v)

            oov_static = {}
            outs = {}
            for i in range(max_oov):
                oov_static[i] = 0
                outs[i] = open(pre_out_path + f'test.oov{i}', 'w', encoding='utf-8')
            oov_static[f'>={max_oov}'] = 0
            outs[f'>={max_oov}'] = open(pre_out_path + f'test.oov>={max_oov}', 'w', encoding='utf-8')

            for line in f.readlines():
                segs = line.strip().split()
                oov_num = 0

                for word_pos in segs:
                    if word_pos not in vocab:
                        oov_num += 1

                if oov_num >= max_oov:
                    oov_num = f'>={max_oov}'
                if oov_num in oov_static:
                    oov_static[oov_num] += 1
                else:
                    oov_static[oov_num] = 1
                outs[oov_num].write(line.strip() + '\n')

            print(oov_static)

            for k, v in outs.items():
                v.close()

    def get_oov_rate(self, vocab_path, file_path):
        with open(vocab_path, 'r', encoding='utf-8') as v, open(file_path, 'r', encoding='utf-8') as f:
            vocab = json.load(v)
            total_word = 0
            oov_word = 0

            for line in f.readlines():
                segs = line.strip().split()

                for word_pos in segs:
                    is_oov = False
                    word, poss = word_pos.split('_')

                    poss = poss.split(',')

                    if len(set(poss)) != 1:
                        print(f'some thing wrong with {word_pos}')
                        is_oov = True

                    item = word + '_' + poss[0]
                    if item not in vocab:
                        is_oov = True

                    if is_oov is True:
                        oov_word += 1
                    total_word += 1

        print(f'total words: {total_word}, oov words: {oov_word}, oov ration: {oov_word / total_word}')


class Vocabulary(object):
    PAD = 0
    UNK = 1
    BOS = 2
    EOS = 3

    def __init__(self, type, dict_path, max_n_words=-1):

        self.dict_path = dict_path
        self._max_n_words = max_n_words

        self._load_vocab(self.dict_path)
        self._id2token = dict([(ii[0], ww) for ww, ii in self._token2id_feq.items()])
        self.tokenizer = Tokenizer(type=type)  # type: _Tokenizer

    @property
    def max_n_words(self):

        if self._max_n_words == -1:
            return len(self._token2id_feq)
        else:
            return self._max_n_words

    def _init_dict(self):

        return {
            "<pad>": (self.PAD, 0),
            "<unk>": (self.UNK, 0),
            "<bos>": (self.BOS, 0),
            "<eos>": (self.EOS, 0)
        }

    def _load_vocab(self, path):
        """
        Load vocabulary from file

        If file is formatted as json, for each item the key is the token, while the value is a tuple such as
        (word_id, word_feq), or a integer which is the index of the token. The index should start from 0.

        If file is formatted as a text file, each line is a token
        """
        self._token2id_feq = self._init_dict()
        N = len(self._token2id_feq)

        if path.endswith(".json"):

            with open(path, encoding='utf-8') as f:
                _dict = json.load(f)
                # Word to word index and word frequence.
                for ww, vv in _dict.items():
                    if isinstance(vv, int):
                        self._token2id_feq[ww] = (vv + N, 0)
                    else:
                        self._token2id_feq[ww] = (vv[0] + N, vv[1])
        else:
            with open(path) as f:
                for i, line in enumerate(f):
                    ww = line.strip().split()[0]
                    self._token2id_feq[ww] = (i + N, 0)

    def token2id(self, word):

        if word in self._token2id_feq and self._token2id_feq[word][0] < self.max_n_words:

            return self._token2id_feq[word][0]
        else:
            return self.UNK

    def id2token(self, word_id):

        return self._id2token[word_id]

    def bos(self):
        """Helper to get index of beginning-of-sentence symbol"""
        return self.BOS

    def pad(self):
        """Helper to get index of pad symbol"""
        return self.PAD

    def eos(self):
        """Helper to get index of end-of-sentence symbol"""
        return self.EOS

    def unk(self):
        """Helper to get index of unk symbol"""
        return self.UNK


class LabelVocabulary(object):
    PAD = 0
    UNK = 1
    O = 2

    def __init__(self, type, dict_path, max_n_words=-1):

        self.dict_path = dict_path
        self._max_n_words = max_n_words

        self._load_vocab(self.dict_path)
        self._id2token = dict([(ii[0], ww) for ww, ii in self._token2id_feq.items()])
        self.tokenizer = Tokenizer(type=type)  # type: _Tokenizer

    @property
    def max_n_words(self):

        if self._max_n_words == -1:
            return len(self._token2id_feq)
        else:
            return self._max_n_words

    def _init_dict(self):

        return {
            "<PAD>": (self.PAD, 0),
            "<UNK>": (self.UNK, 0),
            "O": (self.O, 0),
        }

    def _load_vocab(self, path):
        """
        Load vocabulary from file

        If file is formatted as json, for each item the key is the token, while the value is a tuple such as
        (word_id, word_feq), or a integer which is the index of the token. The index should start from 0.

        If file is formatted as a text file, each line is a token
        """
        self._token2id_feq = self._init_dict()
        N = len(self._token2id_feq)

        if path.endswith(".json"):

            with open(path, encoding='utf-8') as f:
                _dict = json.load(f)
                # Word to word index and word frequence.
                for ww, vv in _dict.items():
                    if isinstance(vv, int):
                        self._token2id_feq[ww] = (vv + N, 0)
                    else:
                        self._token2id_feq[ww] = (vv[0] + N, vv[1])
        else:
            with open(path) as f:
                for i, line in enumerate(f):
                    ww = line.strip().split()[0]
                    self._token2id_feq[ww] = (i + N, 0)

    def token2id(self, word):

        if word in self._token2id_feq and self._token2id_feq[word][0] < self.max_n_words:

            return self._token2id_feq[word][0]
        else:
            return self.UNK

    def id2token(self, word_id):

        return self._id2token[word_id]

    def get_O_id(self):
        return self.O

    def pad(self):
        """Helper to get index of pad symbol"""
        return self.PAD

    def unk(self):
        """Helper to get index of unk symbol"""
        return self.UNK


PAD = Vocabulary.PAD
EOS = Vocabulary.EOS
BOS = Vocabulary.BOS
UNK = Vocabulary.UNK
