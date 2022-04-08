import asyncio
import os
from collections import Counter, OrderedDict
from multiprocessing import Pool

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from kaldi_io import read_mat, read_vec_flt
from sklearn.preprocessing import KBinsDiscretizer, LabelEncoder, StandardScaler
from torch.utils.data import Dataset


class MissingClassMapError(Exception):
    pass


def load_n_col(file, numpy=False):
    data = []
    with open(file) as fp:
        for line in fp:
            data.append(line.strip().split(" "))
    columns = list(zip(*data))
    if numpy:
        columns = [np.array(list(i)) for i in columns]
    else:
        columns = [list(i) for i in columns]
    return columns


def odict_from_2_col(file, numpy=False):
    col0, col1 = load_n_col(file, numpy=numpy)
    return OrderedDict({c0: c1 for c0, c1 in zip(col0, col1)})


def load_one_tomany(file, numpy=False):
    one = []
    many = []
    with open(file) as fp:
        for line in fp:
            line = line.strip().split(" ", 1)
            one.append(line[0])
            m = line[1].split(" ")
            many.append(np.array(m) if numpy else m)
    if numpy:
        one = np.array(one)
    return one, many


def train_transform(feats, seqlen):
    leeway = feats.shape[0] - seqlen
    startslice = np.random.randint(0, int(leeway)) if leeway > 0 else 0
    feats = (
        feats[startslice : startslice + seqlen]
        if leeway > 0
        else np.pad(feats, [(0, -leeway), (0, 0)], "constant")
    )
    return torch.FloatTensor(feats)


async def get_item_train(instructions):
    fpath = instructions[0]
    seqlen = instructions[1]
    raw_feats = read_mat(fpath)
    feats = train_transform(raw_feats, seqlen)
    return feats


async def get_item_test(filepath):
    raw_feats = read_mat(filepath)
    return torch.FloatTensor(raw_feats)


def async_map(coroutine_func, iterable):
    loop = asyncio.get_event_loop()
    future = asyncio.gather(*(coroutine_func(param) for param in iterable))
    return loop.run_until_complete(future)


class SpeakerDataset(Dataset):
    def __init__(
        self,
        data_base_path,
        real_speaker_labels=True,
        asynchr=True,
        num_workers=3,
        test_mode=False,
        class_enc_dict={},
        **kwargs
    ):
        self.data_base_path = data_base_path
        self.num_workers = num_workers
        self.test_mode = test_mode
        self.real_speaker_labels = real_speaker_labels
        # self.label_types = label_types
        if self.test_mode:
            self.label_types = []
        else:
            self.label_types = ["speaker"] if self.real_speaker_labels else []

        if os.path.isfile(os.path.join(data_base_path, "spk2nat")):
            self.label_types.append("nationality")

        if os.path.isfile(os.path.join(data_base_path, "spk2gender")):
            self.label_types.append("gender")

        if os.path.isfile(os.path.join(data_base_path, "utt2age")):
            self.label_types.append("age_regression")
            self.label_types.append("age")

        if os.path.isfile(os.path.join(data_base_path, "utt2rec")):
            self.label_types.append("rec")

        if os.path.isfile(os.path.join(data_base_path, "utt2genre")):
            self.label_types.append("genre")

        if self.test_mode and self.label_types:
            assert class_enc_dict, "Class mapping must be passed to test mode dataset"
        self.class_enc_dict = class_enc_dict

        utt2spk_path = os.path.join(data_base_path, "utt2spk")
        spk2utt_path = os.path.join(data_base_path, "spk2utt")
        feats_scp_path = os.path.join(data_base_path, "feats.scp")

        assert os.path.isfile(utt2spk_path)
        assert os.path.isfile(feats_scp_path)
        assert os.path.isfile(spk2utt_path)

        verilist_path = os.path.join(data_base_path, "veri_pairs")

        if self.test_mode:
            if os.path.isfile(verilist_path):
                self.veri_labs, self.veri_0, self.veri_1 = load_n_col(
                    verilist_path, numpy=True
                )
                self.veri_labs = self.veri_labs.astype(int)
                self.veripairs = True
            else:
                self.veripairs = False

        self.utts, self.uspkrs = load_n_col(utt2spk_path)
        self.utt_fpath_dict = odict_from_2_col(feats_scp_path)

        self.label_enc = LabelEncoder()

        self.original_spkrs, self.spkutts = load_one_tomany(spk2utt_path)
        self.spkrs = self.label_enc.fit_transform(self.original_spkrs)
        self.spk_utt_dict = OrderedDict(
            {k: v for k, v in zip(self.spkrs, self.spkutts)}
        )

        self.spk_original_spk_dict = {
            k: v for k, v in zip(self.spkrs, self.original_spkrs)
        }

        self.uspkrs = self.label_enc.transform(self.uspkrs)
        self.utt_spkr_dict = OrderedDict({k: v for k, v in zip(self.utts, self.uspkrs)})

        self.utt_list = list(self.utt_fpath_dict.keys())
        self.first_batch = True

        self.num_classes = (
            {"speaker": len(self.label_enc.classes_)}
            if self.real_speaker_labels
            else {}
        )
        self.asynchr = asynchr

        if "nationality" in self.label_types:
            self.natspkrs, self.nats = load_n_col(
                os.path.join(data_base_path, "spk2nat")
            )
            self.nats = [n.lower().strip() for n in self.nats]
            self.natspkrs = self.label_enc.transform(self.natspkrs)
            self.nat_label_enc = LabelEncoder()

            if not self.test_mode:
                self.nats = self.nat_label_enc.fit_transform(self.nats)
            else:
                self.nat_label_enc = self.class_enc_dict["nationality"]
                self.nats = self.nat_label_enc.transform(self.nats)

            self.spk_nat_dict = OrderedDict(
                {k: v for k, v in zip(self.natspkrs, self.nats)}
            )
            self.num_classes["nationality"] = len(self.nat_label_enc.classes_)

        if "gender" in self.label_types:
            self.genspkrs, self.genders = load_n_col(
                os.path.join(data_base_path, "spk2gender")
            )
            self.genspkrs = self.label_enc.transform(self.genspkrs)
            self.gen_label_enc = LabelEncoder()

            if not self.test_mode:
                self.genders = self.gen_label_enc.fit_transform(self.genders)
            else:
                self.gen_label_enc = self.class_enc_dict["gender"]
                self.genders = self.gen_label_enc.transform(self.genders)

            self.spk_gen_dict = OrderedDict(
                {k: v for k, v in zip(self.genspkrs, self.genders)}
            )
            self.num_classes["gender"] = len(self.gen_label_enc.classes_)

        if "age" in self.label_types:
            # self.genspkrs, self.genders = load_n_col(os.path.join(data_base_path, 'spk2gender'))
            self.num_age_bins = (
                kwargs["num_age_bins"] if "num_age_bins" in kwargs else 10
            )
            self.ageutts, self.ages = load_n_col(
                os.path.join(data_base_path, "utt2age")
            )
            self.ages = np.array(self.ages).astype(np.float)
            self.age_label_enc = KBinsDiscretizer(
                n_bins=self.num_age_bins, encode="ordinal", strategy="uniform"
            )

            if not self.test_mode or "age" not in self.class_enc_dict:
                self.age_classes = self.age_label_enc.fit_transform(
                    np.array(self.ages).reshape(-1, 1)
                ).flatten()
            else:
                self.age_label_enc = self.class_enc_dict["age"]
                self.age_classes = self.age_label_enc.transform(
                    np.array(self.ages).reshape(-1, 1)
                ).flatten()

            self.utt_age_class_dict = OrderedDict(
                {k: v for k, v in zip(self.ageutts, self.age_classes)}
            )
            self.num_classes["age"] = self.num_age_bins

        if "age_regression" in self.label_types:
            # self.genspkrs, self.genders = load_n_col(os.path.join(data_base_path, 'spk2gender'))
            self.ageutts, self.ages = load_n_col(
                os.path.join(data_base_path, "utt2age")
            )
            self.ages = np.array(self.ages).astype(np.float)
            self.age_reg_enc = StandardScaler()

            if not self.test_mode or "age_regression" not in self.class_enc_dict:
                self.ages = self.age_reg_enc.fit_transform(
                    np.array(self.ages).reshape(-1, 1)
                ).flatten()
            else:
                self.age_reg_enc = self.class_enc_dict["age_regression"]
                self.ages = self.age_reg_enc.transform(
                    np.array(self.ages).reshape(-1, 1)
                ).flatten()

            self.utt_age_dict = OrderedDict(
                {k: v for k, v in zip(self.ageutts, self.ages)}
            )
            self.num_classes["age_regression"] = 1

        if "rec" in self.label_types:
            self.recutts, self.recs = load_n_col(
                os.path.join(data_base_path, "utt2rec")
            )
            self.recs = np.array(self.recs)
            self.rec_label_enc = LabelEncoder()

            if not self.test_mode:
                self.recs = self.rec_label_enc.fit_transform(self.recs)
            else:
                self.rec_label_enc = self.class_enc_dict["rec"]
                self.recs = self.rec_label_enc.transform(self.recs)

            self.utt_rec_dict = OrderedDict(
                {k: v for k, v in zip(self.recutts, self.recs)}
            )
            self.num_classes["rec"] = len(self.rec_label_enc.classes_)

        if "genre" in self.label_types:
            self.genreutts, self.genres = load_n_col(
                os.path.join(data_base_path, "utt2genre")
            )
            self.genres = np.array(self.genres)
            self.genre_label_enc = LabelEncoder()

            if not self.test_mode:
                self.genres = self.genre_label_enc.fit_transform(self.genres)
                self.utt_genre_dict = OrderedDict(
                    {k: v for k, v in zip(self.genreutts, self.genres)}
                )
                self.num_classes["genre"] = len(self.genre_label_enc.classes_)
            else:
                # TODO: add this check to other attributes
                if "genre" in self.class_enc_dict:
                    self.genre_label_enc = self.class_enc_dict["genre"]
                    self.genres = self.genre_label_enc.transform(self.genres)
                    self.utt_genre_dict = OrderedDict(
                        {k: v for k, v in zip(self.genreutts, self.genres)}
                    )
                    self.num_classes["genre"] = len(self.genre_label_enc.classes_)
                else:
                    self.label_types.remove("genre")

        self.class_enc_dict = self.get_class_encs()

    def __len__(self):
        return len(self.utt_list)

    def get_class_encs(self):
        class_enc_dict = {}
        if "speaker" in self.label_types:
            class_enc_dict["speaker"] = self.label_enc
        if "age" in self.label_types:
            class_enc_dict["age"] = self.age_label_enc
        if "age_regression" in self.label_types:
            class_enc_dict["age_regression"] = self.age_reg_enc
        if "nationality" in self.label_types:
            class_enc_dict["nationality"] = self.nat_label_enc
        if "gender" in self.label_types:
            class_enc_dict["gender"] = self.gen_label_enc
        if "rec" in self.label_types:
            class_enc_dict["rec"] = self.rec_label_enc
        if "genre" in self.label_types:
            class_enc_dict["genre"] = self.genre_label_enc
        self.class_enc_dict = class_enc_dict
        return class_enc_dict

    @staticmethod
    def get_item(instructions):
        fpath = instructions[0]
        seqlen = instructions[1]
        feats = read_mat(fpath)
        feats = train_transform(feats, seqlen)
        return feats

    def get_item_test(self, idx):
        utt = self.utt_list[idx]
        fpath = self.utt_fpath_dict[utt]
        feats = read_mat(fpath)
        feats = torch.FloatTensor(feats)

        label_dict = {}
        speaker = self.utt_spkr_dict[utt]

        if "speaker" in self.label_types:
            label_dict["speaker"] = torch.LongTensor([speaker])
        if "gender" in self.label_types:
            label_dict["gender"] = torch.LongTensor([self.spk_gen_dict[speaker]])
        if "nationality" in self.label_types:
            label_dict["nationality"] = torch.LongTensor([self.spk_nat_dict[speaker]])
        if "age" in self.label_types:
            label_dict["age"] = torch.LongTensor([self.utt_age_class_dict[utt]])
        if "age_regression" in self.label_types:
            label_dict["age_regression"] = torch.FloatTensor([self.utt_age_dict[utt]])
        if "genre" in self.label_types:
            label_dict["genre"] = torch.LongTensor([self.utt_genre_dict[utt]])

        return feats, label_dict

    def get_test_items(self, num_items=-1, exclude_speakers=None, use_async=True):
        utts = self.utt_list
        if num_items >= 1:
            replace = len(utts) <= num_items
            utts = np.random.choice(utts, size=num_items, replace=replace)

        utts = np.array(utts)
        spkrs = np.array([self.utt_spkr_dict[utt] for utt in utts])
        original_spkrs = np.array([self.spk_original_spk_dict[spkr] for spkr in spkrs])

        if exclude_speakers:
            mask = np.array(
                [False if s in exclude_speakers else True for s in original_spkrs]
            )
            utts = utts[mask]
            spkrs = spkrs[mask]
            original_spkrs = original_spkrs[mask]

        fpaths = [self.utt_fpath_dict[utt] for utt in utts]
        if use_async:
            feats = async_map(get_item_test, fpaths)
        else:
            feats = [torch.FloatTensor(read_mat(f)) for f in fpaths]

        label_dict = {}
        label_dict["speaker"] = np.array(spkrs)
        label_dict["original_speaker"] = np.array(original_spkrs)

        if "nationality" in self.label_types:
            label_dict["nationality"] = np.array([self.spk_nat_dict[s] for s in spkrs])
        if "gender" in self.label_types:
            label_dict["gender"] = np.array([self.spk_gen_dict[s] for s in spkrs])
        if "age" in self.label_types:
            label_dict["age"] = np.array([self.utt_age_class_dict[utt] for utt in utts])
        if "age_regression" in self.label_types:
            label_dict["age_regression"] = np.array(
                [self.utt_age_dict[utt] for utt in utts]
            )
        if "genre" in self.label_types:
            label_dict["genre"] = np.array([self.utt_genre_dict[utt] for utt in utts])

        return feats, label_dict, utts

    def get_batches(self, batch_size=256, max_seq_len=400, sp_tensor=True):
        """
        Main data iterator, specify batch_size and max_seq_len
        sp_tensor determines whether speaker labels are returned as Tensor object or not
        """
        # with Parallel(n_jobs=self.num_workers) as parallel:
        self.idpool = self.spkrs.copy()
        assert batch_size < len(
            self.idpool
        )  # Metric learning assumption large num classes
        lens = [max_seq_len for _ in range(batch_size)]
        while True:
            if len(self.idpool) <= batch_size:
                batch_ids = np.array(self.idpool)
                self.idpool = self.spkrs.copy()
                rem_ids = np.random.choice(
                    self.idpool, size=batch_size - len(batch_ids), replace=False
                )
                batch_ids = np.concatenate([batch_ids, rem_ids])
                self.idpool = list(set(self.idpool) - set(rem_ids))
            else:
                batch_ids = np.random.choice(
                    self.idpool, size=batch_size, replace=False
                )
                self.idpool = list(set(self.idpool) - set(batch_ids))

            batch_fpaths = []
            batch_utts = []
            for i in batch_ids:
                utt = np.random.choice(self.spk_utt_dict[i])
                batch_utts.append(utt)
                batch_fpaths.append(self.utt_fpath_dict[utt])

            if self.asynchr:
                batch_feats = async_map(get_item_train, zip(batch_fpaths, lens))
            else:
                batch_feats = [self.get_item(a) for a in zip(batch_fpaths, lens)]
            # batch_feats = parallel(delayed(self.get_item)(a) for a in zip(batch_fpaths, lens))

            label_dict = {}
            if "speaker" in self.label_types:
                label_dict["speaker"] = (
                    torch.LongTensor(batch_ids) if sp_tensor else batch_ids
                )
            if "nationality" in self.label_types:
                label_dict["nationality"] = torch.LongTensor(
                    [self.spk_nat_dict[s] for s in batch_ids]
                )
            if "gender" in self.label_types:
                label_dict["gender"] = torch.LongTensor(
                    [self.spk_gen_dict[s] for s in batch_ids]
                )
            if "age" in self.label_types:
                label_dict["age"] = torch.LongTensor(
                    [self.utt_age_class_dict[u] for u in batch_utts]
                )
            if "age_regression" in self.label_types:
                label_dict["age_regression"] = torch.FloatTensor(
                    [self.utt_age_dict[u] for u in batch_utts]
                )
            if "rec" in self.label_types:
                label_dict["rec"] = torch.LongTensor(
                    [self.utt_rec_dict[u] for u in batch_utts]
                )
            if "genre" in self.label_types:
                label_dict["genre"] = torch.LongTensor(
                    [self.utt_genre_dict[u] for u in batch_utts]
                )

            yield torch.stack(batch_feats), label_dict

    def get_batches_naive(self, batch_size=256, max_seq_len=400, sp_tensor=True):
        """
        Main data iterator, specify batch_size and max_seq_len
        sp_tensor determines whether speaker labels are returned as Tensor object or not
        """
        self.idpool = self.spkrs.copy()
        # assert batch_size < len(self.idpool) #Metric learning assumption large num classes
        lens = [max_seq_len for _ in range(batch_size)]
        while True:
            batch_ids = np.random.choice(self.idpool, size=batch_size)
            batch_fpaths = []
            batch_utts = []
            for i in batch_ids:
                utt = np.random.choice(self.spk_utt_dict[i])
                batch_utts.append(utt)
                batch_fpaths.append(self.utt_fpath_dict[utt])

            if self.asynchr:
                batch_feats = async_map(get_item_train, zip(batch_fpaths, lens))
            else:
                batch_feats = [self.get_item(a) for a in zip(batch_fpaths, lens)]
            # batch_feats = parallel(delayed(self.get_item)(a) for a in zip(batch_fpaths, lens))

            label_dict = {}
            if "speaker" in self.label_types:
                label_dict["speaker"] = (
                    torch.LongTensor(batch_ids) if sp_tensor else batch_ids
                )
            if "nationality" in self.label_types:
                label_dict["nationality"] = torch.LongTensor(
                    [self.spk_nat_dict[s] for s in batch_ids]
                )
            if "gender" in self.label_types:
                label_dict["gender"] = torch.LongTensor(
                    [self.spk_gen_dict[s] for s in batch_ids]
                )
            if "age" in self.label_types:
                label_dict["age"] = torch.LongTensor(
                    [self.utt_age_class_dict[u] for u in batch_utts]
                )
            if "age_regression" in self.label_types:
                label_dict["age_regression"] = torch.FloatTensor(
                    [self.utt_age_dict[u] for u in batch_utts]
                )
            if "rec" in self.label_types:
                label_dict["rec"] = torch.LongTensor(
                    [self.utt_rec_dict[u] for u in batch_utts]
                )
            if "genre" in self.label_types:
                label_dict["genre"] = torch.LongTensor(
                    [self.utt_genre_dict[u] for u in batch_utts]
                )

            yield torch.stack(batch_feats), label_dict

    def get_batches_balance(
        self, balance_attribute="speaker", batch_size=256, max_seq_len=400
    ):
        """
        Main data iterator, specify batch_size and max_seq_len
        Specify which attribute to balance
        """
        assert balance_attribute in self.label_types

        if balance_attribute == "speaker":
            self.anchorpool = self.spkrs.copy()
            self.get_utt_method = lambda x: np.random.choice(self.spk_utt_dict[x])

        if balance_attribute == "nationality":
            self.anchorpool = sorted(list(set(self.nats)))
            self.nat_utt_dict = OrderedDict({k: [] for k in self.anchorpool})
            for u in self.utt_list:
                spk = self.utt_spkr_dict[u]
                nat = self.spk_nat_dict[spk]
                self.nat_utt_dict[nat].append(u)
            for n in self.nat_utt_dict:
                self.nat_utt_dict[u] = np.array(self.nat_utt_dict[u])
            self.get_utt_method = lambda x: np.random.choice(self.nat_utt_dict[x])

        if balance_attribute == "age":
            self.anchorpool = sorted(list(set(self.age_classes)))
            self.age_utt_dict = OrderedDict({k: [] for k in self.anchorpool})
            for u in self.utt_age_class_dict:
                nat_class = self.utt_age_class_dict[u]
                self.age_utt_dict[nat_class].append(u)
            for a in self.age_utt_dict:
                self.age_utt_dict[a] = np.array(self.age_utt_dict[a])
            self.get_utt_method = lambda x: np.random.choice(self.age_utt_dict[x])

        lens = [max_seq_len for _ in range(batch_size)]
        while True:
            anchors = np.random.choice(self.anchorpool, size=batch_size)
            batch_utts = [self.get_utt_method(a) for a in anchors]

            batch_fpaths = []
            batch_ids = []
            for utt in batch_utts:
                batch_fpaths.append(self.utt_fpath_dict[utt])
                batch_ids.append(self.utt_spkr_dict[utt])

            if self.asynchr:
                batch_feats = async_map(get_item_train, zip(batch_fpaths, lens))
            else:
                batch_feats = [self.get_item(a) for a in zip(batch_fpaths, lens)]

            label_dict = {}
            if "speaker" in self.label_types:
                label_dict["speaker"] = torch.LongTensor(batch_ids)
            if "nationality" in self.label_types:
                label_dict["nationality"] = torch.LongTensor(
                    [self.spk_nat_dict[s] for s in batch_ids]
                )
            if "gender" in self.label_types:
                label_dict["gender"] = torch.LongTensor(
                    [self.spk_gen_dict[s] for s in batch_ids]
                )
            if "age" in self.label_types:
                label_dict["age"] = torch.LongTensor(
                    [self.utt_age_class_dict[u] for u in batch_utts]
                )
            if "age_regression" in self.label_types:
                label_dict["age_regression"] = torch.FloatTensor(
                    [self.utt_age_dict[u] for u in batch_utts]
                )
            if "rec" in self.label_types:
                label_dict["rec"] = torch.LongTensor(
                    [self.utt_rec_dict[u] for u in batch_utts]
                )
            if "genre" in self.label_types:
                label_dict["genre"] = torch.LongTensor(
                    [self.utt_genre_dict[u] for u in batch_utts]
                )

            yield torch.stack(batch_feats), label_dict

    def get_alldata_batches(self, batch_size=256, max_seq_len=400):
        utt_list = self.utt_list
        start_index = 0
        lens = [max_seq_len for _ in range(batch_size)]
        while start_index <= len(utt_list):
            batch_utts = utt_list[start_index : start_index + batch_size]
            batch_fpaths = []
            batch_ids = []
            for utt in batch_utts:
                batch_fpaths.append(self.utt_fpath_dict[utt])
                batch_ids.append(self.utt_spkr_dict[utt])

            if self.asynchr:
                batch_feats = async_map(get_item_train, zip(batch_fpaths, lens))
            else:
                batch_feats = [self.get_item(a) for a in zip(batch_fpaths, lens)]

            yield torch.stack(batch_feats), np.array(batch_ids)
