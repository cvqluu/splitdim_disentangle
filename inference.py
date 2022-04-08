import argparse
import configparser
import os
import json

from collections import OrderedDict
import numpy as np

import torch
from data_io import SpeakerDataset
from models.extractors import ETDNN, FTDNN, XTDNN
from sklearn.preprocessing import normalize
from tqdm import tqdm
from utils import SpeakerRecognitionMetrics


def mtd(stuff, device):
    if isinstance(stuff, torch.Tensor):
        return stuff.to(device)
    else:
        return [mtd(s, device) for s in stuff]


def probe_attributes(
    model_dict,
    ds_train,
    ds_test,
    probe_model,
    device,
    num_train_examples=10000,
    exclude_train_speakers_from_test=True,
):
    """
    Evaluate by probing attributes in data
    """
    assert ds_test.test_mode

    for m in model_dict:
        model_dict[m]["model"].eval()

    label_types = []
    for l in ds_test.label_types:
        if (
            l in ds_train.label_types
            and l not in ["speaker", "rec"]
            and not l.endswith("regression")
        ):
            label_types.append(l)
    # Include ones that aren't being trained

    print(f"Probing embeddings for: {label_types}")
    if len(label_types) == 0:
        return {}

    embed_arrays = []
    label_dicts = []

    tspeakers = set()

    with torch.no_grad():
        for ds, num_egs in zip([ds_train, ds_test], [num_train_examples, -1]):
            feats, label_dict, _ = ds.get_test_items(
                num_egs, exclude_speakers=tspeakers
            )

            if exclude_train_speakers_from_test:
                # Pass through train speakers to test call
                tspeakers = set(label_dict["original_speaker"])

            embeds = []
            for feat in tqdm(feats):
                feat = feat.unsqueeze(0).to(device)
                embed = model_dict["generator"]["model"](feat)
                embeds.append(embed.cpu().numpy())

            embed_arrays.append(np.vstack(embeds))
            label_dicts.append(label_dict)

    X_train, X_test = embed_arrays[0], embed_arrays[1]
    ldict_train, ldict_test = label_dicts[0], label_dicts[1]
    accuracy_dict = {}

    for l in label_types:
        print(f"Probing embeds for {l} information...")
        y_train, y_test = ldict_train[l], ldict_test[l]

        # Fit model
        probe_model.fit(X_train, y_train)

        # Evaluate performance on test
        preds = probe_model.predict(X_test)
        accuracy_dict[l] = np.equal(y_test, preds).sum() / len(y_test)

    for m in model_dict:
        model_dict[m]["model"].train()

    return accuracy_dict


def test_all_factors(model_dict, ds_test, device, verification_dims=None):
    """
    Tests all factors

    verification_dims can be used to specify which embedding dims to perform verification with
    """

    assert ds_test.test_mode

    for m in model_dict:
        model_dict[m]["model"].eval()

    clf_heads = []
    for m in model_dict:
        if "label_type" in model_dict[m]:
            if model_dict[m]["label_type"] in ds_test.label_types:
                clf_heads.append(m)

    # label_types = [l for l in ds_test.label_types if l in model_label_types]

    with torch.no_grad():
        feats, label_dict, all_utts = ds_test.get_test_items()
        all_embeds = []
        pred_dict = {m: [] for m in clf_heads}
        for feat in tqdm(feats):
            feat = feat.unsqueeze(0).to(device)
            embed = model_dict["generator"]["model"](feat)
            for m in clf_heads:
                if "dim_split" not in model_dict[m]:
                    head_input = embed
                else:
                    head_input = []
                    for splits in model_dict[m]["dim_split"]:
                        head_input.append(embed[:, splits[0] : splits[1]])

                    head_input = torch.cat(head_input, dim=1)

                if model_dict[m]["label_type"].endswith("regression"):
                    if model_dict[m]["is_adversary"]:
                        pred = model_dict[m]["model"](head_input)
                    else:
                        pred = model_dict[m]["model"](head_input, label=None)
                else:
                    if model_dict[m]["is_adversary"]:
                        pred = torch.argmax(model_dict[m]["model"](head_input), dim=1)
                    else:
                        pred = torch.argmax(
                            model_dict[m]["model"](head_input, label=None), dim=1
                        )
                pred_dict[m].append(pred.cpu().numpy()[0])
            all_embeds.append(embed.cpu().numpy())

    accuracy_dict = {
        m: np.equal(label_dict[model_dict[m]["label_type"]], pred_dict[m]).sum()
        / len(all_utts)
        for m in clf_heads
        if not model_dict[m]["label_type"].endswith("regression")
    }

    for m in clf_heads:
        label_type = model_dict[m]["label_type"]
        if label_type.endswith("regression"):
            accuracy_dict[m] = np.mean((label_dict[label_type] - pred_dict[m]) ** 2)
            l1losses = np.abs(label_dict[label_type] - pred_dict[m])
            print(
                "Mean L1Loss: {} ± {}".format(
                    l1losses.mean(), l1losses.std() / len(l1losses)
                )
            )
        if label_type == "age":
            age_labels = np.array(label_dict["age"])
            age_preds = np.array(pred_dict[m])

            # Speaker balanced age acc
            spkrs = np.array([ds_test.utt_spkr_dict[u] for u in all_utts])
            set_spkrs = set(spkrs)
            age_accs = []
            for s in set_spkrs:
                idxs = spkrs == s
                sp_acc = np.equal(age_labels[idxs], age_preds[idxs]).sum() / sum(idxs)
                age_accs.append(sp_acc)
            accuracy_dict[f"{m}_sp_balanced_age"] = np.mean(age_accs)

            # Balanced age acc
            set_ages = set(age_labels)
            bal_age_accs = []
            for a in set_ages:
                idxs = age_labels == a
                bal_acc = np.equal(age_labels[idxs], age_preds[idxs]).sum() / sum(idxs)
                bal_age_accs.append(bal_acc)
            accuracy_dict[f"{m}_balanced_age"] = np.mean(bal_age_accs)

            # Fuzzy accuracy
            fuzzy_score = []
            for l, p in zip(age_labels, age_preds):
                if p >= l - 1 and p <= l + 1:
                    fuzzy_score.append(1)
                else:
                    fuzzy_score.append(0)
            accuracy_dict[f"{m}_fuzzy_age"] = np.array(fuzzy_score).mean()

    if ds_test.veripairs:
        metric = SpeakerRecognitionMetrics(distance_measure="cosine")
        all_embeds = np.vstack(all_embeds)
        all_embeds = normalize(all_embeds, axis=1)
        all_utts = np.array(all_utts)

        if verification_dims:
            start_dim = verification_dims[0]
            end_dim = verification_dims[1]
            all_embeds = all_embeds[:, start_dim:end_dim]

        utt_embed = OrderedDict({k: v for k, v in zip(all_utts, all_embeds)})
        emb0 = np.array([utt_embed[utt] for utt in ds_test.veri_0])
        emb1 = np.array([utt_embed[utt] for utt in ds_test.veri_1])

        scores = metric.scores_from_pairs(emb0, emb1)
        print("Min score: {}, max score {}".format(min(scores), max(scores)))
        eer, mindcf1 = metric.compute_min_cost(scores, 1 - ds_test.veri_labs)
    else:
        eer, mindcf1 = 0.0, 0.0

    for m in model_dict:
        model_dict[m]["model"].train()

    return eer, mindcf1, accuracy_dict
