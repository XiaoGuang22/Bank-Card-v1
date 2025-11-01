from collections import defaultdict

import torch
import numpy as np
from scipy.special import logsumexp

NINF = -1 * float('inf')
DEFAULT_EMISSION_THRESHOLD = 0.01


def _reconstruct(labels, blank=0, eos_label=None):
    """
    重构标签序列
    Args:
        labels: 原始标签序列
        blank: CTC blank标签
        eos_label: 结束标记的label（遇到就停止）
    """
    new_labels = []
    previous = None
    for l in labels:
        # ⭐ 遇到结束标记就停止
        if eos_label is not None and l == eos_label:
            break
        # 合并相同标签
        if l != previous:
            new_labels.append(l)
            previous = l
    # 删除blank
    new_labels = [l for l in new_labels if l != blank]
    return new_labels


def greedy_decode(emission_log_prob, blank=0, **kwargs):
    eos_label = kwargs.get('eos_label', None)  # ⭐ 获取结束标记
    labels = np.argmax(emission_log_prob, axis=-1)
    labels = _reconstruct(labels, blank=blank, eos_label=eos_label)
    return labels


def beam_search_decode(emission_log_prob, blank=0, **kwargs):
    beam_size = kwargs['beam_size']
    emission_threshold = kwargs.get('emission_threshold', np.log(DEFAULT_EMISSION_THRESHOLD))
    eos_label = kwargs.get('eos_label', None)  # ⭐ 获取结束标记

    length, class_count = emission_log_prob.shape

    beams = [([], 0)]
    for t in range(length):
        new_beams = []
        for prefix, accumulated_log_prob in beams:
            for c in range(class_count):
                log_prob = emission_log_prob[t, c]
                if log_prob < emission_threshold:
                    continue
                new_prefix = prefix + [c]
                new_accu_log_prob = accumulated_log_prob + log_prob
                new_beams.append((new_prefix, new_accu_log_prob))

        new_beams.sort(key=lambda x: x[1], reverse=True)
        beams = new_beams[:beam_size]

    total_accu_log_prob = {}
    for prefix, accu_log_prob in beams:
        labels = tuple(_reconstruct(prefix, eos_label=eos_label))  # ⭐ 传入结束标记
        total_accu_log_prob[labels] = \
            logsumexp([accu_log_prob, total_accu_log_prob.get(labels, NINF)])

    labels_beams = [(list(labels), accu_log_prob)
                    for labels, accu_log_prob in total_accu_log_prob.items()]
    labels_beams.sort(key=lambda x: x[1], reverse=True)
    labels = labels_beams[0][0]

    return labels


def prefix_beam_decode(emission_log_prob, blank=0, **kwargs):
    beam_size = kwargs['beam_size']
    emission_threshold = kwargs.get('emission_threshold', np.log(DEFAULT_EMISSION_THRESHOLD))
    eos_label = kwargs.get('eos_label', None)  # ⭐ 获取结束标记

    length, class_count = emission_log_prob.shape

    beams = [(tuple(), (0, NINF))]

    for t in range(length):
        new_beams_dict = defaultdict(lambda: (NINF, NINF))

        for prefix, (lp_b, lp_nb) in beams:
            for c in range(class_count):
                log_prob = emission_log_prob[t, c]
                if log_prob < emission_threshold:
                    continue

                end_t = prefix[-1] if prefix else None

                new_lp_b, new_lp_nb = new_beams_dict[prefix]

                if c == blank:
                    new_beams_dict[prefix] = (
                        logsumexp([new_lp_b, lp_b + log_prob, lp_nb + log_prob]),
                        new_lp_nb
                    )
                    continue
                if c == end_t:
                    new_beams_dict[prefix] = (
                        new_lp_b,
                        logsumexp([new_lp_nb, lp_nb + log_prob])
                    )

                new_prefix = prefix + (c,)
                new_lp_b, new_lp_nb = new_beams_dict[new_prefix]

                if c != end_t:
                    new_beams_dict[new_prefix] = (
                        new_lp_b,
                        logsumexp([new_lp_nb, lp_b + log_prob, lp_nb + log_prob])
                    )
                else:
                    new_beams_dict[new_prefix] = (
                        new_lp_b,
                        logsumexp([new_lp_nb, lp_b + log_prob])
                    )

        beams = sorted(new_beams_dict.items(), key=lambda x: logsumexp(x[1]), reverse=True)
        beams = beams[:beam_size]

    labels = list(beams[0][0])
    # ⭐ 对最终结果也应用结束标记截断
    labels = _reconstruct(labels, blank=blank, eos_label=eos_label)
    return labels


def ctc_decode(log_probs, label2char=None, blank=0, method='beam_search', beam_size=10, eos_label=None):
    """
    CTC解码
    Args:
        eos_label: 结束标记的label（遇到就停止解码）
    """
    emission_log_probs = np.transpose(log_probs.cpu().detach().numpy(), (1, 0, 2))

    decoders = {
        'greedy': greedy_decode,
        'beam_search': beam_search_decode,
        'prefix_beam_search': prefix_beam_decode,
    }
    decoder = decoders[method]

    decoded_list = []
    for emission_log_prob in emission_log_probs:
        decoded = decoder(emission_log_prob, blank=blank, beam_size=beam_size, eos_label=eos_label)  # ⭐ 传入结束标记
        if label2char:
            decoded = [label2char[l] for l in decoded]
        decoded_list.append(decoded)
    return decoded_list
