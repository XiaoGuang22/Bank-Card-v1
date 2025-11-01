from collections import defaultdict

import torch
import numpy as np
from scipy.special import logsumexp

NINF = -1 * float('inf')
DEFAULT_EMISSION_THRESHOLD = 0.01


def _reconstruct(labels, blank=0, bos_label=None, eos_label=None):
    """
    重构标签序列
    
    处理逻辑：
    1. 删除开头的BOS标记（如果存在）
    2. 遇到EOS标记就停止（删除EOS及其后面的所有内容）
    3. 合并连续重复的标签
    4. 删除blank标签
    
    Args:
        labels: 原始标签序列
        blank: CTC blank标签
        bos_label: 开始标记的label（会被删除）
        eos_label: 结束标记的label（遇到就停止）
    
    Returns:
        new_labels: 重构后的标签序列
    
    示例：
        >>> # BOS=12('<'), EOS=13('>')
        >>> _reconstruct([12, 4, 5, 5, 3, 2, 13, 1, 2], blank=0, bos_label=12, eos_label=13)
        [4, 5, 3, 2]  # 删除了<，遇到>就停止，合并了重复的5
    """
    new_labels = []
    previous = None
    
    # ⭐ 删除开头的BOS
    start_idx = 0
    if bos_label is not None and len(labels) > 0 and labels[0] == bos_label:
        start_idx = 1
    
    # ⭐ 处理标签：遇到EOS就停止
    for l in labels[start_idx:]:
        # 遇到EOS就停止（不包含EOS）
        if eos_label is not None and l == eos_label:
            break
        
        # 合并连续重复的标签
        if l != previous:
            new_labels.append(l)
            previous = l
    
    # 删除blank
    new_labels = [l for l in new_labels if l != blank]
    
    # ⭐ 再次过滤BOS和EOS（防止中间出现）
    if bos_label is not None:
        new_labels = [l for l in new_labels if l != bos_label]
    if eos_label is not None:
        new_labels = [l for l in new_labels if l != eos_label]
    
    return new_labels


def greedy_decode(emission_log_prob, blank=0, **kwargs):
    """
    贪心解码
    
    Args:
        emission_log_prob: (T, C) 发射概率（log空间）
        blank: CTC blank标签
        **kwargs: 其他参数
            - bos_label: 开始标记的label
            - eos_label: 结束标记的label
    
    Returns:
        labels: 解码后的标签序列
    """
    bos_label = kwargs.get('bos_label', None)
    eos_label = kwargs.get('eos_label', None)
    
    labels = np.argmax(emission_log_prob, axis=-1)
    labels = _reconstruct(labels, blank=blank, bos_label=bos_label, eos_label=eos_label)
    return labels


def beam_search_decode(emission_log_prob, blank=0, **kwargs):
    """
    Beam Search解码
    
    Args:
        emission_log_prob: (T, C) 发射概率（log空间）
        blank: CTC blank标签
        **kwargs: 其他参数
            - beam_size: beam大小
            - emission_threshold: 发射阈值
            - bos_label: 开始标记的label
            - eos_label: 结束标记的label
    
    Returns:
        labels: 解码后的标签序列
    """
    beam_size = kwargs['beam_size']
    emission_threshold = kwargs.get('emission_threshold', np.log(DEFAULT_EMISSION_THRESHOLD))
    bos_label = kwargs.get('bos_label', None)
    eos_label = kwargs.get('eos_label', None)

    length, class_count = emission_log_prob.shape

    # 初始化beam：[(前缀, 累积log概率)]
    beams = [([], 0)]
    
    for t in range(length):
        new_beams = []
        for prefix, accumulated_log_prob in beams:
            for c in range(class_count):
                log_prob = emission_log_prob[t, c]
                
                # 剪枝：跳过概率太低的类别
                if log_prob < emission_threshold:
                    continue
                
                new_prefix = prefix + [c]
                new_accu_log_prob = accumulated_log_prob + log_prob
                new_beams.append((new_prefix, new_accu_log_prob))

        # 排序并保留top-k
        new_beams.sort(key=lambda x: x[1], reverse=True)
        beams = new_beams[:beam_size]

    # 合并相同的解码结果
    total_accu_log_prob = {}
    for prefix, accu_log_prob in beams:
        labels = tuple(_reconstruct(prefix, blank=blank, bos_label=bos_label, eos_label=eos_label))
        total_accu_log_prob[labels] = \
            logsumexp([accu_log_prob, total_accu_log_prob.get(labels, NINF)])

    # 选择概率最高的结果
    labels_beams = [(list(labels), accu_log_prob)
                    for labels, accu_log_prob in total_accu_log_prob.items()]
    labels_beams.sort(key=lambda x: x[1], reverse=True)
    labels = labels_beams[0][0]

    return labels


def prefix_beam_decode(emission_log_prob, blank=0, **kwargs):
    """
    Prefix Beam Search解码（更高效的beam search）
    
    Args:
        emission_log_prob: (T, C) 发射概率（log空间）
        blank: CTC blank标签
        **kwargs: 其他参数
            - beam_size: beam大小
            - emission_threshold: 发射阈值
            - bos_label: 开始标记的label
            - eos_label: 结束标记的label
    
    Returns:
        labels: 解码后的标签序列
    """
    beam_size = kwargs['beam_size']
    emission_threshold = kwargs.get('emission_threshold', np.log(DEFAULT_EMISSION_THRESHOLD))
    bos_label = kwargs.get('bos_label', None)
    eos_label = kwargs.get('eos_label', None)

    length, class_count = emission_log_prob.shape

    # 初始化beam：[(前缀, (lp_b, lp_nb))]
    # lp_b: 以blank结尾的log概率
    # lp_nb: 以非blank结尾的log概率
    beams = [(tuple(), (0, NINF))]

    for t in range(length):
        new_beams_dict = defaultdict(lambda: (NINF, NINF))

        for prefix, (lp_b, lp_nb) in beams:
            for c in range(class_count):
                log_prob = emission_log_prob[t, c]
                
                # 剪枝：跳过概率太低的类别
                if log_prob < emission_threshold:
                    continue

                end_t = prefix[-1] if prefix else None

                new_lp_b, new_lp_nb = new_beams_dict[prefix]

                # 情况1：当前字符是blank
                if c == blank:
                    new_beams_dict[prefix] = (
                        logsumexp([new_lp_b, lp_b + log_prob, lp_nb + log_prob]),
                        new_lp_nb
                    )
                    continue
                
                # 情况2：当前字符与前缀末尾相同
                if c == end_t:
                    new_beams_dict[prefix] = (
                        new_lp_b,
                        logsumexp([new_lp_nb, lp_nb + log_prob])
                    )

                # 情况3：扩展前缀
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

        # 排序并保留top-k
        beams = sorted(new_beams_dict.items(), key=lambda x: logsumexp(x[1]), reverse=True)
        beams = beams[:beam_size]

    # 选择概率最高的结果
    labels = list(beams[0][0])
    labels = _reconstruct(labels, blank=blank, bos_label=bos_label, eos_label=eos_label)
    return labels


def ctc_decode(log_probs, label2char=None, blank=0, method='beam_search', beam_size=10, 
               bos_label=None, eos_label=None):
    """
    CTC解码（支持BOS和EOS标记）
    
    Args:
        log_probs: (T, B, C) 或 (B, T, C) 的log概率张量
        label2char: 标签到字符的映射字典（可选）
        blank: CTC blank标签（默认0）
        method: 解码方法
            - 'greedy': 贪心解码（最快）
            - 'beam_search': Beam Search解码（推荐）
            - 'prefix_beam_search': Prefix Beam Search解码（最准确）
        beam_size: beam大小（仅用于beam search）
        bos_label: 开始标记的label（会被删除）
        eos_label: 结束标记的label（遇到就停止）
    
    Returns:
        decoded_list: 解码后的标签序列列表（batch）
    
    示例：
        >>> log_probs = torch.randn(50, 8, 14)  # (T=50, B=8, C=14)
        >>> decoded = ctc_decode(
        ...     log_probs, 
        ...     method='beam_search', 
        ...     beam_size=10,
        ...     bos_label=12,  # '<'
        ...     eos_label=13   # '>'
        ... )
        >>> print(decoded[0])  # 第一个样本的解码结果
        [4, 5, 3, 2, 11, 1, 2, 3, 4]
    """
    # 转换为numpy并调整维度：(T, B, C) -> (B, T, C)
    emission_log_probs = np.transpose(log_probs.cpu().detach().numpy(), (1, 0, 2))

    # 解码器映射
    decoders = {
        'greedy': greedy_decode,
        'beam_search': beam_search_decode,
        'prefix_beam_search': prefix_beam_decode,
    }
    
    if method not in decoders:
        raise ValueError(f"❌ 不支持的解码方法: {method}。可选: {list(decoders.keys())}")
    
    decoder = decoders[method]

    # 对batch中的每个样本进行解码
    decoded_list = []
    for emission_log_prob in emission_log_probs:
        decoded = decoder(
            emission_log_prob, 
            blank=blank, 
            beam_size=beam_size, 
            bos_label=bos_label, 
            eos_label=eos_label
        )
        
        # 如果提供了label2char映射，转换为字符
        if label2char:
            decoded = [label2char[l] for l in decoded]
        
        decoded_list.append(decoded)
    
    return decoded_list


# ========== 测试代码 ==========
if __name__ == '__main__':
    """
    测试CTC解码器（BOS='<', EOS='>'）
    """
    print("="*60)
    print("测试CTC解码器（BOS='<', EOS='>'）")
    print("="*60)
    
    # 模拟log概率
    # T=20, B=2, C=14 (0-9数字 + / + < + > + blank)
    torch.manual_seed(42)
    log_probs = torch.randn(20, 2, 14)
    
    # 定义字符集
    CHARS = '0123456789/<>'
    CHAR2LABEL = {char: i + 1 for i, char in enumerate(CHARS)}
    LABEL2CHAR = {label: char for char, label in CHAR2LABEL.items()}
    BOS_LABEL = CHAR2LABEL['<']  # 12
    EOS_LABEL = CHAR2LABEL['>']  # 13
    
    print(f"\n字符集: '{CHARS}'")
    print(f"BOS标记: '<' (label={BOS_LABEL})")
    print(f"EOS标记: '>' (label={EOS_LABEL})")
    print(f"Blank标记: label=0")
    
    # 测试不同的解码方法
    methods = ['greedy', 'beam_search', 'prefix_beam_search']
    
    for method in methods:
        print(f"\n{'='*60}")
        print(f"解码方法: {method}")
        print(f"{'='*60}")
        
        decoded = ctc_decode(
            log_probs,
            label2char=None,  # 返回label序列
            blank=0,
            method=method,
            beam_size=10,
            bos_label=BOS_LABEL,
            eos_label=EOS_LABEL
        )
        
        for i, labels in enumerate(decoded):
            text = ''.join([LABEL2CHAR.get(l, '?') for l in labels])
            print(f"样本 {i}: {labels}")
            print(f"       '{text}'")
    
    # 测试带label2char的解码
    print(f"\n{'='*60}")
    print("测试带label2char的解码")
    print(f"{'='*60}")
    
    decoded = ctc_decode(
        log_probs,
        label2char=LABEL2CHAR,  # 直接返回字符
        blank=0,
        method='beam_search',
        beam_size=10,
        bos_label=BOS_LABEL,
        eos_label=EOS_LABEL
    )
    
    for i, chars in enumerate(decoded):
        text = ''.join(chars)
        print(f"样本 {i}: '{text}'")
    
    # 测试_reconstruct函数
    print(f"\n{'='*60}")
    print("测试_reconstruct函数")
    print(f"{'='*60}")
    
    test_cases = [
        # (输入, BOS, EOS, 预期输出, 描述)
        ([12, 4, 5, 3, 2, 0, 1, 2, 3, 4, 13], 12, 13, [4, 5, 3, 2, 1, 2, 3, 4], "正常情况：<...>"),
        ([12, 1, 1, 2, 2, 3, 13], 12, 13, [1, 2, 3], "合并重复"),
        ([12, 0, 1, 0, 2, 0, 13], 12, 13, [1, 2], "删除blank"),
        ([12, 4, 5, 13, 6, 7], 12, 13, [4, 5], "遇到>就停止"),
        ([4, 5, 3, 2, 13], None, 13, [4, 5, 3, 2], "无BOS"),
        ([12, 4, 5, 3, 2], 12, None, [4, 5, 3, 2], "无EOS"),
        ([12, 4, 12, 5, 13], 12, 13, [4, 5], "中间的<被删除"),
        ([12, 13], 12, 13, [], "只有<>"),
    ]
    
    for i, (labels, bos, eos, expected, desc) in enumerate(test_cases):
        result = _reconstruct(labels, blank=0, bos_label=bos, eos_label=eos)
        status = "✅" if result == expected else "❌"
        print(f"\n测试 {i+1}: {status} - {desc}")
        print(f"  输入: {labels}")
        print(f"  BOS={bos}, EOS={eos}")
        print(f"  预期: {expected}")
        print(f"  结果: {result}")
    
    print(f"\n{'='*60}")
    print("测试完成！")
    print(f"{'='*60}")
