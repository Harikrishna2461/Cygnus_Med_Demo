import os, torch, cv2
import numpy as np
from scipy.optimize import minimize_scalar
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib as mpl

# -------------------------- 关键配置：解决plt中文/负号显示 + 无GUI运行问题 --------------------------
plt.switch_backend('Agg')  # 无头模式，服务器/无桌面环境必加
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'SimHei']  # 支持英文+中文
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
plt.rcParams['font.weight'] = 'bold'  # 默认加粗，和原LaTeX的textbf一致
plt.rcParams['figure.constrained_layout.use'] = True  # 自动适配布局，防止文字截断


def rank_guassian_filter(img, kernel_size=3):
    """
    Apply a rank-based Gaussian-weighted filter for robust activation map denoising.
    """
    filtered_img = np.zeros_like(img)
    pad_width = kernel_size // 2
    padded_img = np.pad(img, pad_width, mode='reflect')
    ax = np.array(range(kernel_size ** 2)) - kernel_size ** 2 // 2

    for i in range(pad_width, img.shape[0] + pad_width):
        for j in range(pad_width, img.shape[1] + pad_width):
            window = padded_img[i - pad_width:i + pad_width + 1,
                     j - pad_width:j + pad_width + 1]

            sorted_window = np.sort(window.flatten())
            mean = sorted_window.mean()
            if mean > 0:
                sigma = sorted_window.std() / mean  # std -> cov
                kernel = np.exp(-(ax ** 2) / (2 * sigma ** 2))
                kernel = kernel / np.sum(kernel)
                value = (sorted_window * kernel).sum()
            else:
                value = 0
            filtered_img[i - pad_width, j - pad_width] = value

    return filtered_img


def least_squares(map1, map2):
    """
    Find the scalar that minimizes the squared difference between map1 and scalar * map2.
    """

    def diff(x, map1, map2):
        return np.sum((map1 - map2 * x) ** 2)

    result = minimize_scalar(diff, args=(map1, map2))
    return result.x


def vis_text_plt(words, relevances, vis_width=500, font_size=8):
    """
    【替代原LaTeX的核心函数】固定每10个单词换行，彻底解决重叠问题
    ✅ 核心规则：每10个单词强制换行 ✅ 兼容特殊标记换行 ✅ 配色1:1复刻原代码
    Args:
        words: 所有待可视化的token列表
        relevances: 每个token对应的相关性分数
        vis_width: 图像宽度，和上方热力图对齐
        font_size: 绘制字号
    Returns:
        np.ndarray: BGR格式文本图像，可直接和cv2图像拼接
    """
    # 特殊字符替换 + 空值兼容
    all_words = []
    for word in words:
        if not word or len(word) == 0:
            word = ' '
        word = word.replace('▁', ' ').replace('Ġ', ' ') \
            .replace('\\', '\\backslash').replace('\n', ' ') \
            .replace('_', '_').replace('^', '^')
        all_words.append(word)
    words = all_words

    # ========== 核心配置 ==========
    char_width = font_size   # 字符间距系数，可微调
    line_height = 1  # 行间距，换行时下移距离
    words_per_line = 10  # 固定每行显示10个单词
    word_count = 0  # 单词计数器，累计到10换行

    # 初始化坐标
    x_pos = 2  # 每行起始横坐标
    y_pos = 15  # 第一行起始纵坐标

    # 画布高度自适应：
    max_lines = len(words) // words_per_line
    fig_height = max_lines * (line_height / 10)
    fig, ax = plt.subplots(figsize=(vis_width / 100, fig_height), dpi=100)
    ax.set_xlim(0, vis_width)
    ax.set_ylim(0, max_lines * line_height)  # 足够高度容纳所有行
    ax.axis('off')
    fig.patch.set_facecolor('white')
    fig.patch.set_alpha(1.0)

    # jet色卡，和原cv2.COLORMAP_JET一致
    jet_cmap = mpl.colormaps['jet']

    # ========== 遍历绘制：每10个单词强制换行 ==========
    for word, rel in zip(words, relevances):
        if not word:
            x_pos += char_width
            continue

        word_length = len(word) * char_width
        if word_length < 2:
            word_length = char_width

        # ========== 1. 相关性>=0 → jet热力色 ==========
        if rel >= 0:
            color = jet_cmap(rel)[:3]
            if word[:2] == '$ ' and word[-1] == '$':
                ax.text(x_pos, y_pos, f'{word},', fontsize=font_size, color=color,
                        weight='bold', ha='left', va='center')
            else:
                ax.text(x_pos, y_pos, word, fontsize=font_size, color=color,
                        weight='bold', ha='left', va='center')
            x_pos += word_length + 1.2
            word_count += 1

        # ========== 2. rel=-1 → 当前token：黑底白字 ==========
        elif rel == -1:
            bbox_props = dict(boxstyle="square,pad=0.05", facecolor='black', edgecolor='none')
            ax.text(x_pos, y_pos, word, fontsize=font_size, color='white', weight='bold',
                    ha='left', va='center', bbox=bbox_props)
            x_pos += word_length + 1.2
            word_count += 1

        # ========== 3. rel=-2 → 后续token：灰色 ==========
        elif rel == -2:
            gray_color = (200 / 255, 200 / 255, 200 / 255)
            ax.text(x_pos, y_pos, word, fontsize=font_size, color=gray_color,
                    weight='bold', ha='left', va='center')
            x_pos += word_length + 1.2
            word_count += 1

        # ========== 4. rel=-3 → 强制换行 + Candidates ==========
        elif rel == -3:
            x_pos = 8
            y_pos -= line_height
            word_count = 0  # 重置计数器
            ax.text(x_pos, y_pos, 'Candidates:', fontsize=font_size + 1, color='black',
                    weight='bold', ha='left', va='center')
            x_pos += len('Candidates:') * char_width + 3

        # ========== 5. rel=-4 → 强制换行 + 自定义文本 ==========
        elif rel == -4:
            x_pos = 8
            y_pos -= line_height
            word_count = 0  # 重置计数器
            ax.text(x_pos, y_pos, word, fontsize=font_size, color='black',
                    weight='bold', ha='left', va='center')
            x_pos += len(word) * char_width + 2

        # ========== 核心逻辑：每10个单词强制换行 ==========
        if word_count >= words_per_line:
            x_pos = 8  # 回到行首
            y_pos -= line_height  # 下移一行
            word_count = 0  # 重置计数器

    # ========== matplotlib全版本兼容：画布转cv2数组 ==========
    fig.canvas.draw()
    try:
        img_data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    except AttributeError:
        img_data = np.asarray(fig.canvas.buffer_rgba())[:, :, :3]
        img_data = img_data.flatten()

    # 格式转换：RGB → BGR，和cv2图像无缝拼接
    img_data = img_data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    img_data = cv2.cvtColor(img_data, cv2.COLOR_RGB2BGR)

    # 释放内存
    plt.close(fig)
    plt.clf()
    plt.cla()

    return img_data


def multimodal_process(raw_img, vision_shape, img_scores, txt_scores, txts, candidates, candi_scores, \
                       vis_token_idx, img_save_fn, eval_only=False, vis_width=-1):
    """
    其余功能完全不变，仅保留文本绘图调用
    """
    txt_scores = txt_scores[:-1]
    all_scores = np.concatenate([img_scores, txt_scores], 0)
    all_scores = (all_scores - all_scores.min()) / (all_scores.max() - all_scores.min())
    img_scores = all_scores[:len(img_scores)]
    txt_scores = all_scores[len(img_scores):]

    eval_only = True if img_save_fn == "" else False

    # for multiple imgs
    if isinstance(vision_shape[0], tuple):
        resized_img, img_map = [], []
        start_idx = 0
        for n in range(len(vision_shape)):
            t_h, t_w = vision_shape[n]
            h, w, c = raw_img[n].shape

            if vis_width > 0:
                h = int(vis_width)
                w = int(float(w) / h * vis_width)

            end_idx = start_idx + int(t_h * t_w)
            img_map_ = rank_guassian_filter(img_scores[start_idx: end_idx].reshape(t_h, t_w), 3)
            start_idx = end_idx
            img_map_ = (img_map_ * 255).astype('uint8')

            if not eval_only:
                img_map_ = cv2.applyColorMap(img_map_, cv2.COLORMAP_JET)
                img_map_ = cv2.resize(img_map_, (w, h))
                if vis_width > 0:
                    raw_img_ = cv2.resize(raw_img[n], (w, h))
                    resized_img.append(raw_img_)

            img_map.append(img_map_)

        if eval_only:
            return None, img_map

        out_img = [img_map[i] * 0.5 + resized_img[i] * 0.5 for i in range(len(vision_shape))]
        out_img = np.concatenate(out_img, 1)
        vis_w = out_img.shape[1] if vis_width < 0 else 500

        try:
            txt_map = vis_text_plt(txts, txt_scores, vis_width=vis_w, font_size=5)
        except Exception as e:
            print(f'Skip text visualization, error: {str(e)[:100]}')
            return out_img, img_map

        if not isinstance(txt_map, np.ndarray) or txt_map.size == 0:
            print('Skip txt visualization, plt generate empty image')
            return out_img, img_map

        txt_map = cv2.resize(txt_map, (out_img.shape[1], txt_map.shape[0]))
        out_img = np.concatenate([out_img, txt_map], 0)

        return out_img, img_map

    # single img
    elif len(vision_shape) == 2:
        t_h, t_w = vision_shape
        h, w, c = raw_img.shape
        if vis_width > 0:
            h = int(float(h) / w * vis_width)
            w = int(vis_width)

        img_scores = rank_guassian_filter(img_scores.reshape(t_h, t_w), 3)
        img_scores = (img_scores * 255).astype('uint8')

        if eval_only:
            return None, img_scores

        img_map = cv2.applyColorMap(img_scores, cv2.COLORMAP_JET)
        img_map = cv2.resize(img_map, (w, h))
        if vis_width > 0:
            raw_img = cv2.resize(raw_img, (w, h))
        out_img = img_map * 0.5 + raw_img * 0.5
        vis_w = out_img.shape[1] if vis_width < 0 else 500

        try:
            txt_map = vis_text_plt(txts, txt_scores, vis_width=vis_w, font_size=7)
        except Exception as e:
            print(f'Skip text visualization, error: {str(e)[:100]}')
            return out_img, img_scores

        if not isinstance(txt_map, np.ndarray) or txt_map.size == 0:
            print('Skip txt visualization, plt generate empty image')
            return out_img, img_scores

        txt_map = cv2.resize(txt_map, (w, txt_map.shape[0]))
        out_img = np.concatenate([out_img, txt_map], 0)

        return out_img, img_scores

    # video
    else:
        b, t_h, t_w = vision_shape
        h, w, c = raw_img[0].shape
        if vis_width > 0:
            h = int(float(h) / w * vis_width)
            w = int(vis_width)

        img_scores = np.array([rank_guassian_filter(_.reshape(t_h, t_w), 3) for _ in np.array_split(img_scores, b)])
        img_scores = (img_scores * 255).astype('uint8')

        if eval_only:
            return None, img_scores

        img_map = [cv2.resize(cv2.applyColorMap(_, cv2.COLORMAP_JET), (w, h)) for _ in img_scores]
        if vis_width > 0:
            raw_img = [cv2.resize(_, (w, h)) for _ in raw_img]
        out_img = [img_map[i] * 0.5 + raw_img[i] * 0.5 for i in range(b)]
        out_img = np.concatenate(out_img, 1)
        vis_w = out_img.shape[1] if vis_width < 0 else 500

        try:
            txt_map = vis_text_plt(txts, txt_scores, vis_width=vis_w, font_size=5)
        except Exception as e:
            print(f'Skip text visualization, error: {str(e)[:100]}')
            return out_img, img_scores

        if not isinstance(txt_map, np.ndarray) or txt_map.size == 0:
            print('Skip txt visualization, plt generate empty image')
            return out_img, img_scores

        txt_map = cv2.resize(txt_map, (int(w * b), txt_map.shape[0]))
        out_img = np.concatenate([out_img, txt_map], 0)

        return out_img, img_scores


def id2idx(inp_id, target_id, return_last=False):
    """完全保留原函数"""
    if isinstance(target_id, list):
        n = len(target_id)
        indexes = [i for i in range(len(inp_id) - n + 1) if inp_id[i:i + n] == target_id]
        if len(indexes) > 0:
            idx = indexes[-1]
            if return_last:
                idx += len(target_id) - 1
        else:
            idx = -1
    else:
        try:
            idx = inp_id.index(target_id)
        except:
            idx = -1
    return idx


def TAM(tokens, vision_shape, logit_list, special_ids, vision_input, \
        processor, save_fn, target_token, img_scores_list, eval_only=False):
    """
    完全保留所有核心逻辑，无任何修改
    """
    img_id = special_ids['img_id']
    prompt_id = special_ids['prompt_id']
    answer_id = special_ids['answer_id']

    if len(img_id) == 1:
        img_idx = (np.array(tokens) == img_id[0]).nonzero()[0]
    else:
        img_idx = [id2idx(tokens, img_id[0], True), id2idx(tokens, img_id[1])]

    prompt_idx = [id2idx(tokens, prompt_id[0], True), id2idx(tokens, prompt_id[1])]
    answer_idx = [id2idx(tokens, answer_id[0], True), id2idx(tokens, answer_id[1])]

    prompt = processor.tokenizer.tokenize(processor.batch_decode([tokens[prompt_idx[0] + 1: prompt_idx[1]]], \
                                                                 skip_special_tokens=False,
                                                                 clean_up_tokenization_spaces=False)[0])
    answer = processor.tokenizer.tokenize(processor.batch_decode([tokens[answer_idx[0] + 1:]], \
                                                                 skip_special_tokens=False,
                                                                 clean_up_tokenization_spaces=False)[0])
    txt_all = prompt + answer

    round_idx = -1
    this_token_idx = 0

    if isinstance(target_token, int):
        round_idx = target_token
        this_token_idx = -1
        vis_token_idx = len(prompt) + target_token
    else:
        round_idx, prompt_token_idx = target_token
        this_token_idx = prompt_idx[0] + prompt_token_idx + 1
        vis_token_idx = prompt_token_idx

    if round_idx == 0 and isinstance(target_token, int):
        for t in range(len(prompt) + 1):
            img_map = TAM(tokens, vision_shape, logit_list, special_ids, vision_input, processor, \
                          save_fn if t == len(prompt) else '', [0, t], img_scores_list, eval_only)
            if t == 0:
                first_ori = img_map
        return first_ori

    if round_idx == 0:
        if prompt_token_idx == len(prompt):
            this_token_idx = logit_list[0].shape[1] - 1
            cls_id = tokens[this_token_idx]
        elif prompt_token_idx == 0:
            cls_id = logit_list[0][0, prompt_idx[0] + 1].argmax(0)
        else:
            cls_id = tokens[this_token_idx]
    else:
        cls_id = tokens[answer_idx[0] + round_idx + 1]

    scores = torch.cat([logit_list[_][0, :, cls_id] for _ in range(round_idx + 1)], -1).clip(min=0)
    scores = scores.detach().cpu().float().numpy()
    prompt_scores = scores[prompt_idx[0] + 1: prompt_idx[1]]
    last_prompt = scores[logit_list[0].shape[1] - 1: logit_list[0].shape[1]]
    answer_scores = scores[answer_idx[0] + 1:]
    txt_scores = np.concatenate([prompt_scores, last_prompt, answer_scores], -1)

    if isinstance(img_idx, list):
        img_scores = scores[img_idx[0] + 1: img_idx[1]]
    else:
        img_scores = scores[img_idx]

    img_scores_list.append(img_scores)

    if len(img_scores_list) > 1 and vis_token_idx < len(txt_all):
        non_repeat_idx = []
        for i in range(vis_token_idx):
            if i < len(txt_all) and txt_all[i] != txt_all[vis_token_idx]:
                non_repeat_idx.append(i)
        txt_scores_ = txt_scores[non_repeat_idx]
        img_scores_list_ = [img_scores_list[_] for _ in non_repeat_idx]

        w = txt_scores_
        w = w / (w.sum() + 1e-8)
        interf_img_scores = (np.stack(img_scores_list_, 0) * w.reshape(-1, 1)).sum(0)
        scaled_map = least_squares(img_scores, interf_img_scores)
        img_scores = (img_scores - interf_img_scores * scaled_map).clip(min=0)

    if isinstance(vision_shape[0], tuple):
        cv_img = [cv2.cvtColor(np.array(_), cv2.COLOR_RGB2BGR) for _ in vision_input]
    elif len(vision_shape) == 2:
        cv_img = np.array(vision_input)
        if len(cv_img.shape) == 4 and cv_img.shape[0] == 1:
            cv_img = cv_img[0]
        cv_img = cv2.cvtColor(cv_img, cv2.COLOR_RGB2BGR)
    else:
        cv_img = [cv2.cvtColor(np.array(_), cv2.COLOR_RGB2BGR) for _ in vision_input[0]]

    candi_scores, candi_ids = logit_list[round_idx][0, this_token_idx].topk(3)
    candi_scores = candi_scores.softmax(0)
    candidates = processor.batch_decode([[_] for _ in candi_ids])

    vis_img, img_map = multimodal_process(cv_img, vision_shape, img_scores, txt_scores, txt_all, candidates,
                                          candi_scores, vis_token_idx, \
                                          save_fn, eval_only=eval_only, vis_width=-1 if eval_only else 500)

    if save_fn != '' and vis_token_idx < (len(txt_all) - 1) and isinstance(vis_img, np.ndarray):
        os.makedirs(os.path.dirname(save_fn), exist_ok=True)
        cv2.imwrite(save_fn, vis_img)

    return img_map