# Copyright (c) 2024 Alibaba Inc (authors: Xiang Lyu, Zhihao Du)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import re
import regex
chinese_char_pattern = re.compile(r'[\u4e00-\u9fff]+')


# whether contain chinese character
def contains_chinese(text):
    return bool(chinese_char_pattern.search(text))


# replace special symbol
def replace_corner_mark(text):
    text = text.replace('²', '平方')
    text = text.replace('³', '立方')
    return text


# remove meaningless symbol
def remove_bracket(text):
    text = text.replace('（', '').replace('）', '')
    text = text.replace('【', '').replace('】', '')
    text = text.replace('`', '').replace('`', '')
    text = text.replace("——", " ")
    return text


# spell Arabic numerals
def spell_out_number(text: str, inflect_parser):
    new_text = []
    st = None
    for i, c in enumerate(text):
        if not c.isdigit():
            if st is not None:
                num_str = inflect_parser.number_to_words(text[st: i])
                new_text.append(num_str)
                st = None
            new_text.append(c)
        else:
            if st is None:
                st = i
    if st is not None and st < len(text):
        num_str = inflect_parser.number_to_words(text[st:])
        new_text.append(num_str)
    return ''.join(new_text)


# split paragrah logic：
# 1. per sentence max len token_max_n, min len token_min_n, merge if last sentence len less than merge_len
# 2. cal sentence len according to lang
# 3. split sentence according to puncatation
def split_paragraph(text: str, tokenize, lang="zh", token_max_n=80, token_min_n=60, merge_len=20, comma_split=False):
    def calc_utt_length(_text: str):
        if lang == "zh":
            return len(_text)
        else:
            return len(tokenize(_text))

    def should_merge(_text: str):
        if lang == "zh":
            return len(_text) < merge_len
        else:
            return len(tokenize(_text)) < merge_len

    def force_split_by_length(text: str, max_length: int):
        """强制按长度分割文本"""
        chunks = []
        
        # 优先在逗号处分割
        if '，' in text:
            parts = text.split('，')
            current = ""
            for i, part in enumerate(parts):
                test_text = current + ('，' if current else '') + part
                if calc_utt_length(test_text) > max_length and current:
                    chunks.append(current)
                    current = part
                else:
                    current = test_text
            if current:
                if calc_utt_length(current) > max_length:
                    # 如果还是太长，按更小的块进行字符分割
                    # 对于中文，使用max_length的一半作为字符分割长度，更保守
                    char_split_len = max(10, max_length // 2)  # 至少10个字符，最多max_length/2
                    for j in range(0, len(current), char_split_len):
                        chunk = current[j:j+char_split_len]
                        if chunk.strip():
                            chunks.append(chunk.strip())
                else:
                    chunks.append(current)
        else:
            # 没有逗号，按更保守的字符长度分割
            char_split_len = max(10, max_length // 2)  # 至少10个字符，最多max_length/2
            for i in range(0, len(text), char_split_len):
                chunk = text[i:i+char_split_len]
                if chunk.strip():
                    chunks.append(chunk.strip())
        
        return chunks

    if lang == "zh":
        pounc = ['。', '？', '！', '；', '：', '、', '.', '?', '!', ';']
    else:
        pounc = ['.', '?', '!', ';', ':']
    if comma_split:
        pounc.extend(['，', ','])

    if text[-1] not in pounc:
        if lang == "zh":
            text += "。"
        else:
            text += "."

    st = 0
    utts = []
    for i, c in enumerate(text):
        if c in pounc:
            if len(text[st: i]) > 0:
                utts.append(text[st: i] + c)
            if i + 1 < len(text) and text[i + 1] in ['"', '”']:
                tmp = utts.pop(-1)
                utts.append(tmp + text[i + 1])
                st = i + 2
            else:
                st = i + 1

    final_utts = []
    cur_utt = ""
    for utt in utts:
        if calc_utt_length(cur_utt + utt) > token_max_n and calc_utt_length(cur_utt) > token_min_n:
            final_utts.append(cur_utt)
            cur_utt = ""
        cur_utt = cur_utt + utt
    if len(cur_utt) > 0:
        if should_merge(cur_utt) and len(final_utts) != 0:
            # 检查合并后是否超长
            merged_text = final_utts[-1] + cur_utt
            if calc_utt_length(merged_text) > token_max_n:
                # 合并后超长，不合并，分别处理
                # 先处理cur_utt
                if calc_utt_length(cur_utt) > token_max_n:
                    force_split_chunks = force_split_by_length(cur_utt, token_max_n)
                    final_utts.extend(force_split_chunks)
                else:
                    final_utts.append(cur_utt)
            else:
                # 合并后不超长，正常合并
                final_utts[-1] = merged_text
        else:
            # 检查是否超长，如果超长则强制分割
            if calc_utt_length(cur_utt) > token_max_n:
                force_split_chunks = force_split_by_length(cur_utt, token_max_n)
                final_utts.extend(force_split_chunks)
            else:
                final_utts.append(cur_utt)

    return final_utts


# remove blank between chinese character
def replace_blank(text: str):
    out_str = []
    for i, c in enumerate(text):
        if c == " ":
            if ((text[i + 1].isascii() and text[i + 1] != " ") and
                    (text[i - 1].isascii() and text[i - 1] != " ")):
                out_str.append(c)
        else:
            out_str.append(c)
    return "".join(out_str)


def is_only_punctuation(text):
    # Regular expression: Match strings that consist only of punctuation marks or are empty.
    punctuation_pattern = r'^[\p{P}\p{S}]*$'
    return bool(regex.fullmatch(punctuation_pattern, text))
