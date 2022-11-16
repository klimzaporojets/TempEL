import editdistance
from typing import List


def from_bert_to_text(bert_tokens: List):
    output_text = ''
    for curr_bert_token in bert_tokens:
        if curr_bert_token.startswith('##'):
            output_text += curr_bert_token[2:]
        else:
            output_text += ' '
            output_text += curr_bert_token
    return output_text

def get_ratio_edit_distance(string1, string2):
    edit_distance = editdistance.eval(string1, string2)
    ratio_edit_distance = edit_distance / \
                          ((len(string1) + len(string2)) / 2)
    return edit_distance, ratio_edit_distance


def get_ratio_edit_distance_v2(string1, string2):
    edit_distance = editdistance.eval(string1, string2)
    len_longest_string = max(len(string1), len(string2))
    ratio_edit_distance = (edit_distance / len_longest_string)
    return edit_distance, ratio_edit_distance
