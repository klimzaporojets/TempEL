import re
from enum import Enum, auto


class States(Enum):
    OUTSIDE = auto()
    IN_FILE = auto()
    IN_SQ = auto()
    ACCEPT = auto()


class StatesKlim(Enum):
    OUTSIDE = auto()
    # IN_FILE = auto()
    IN_SQ = auto()
    IN_PRE = auto()
    # ACCEPT = auto()


def try_accept(text, query):
    try:
        pos = text.index(query)
        return text[pos + len(query):], query, text[:pos]
    except ValueError:
        return False, None, None


def try_accept_re(text, query):
    srch = re.search(query, text)
    if srch is None:
        return False, None, None
    pos = srch.start()

    return text[srch.end():], query, text[:pos]


def accept_re(text, *query):
    if len(query) == 1:
        return try_accept_re(text, query[0])

    results = dict()
    for q in query:
        results[q] = try_accept_re(text, q)

    minlen = float("inf")
    shortest = None
    last = None
    for result in results:
        last = results[result]
        if results[result][2] is not None and len(results[result][2]) < minlen:
            minlen = len(results[result][2])
            shortest = results[result]
    return shortest if shortest is not None else last


def accept(text, *query):
    if len(query) == 1:
        return try_accept(text, query[0])

    results = dict()
    for q in query:
        results[q] = try_accept(text, q)

    minlen = float("inf")
    shortest = None
    last = None
    for result in results:
        last = results[result]
        if results[result][2] is not None and len(results[result][2]) < minlen:
            minlen = len(results[result][2])
            shortest = results[result]
    return shortest if shortest is not None else last


def recursive_clean(text, begin, end, pre=None):
    state = [States.OUTSIDE]
    ret = ""
    if pre is None or len(pre) == 0:
        pre = begin

    while state[-1] is not States.ACCEPT:
        if len(state) == 0:
            ret += text
            state.append(States.ACCEPT)
            break

        elif state[-1] == States.OUTSIDE:
            accept_result, _, before = accept(text, *pre)

            if not accept_result:
                ret += text
                state.append(States.ACCEPT)
            else:
                ret += before
                text = accept_result
                state.append(States.IN_SQ)

        elif state[-1] == States.IN_SQ:
            accept_result, accepted, before = accept(text, *begin, *end)
            if not accept_result:
                # parse error
                return ret + text
            else:
                if accepted in begin:
                    text = accept_result
                    state.append(States.IN_SQ)
                elif accepted in end:
                    text = accept_result
                    state.pop()

    return ret


def detect_next_pre(pre_start, text, look_pre_in_beginning):
    try:
        if not look_pre_in_beginning:
            pos = text.index(pre_start)
            return True, text[:pos], text[pos + len(pre_start):]
        else:
            if len(text) >= len(pre_start):
                if text[:len(pre_start)] == pre_start:
                    return True, '', text[len(pre_start):]
                else:
                    return False, None, None
            else:
                return False, None, None
    except ValueError:
        return False, None, None


def detect_next_inside(begin, end, pre_end, text_after):
    # found, text_before, text_after = detect_next_inside(begin, end, pre_end, text_after)
    # lst_type = ['BEGIN', 'END', 'PRE_END']
    lst_to_match = [begin, end, pre_end]
    min_pos = -1
    min_pos_idx = -1
    # for idx, (curr_type, curr_match) in enumerate(zip(lst_type, lst_to_match)):
    for idx, curr_match in enumerate(lst_to_match):
        try:
            curr_pos = text_after.index(lst_to_match[idx])
            if min_pos_idx == -1:
                min_pos_idx = idx
                min_pos = curr_pos
            else:
                if curr_pos < min_pos:
                    min_pos = curr_pos
                    min_pos_idx = idx
        except ValueError:
            pass
    if min_pos == -1:
        return None, None, None

    return lst_to_match[min_pos_idx], text_after[:min_pos], \
           text_after[min_pos + len(lst_to_match[min_pos_idx]):]


def recursive_clean_klim(text, begin, end, pre_start, pre_end, min_nr_levels=2,
                         look_pre_in_beginning=True):
    """

    :param text:
    :param begin:
    :param end:
    :param pre_start:
    :param pre_end:
    :param min_nr_levels:
    :param look_pre_in_beginning: looks for pre in the beginning of passed "text", like if
    pre_start is "{{" , this function only will be activated if the first characters of the "text"
    are "{{"
    :return:
    """
    curr_level = 0
    # first looks for pre to happen
    # the pre has to happen on the whole text, not on a piece of text
    is_pre_found, text_before_pre, text_after = detect_next_pre(pre_start, text, look_pre_in_beginning)
    if not is_pre_found:
        return text
    curr_level += 1
    max_nr_levels = curr_level
    while curr_level > 0:
        found, text_before, text_after = detect_next_inside(begin, end, pre_end, text_after)
        if found is None:
            return text

        if found == begin:
            curr_level += 1
            if max_nr_levels < curr_level:
                max_nr_levels = curr_level
        elif curr_level == 1 and found == pre_end:
            if max_nr_levels >= min_nr_levels:
                return text_before_pre + text_after
            else:
                return text  # returns text without changing
        elif curr_level > 1 and found == end:
            curr_level -= 1
        else:
            return text
    # can this happen?
    return text


if __name__ == "__main__":
    print(recursive_clean("This text has {{some}} curly braces {{wrapped {{in }} a {{funny}} manner}} end.", {"{{"},
                          {"}}"}))
    print(recursive_clean("This [[is the]] [[File: has some [[hidden]] atrributes]] end.", {"[["}, {"]]"},
                          {"[[File:", "[[Image:"}))
