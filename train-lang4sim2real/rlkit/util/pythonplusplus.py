"""
General purpose Python functions.
"""
import collections


def identity(x):
    return x


def list_of_dicts__to__dict_of_lists(lst, enforce_consistent_keys=True):
    """
    ```
    x = [
        {'foo': 3, 'bar': 1},
        {'foo': 4, 'bar': 2},
        {'foo': 5, 'bar': 3},
    ]
    ppp.list_of_dicts__to__dict_of_lists(x)
    # Output:
    # {'foo': [3, 4, 5], 'bar': [1, 2, 3]}
    ```
    """
    if len(lst) == 0:
        return {}
    keys = lst[0].keys()
    output_dict = collections.defaultdict(list)
    for d in lst:
        if set(d.keys()) != set(keys):
            print("dropping some keys", d.keys())
        if enforce_consistent_keys:
            assert set(d.keys()) == set(keys)
        for k in keys:
            output_dict[k].append(d[k])
    return output_dict


def safe_json(data):
    if data is None:
        return True
    elif isinstance(data, (bool, int, float)):
        return True
    elif isinstance(data, (tuple, list)):
        return all(safe_json(x) for x in data)
    elif isinstance(data, dict):
        return all(
            isinstance(k, str) and safe_json(v) for k, v in data.items())
    return False


def dict_to_safe_json(d, sort=False):
    """
    Convert each value in the dictionary into a JSON'able primitive.
    :param d:
    :return:
    """
    if isinstance(d, collections.OrderedDict):
        new_d = collections.OrderedDict()
    else:
        new_d = {}
    for key, item in d.items():
        if safe_json(item):
            new_d[key] = item
        else:
            if (
                    isinstance(item, dict)
                    or isinstance(item, collections.OrderedDict)
            ):
                new_d[key] = dict_to_safe_json(item, sort=sort)
            else:
                new_d[key] = str(item)
    if sort:
        return collections.OrderedDict(sorted(new_d.items()))
    else:
        return new_d


"""
Itertools++
"""


def batch(iterable, n=1):
    """
    Split an interable into batches of size `n`. If `n` does not evenly divide
    `iterable`, the last slice will be smaller.

    https://stackoverflow.com/questions/8290397/how-to-split-an-iterable-in-constant-size-chunks

    Usage:
    ```
        for i in batch(range(0,10), 3):
            print i

        [0,1,2]
        [3,4,5]
        [6,7,8]
        [9]
    ```
    """
    le = len(iterable)
    for ndx in range(0, le, n):
        yield iterable[ndx:min(ndx + n, le)]
