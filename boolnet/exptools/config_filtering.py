import re
from collections import MutableMapping


def path_value_pairs(item, path=[]):
    ''' recursively traverses dictionary and yields all full paths. '''
    if isinstance(item, MutableMapping):
        for key, subitem in item.items():
            for child in path_value_pairs(subitem, path + [key]):
                yield child
    else:
        yield path, item


def list_regex_match(pattern, path):
    if len(pattern) != len(path):
        return False
    for pat, node in zip(pattern, path):
        if re.fullmatch(pat, node) is None:
            return False
    return True


def filter_keys(source, path_map):
    ''' this filters a dict (potentially containing sub-dicts)
        using a list of key/path pairs. '''
    sink = {}
    for path, val in path_value_pairs(source):
        for sink_key, pattern in path_map:
            if list_regex_match(pattern, path):
                # regex value insertion
                if '{}' in sink_key:
                    # include any sub-paths that were matched by non-trivial
                    # patterns (i.e. sub-path != pattern)
                    s = '_'.join(
                        p for p, pat in zip(path, pattern) if p != pat)
                    sink_key = sink_key.format(s)
                sink[sink_key] = val
                break  # don't match lower priority maps
    return sink
