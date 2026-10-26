import json
import math
import re

INVALID = object()


def extract_answer(text):
    matches = re.findall(r'<answer>(.*?)</answer>', text, flags=re.DOTALL)
    if len(matches) != 1:
        return INVALID
    value = matches[0].strip()
    if not value or '<' in value or '>' in value:
        return INVALID
    if value.startswith('['):
        try:
            answer = json.loads(value)
            return answer if isinstance(answer, list) and all(isinstance(v, str) for v in answer) else INVALID
        except ValueError:
            return INVALID
    if value.lower() in ('null','nan'):
        return None
    if re.fullmatch(r'[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?',value):
        number=float(value)
        return number if math.isfinite(number) else INVALID
    return value


def canonical(value):
    if value is INVALID:
        return INVALID
    if value is None:
        return 'nan'
    if isinstance(value,list):
        return canonical(value[0]) if len(value)==1 else tuple(sorted(set(canonical(v) for v in value)))
    try:
        number=float(value)
        if math.isnan(number):
            return 'nan'
        if not math.isfinite(number):
            return INVALID
        return f'{0.0 if number == 0 else number:.2f}'
    except (TypeError,ValueError):
        return str(value).strip()


def score(response, label):
    answer=extract_answer(response)
    prediction, reference=canonical(answer), canonical(label)
    return {'format_valid':answer is not INVALID,
            'prediction':None if prediction is INVALID else prediction,
            'reference':reference,
            'published_label_match':prediction is not INVALID and prediction==reference}
