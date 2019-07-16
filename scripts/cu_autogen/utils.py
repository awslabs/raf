def format_enum_name(s):
    s = s[6:]
    if not s[0].isalpha():
        s = 'e' + s
    res = []
    last = '_'
    for i in s:
        if i != '_':
            res.append(i.upper() if not last.isalpha() else i.lower())
        last = i
    return ''.join(res)
