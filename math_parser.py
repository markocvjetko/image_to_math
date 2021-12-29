import re

priority = {0:'[0-9]+',
            1:'[0-9]+\-[0-9]+',
            2:'[0-9]+\+[0-9]+',
            3:'[0-9]+\*[0-9]+',
            4:'[0-9]+\/[0-9]+',
            5:'\([^()]*\)'}

#expression = '1+2'

def max_prio(expression):
    for key, value in sorted(priority.items(), key=lambda x: x[0], reverse=True):
        p = re.compile(value)
        m = p.search(expression)
        if m is not None:
            return key
    return None

def parse_expression(expression):
    prio = max_prio(expression)
    temp = priority.get(prio)
    p = re.compile(priority.get(prio))
    m = p.search(expression)
    while(prio > 0):
        temp = priority.get(prio)
        p = re.compile(priority.get(prio))
        m = p.search(expression)

        if prio == 5:
            bracket_result = parse_expression(expression[m.start()+1:m.end()-1])
            expression = expression.replace(m.group(), str(bracket_result))

        if prio == 4:
            match = expression[m.start():m.end()]
            a, b = match.split("/")
            expression = expression.replace(m.group(), str(int(a)//int(b)))
        if prio == 3:
            match = expression[m.start():m.end()]
            a, b = match.split("*")
            expression = expression.replace(m.group(), str(int(a)*int(b)))
        if prio == 2:
            match = expression[m.start():m.end()]
            a, b = match.split("+")
            expression = expression.replace(m.group(), str(int(a)+int(b)))

        if prio == 1:
            match = expression[m.start():m.end()]
            a, b = match.split("-")
            expression = expression.replace(m.group(), str(int(a)-int(b)))

        prio = max_prio(expression)

    return int(expression)


#expression = '10/1+(2-1)*8'
#print(parse_expression(expression))