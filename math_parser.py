"""
Parses and solves a math expression. Does not check if the expression is valid.
Finds operators with highest priority, calculates their output and inserts the output
in place of the operator and operands. The process is repeated until only a number remains.
"""
import re

#Elements (operators) of a math expression as regex-es, alongside their priority. Key value switch
# might be needed to represent elements of same priority.

priority = {0:'[0-9]+',
            1:'[0-9]+\-[0-9]+',
            2:'[0-9]+\+[0-9]+',
            3:'[0-9]+\*[0-9]+',
            4:'[0-9]+\/[0-9]+',
            5:'\([^()]*\)'}

def max_prio(expression):
    """
    Searches for operators and their operands as regex-es. Returns the priority of the
    found operator with the highest priority.

    :param expression:
    :return:
    """
    for key, value in sorted(priority.items(), key=lambda x: x[0], reverse=True):
        p = re.compile(value)
        m = p.search(expression)
        if m is not None:
            return key
    return None

def parse_expression(expression):       #works, needs refactoring, BADLY!
    """

    Parses and solves an integer math expression. Does not check if the expression is valid.
    Finds operators with the highest priority, calculates their output and inserts the output
    in place of the operator and operands. The process is repeated until only a number remains.

    :param expression: Math expression as string
    :return: expresion result as integer
    """
    try:
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

    except:
        print("Error, invalid math expression")
