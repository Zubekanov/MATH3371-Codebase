operator_symbols = ['+', '-', '*', '/', '^', '%']
parentheses = {
    "(" : ")",
    "[" : "]",
    "{" : "}"  
}

class Expression:    
    def __init__(self, expression: str):
        self.symbolic_expression = []
        expression = "".join(expression.split())
        
        curr_pos = 0
        last_pos = 0
        
        while curr_pos < len(expression):
            if curr_pos == len(expression) - 1:
                if expression[curr_pos] in operator_symbols:
                    raise ValueError("Operator at the end of the expression")
                if expression[curr_pos] in parentheses.keys():
                    raise ValueError("Unmatched left parentheses")
                if expression[curr_pos] in parentheses.values():
                    raise ValueError("Unmatched right parentheses")
                self.symbolic_expression.append(Variable(expression[last_pos:len(expression)]))
            if expression[curr_pos] in operator_symbols:
                if len(self.symbolic_expression) > 0 and isinstance(self.symbolic_expression[-1], Expression):
                    self.symbolic_expression.append(Operator(expression[curr_pos]))
                    last_pos = curr_pos + 1
                else:
                    if last_pos >= curr_pos:
                        raise ValueError("Operator without operands")
                    self.symbolic_expression.append(Variable(expression[last_pos:curr_pos]))
                    last_pos = curr_pos + 1
                    self.symbolic_expression.append(Operator(expression[curr_pos]))
            if expression[curr_pos] in parentheses.keys():
                end_parentheses = parentheses[expression[curr_pos]]
                end_pos = len(expression) - 1
                while end_pos > curr_pos:
                    if expression[end_pos] == end_parentheses:
                        self.symbolic_expression.append(Expression(expression[curr_pos + 1:end_pos]))
                        curr_pos = end_pos
                        last_pos = end_pos + 1
                        break
                    end_pos -= 1
                    if end_pos == curr_pos:
                        raise ValueError("Unmatched left parentheses")
            elif expression[curr_pos] in parentheses.values():
                raise ValueError("Unmatched right parentheses")
            curr_pos += 1
    
    def __str__(self):
        res = "("
        for symbol in self.symbolic_expression:
            res += str(symbol)
        res += ")"
        return res
    
class Variable:
    name = None
    
    def __init__(self, name: str):
        self.name = name
    
    def __str__(self):
        #DEBUG
        return "{" + self.name + "}"

class Operator:
    symbol = None
    
    def __init__(self, symbol: str):
        self.symbol = symbol
    
    def __str__(self):
        #DEBUG
        return "[" + self.symbol + "]"

if __name__ == "__main__":
    # Testing errors are thrown properly
    expression = Expression("777 / (dog + cat) * 3")
    print(expression)