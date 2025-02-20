import commandmappings  as cm   
import utildecorators   as ut
import utilfunctions    as uf

class TerminalInterface:
    pass

if __name__ == '__main__':
    print(uf.get_help_message)
    func = uf.get_help_message
    func()