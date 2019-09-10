import pycparser

class Argument(object):
    
    def __init__(self, const, ty, is_ptr, name):
        self.const = const
        self.type = ty
        self.is_ptr = is_ptr
        self.name = name
        self.parent = None
        self.arg_id = None


    def __str__(self):
        return "%s%s %s%s" % (self.const, self.type, self.is_ptr, self.name)


    def is_attr_field(self):
        # I am not sure if this is too adhoc
        return self.type.endswith('_t') and self.type.startswith('cudnn') and self.type != 'cudnnHandle_t'


def extract_param_info(node):
    class ParamVisitor(pycparser.c_ast.NodeVisitor):

        def __init__(self):
            self.type_name = None
            self.var_name = None
            self.is_ptr = ''
            self.is_const = ''


        def visit_IdentifierType(self, node):
            assert self.type_name is None
            self.type_name = ' '.join(node.names)


        def visit_TypeDecl(self, node):
            self.visit(node.type)
            if u'const' in node.quals:
                self.is_const = 'const '


        def visit_PtrDecl(self, node):
            self.is_ptr = self.is_ptr + '*'
            self.visit(node.type)
            if u'const' in node.quals:
                self.is_const = 'const '


        def visit_Decl(self, node):
            self.var_name = str(node.name)
            self.visit(node.type)

    pv = ParamVisitor()
    pv.visit(node)
    return Argument(pv.is_const, pv.type_name, pv.is_ptr, pv.var_name)


class ExtractVisitor(pycparser.c_ast.NodeVisitor):

    def __init__(self):
        self.parsed_funcs = {}
        self.parsed_enums = {}


    def visit_Decl(self, node):
        if node.name and node.name.startswith('cudnn'):
            if isinstance(node.type, pycparser.c_ast.FuncDecl):
                if isinstance(node.type.args, pycparser.c_ast.ParamList):
                    func_name = str(node.name)
                    args = self.parsed_funcs[func_name] = []
                    for i in node.type.args:
                        info = extract_param_info(i)
                        info.parent = func_name
                        args.append(info)


    def visit_Typedef(self, node):
        if isinstance(node.type, pycparser.c_ast.TypeDecl) and node.name.startswith('cudnn'):
            if isinstance(node.type.type, pycparser.c_ast.Enum):
                e_node = node.type.type
                enums = self.parsed_enums[node.name] = []
                for i in e_node.values.enumerators:
                    enums.append((str(i.name), int(i.value.value.strip('U'))))

def extract_functions(src):
    parser = pycparser.CParser()
    ast = parser.parse(src)
    visitor = ExtractVisitor()
    visitor.visit(ast)
    return visitor
