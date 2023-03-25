
"""
Module argvproc

[argvproc_module]

Simple Example:
    from argvproc import ArgVProcessor, ArgVTypes
    argvproc = ArgVProcessor(typed_keyword_only=False).process
    pargs, kwarg, index, pos = process(sys.argv, start=1)
    print(pargs, kwarg, index, pos)
"""

import os, sys
from numbers import Integral, Number
from collections.abc import Callable, Iterable, Sized, Sequence
from collections.abc import Mapping
from pprint import pprint


def content_to_argv(content, start=0, stop=None,
                    quotes=(("\"\"\"", "\"\"\""),
                            ("'''", "'''"),
                            ("\"", "\""),
                            ("'", "'")),
                    sep=None):
    squotes = tuple(q[0] for q in quotes)
    equotes = tuple(q[1] for q in quotes)
    seplen = (0 if sep is None else len(sep))
    start, stop, _ = slice(0, None, 1).indices(len(content))
    argv, arg, quoted = [], [], None
    prev_index, index = None, start
    while start <= index < stop:
        if prev_index == index:
            raise RuntimeError("index not progressing at {}"
                               .format(index))
        c = content[index]
        prev_index, index = index, index + len(c)
        if c == "\\":
            if index < stop and content[index].isspace():
                index += 1
                arg.append(content[prev_index + 1:index])
            else:
                index += 1
                arg.append(content[prev_index:index])
            continue
        if not quoted:
            if sep:
                i = prev_index + seplen
                if i < stop and content[prev_index:i] == sep:
                    argv.append(''.join(arg))
                    arg = []
                    index = i
                    continue
            elif c.isspace():
                if arg:
                    argv.append(''.join(arg))
                    arg = []
                continue
            if not arg:
                if content.startswith(squotes, prev_index, stop):
                    for i, q in enumerate(squotes):
                        if content[prev_index:prev_index+len(q)] == q:
                            quoted = equotes[i]
                            index = prev_index + len(q)
                            break
                    if quoted:
                        continue
        elif content.startswith(quoted, prev_index, stop):
            argv.append(''.join(arg))
            arg = []
            quoted = None
            continue
        arg.append(c)
    else:
        if quoted:
            raise ValueError("unclosed {!r} quote".format(quoted))
        if arg:
            argv.append(''.join(arg))
            arg = []
    return tuple(arg.encode("utf8").decode(encoding="unicode-escape")
                 for arg in argv)


class StopRequest(StopIteration):
    "Raised to end processing of arguments."
    pass


class SkipRequest(Exception):
    "Raised to skip processing of a single argument."
    pass


class ArgumentError(Exception):
    "Raised when an argument is unacceptable."
    pass


def UIntish(text):
    s = text.strip()
    if not s:
        return None
    if 0 <= (result := int(s)):
        return result
    raise ValueError("invalid literal for uint() with base 10: {!r}"
                     .format(text))

# common sequence types
def typed_n_sequence(text, item_type=str, occurs=(0, None),
                     sequence_type=tuple, allow_none=False):
    if isinstance(occurs, int):
        min = max = occurs
    elif not (isinstance(occurs, tuple)
              and 1 <= (n := len(occurs)) <= 2
              and all(c is None or type(c) is int for c in occurs)):
        raise ValueError("occurs expects an int or 2-tuple(int)"
                         " but found {!r}".format(occurs))
    else:
        min, max = (occurs, None) if n == 1 else occurs
    stripped = text.strip()
    if stripped:
        if max is None:
            result = tuple(item_type(c.strip())
                           for c in s.split(","))
        else:
            result = tuple(item_type(c.strip())
                           for c in s.split(",", max-1))
        n = len(result)
        if not ((min is not None and min > n)
                 or (max is not None and max < n)):
            return result
    elif allow_none:
        return None
    raise ArgumentError(repr(text))

def IntAreaTuple(text):
    try:
        return typed_n_sequence(text, int, 4)
    except ArgumentError:
        pass
    return (0, 0, *typed_n_sequence(text, int, 2))

def FloatAreaTuple(text):
    try:
        return typed_n_sequence(text, float, 4)
    except ArgumentError:
        pass
    return (0, 0, *typed_n_sequence(text, float, 2))

def IntColor(text):
    try:
        return typed_n_sequence(text, int, (3, 4), allow_none=True)
    except ValueError:
        pass
    return text.strip()  # hex or named color

def FloatColor(text):
    try:
        return typed_n_sequence(text, float, (3, 4), allow_none=True)
    except ValueError:
        pass
    return text.strip()  # hex or named color


ArgVTypes = type("ArgVTypes", (),
                 dict(UIntish = UIntish,
                      Intish = (lambda s: int(s) if s else None),
                      FloatScalar = (lambda s: float(s)
                                     if s else None),
                      Boolish = (lambda s: None if not s else
                                 True
                                 if s.upper() == "TRUE"[:len(s)] else
                                 False
                                 if s.upper() == "FALSE"[:len(s)] else
                                 bool(int(s))),
                      Stringish = (lambda s: (s if s else None)),
                      StrTuple = (lambda s: typed_n_sequence(s)),
                      IntTuple = (lambda s: typed_n_sequence(s, int)),
                      FloatTuple = (lambda s:
                                    typed_n_sequence(s, float)),
                      Int2Tuple = (lambda s:
                                   typed_n_sequence(s, int, 2)),
                      Float2Tuple = (lambda s:
                                     typed_n_sequence(s, float, 2)),
                      Int4Tuple = (lambda s:
                                   typed_n_sequence(s, int, 4)),
                      Float4Tuple = (lambda s:
                                     typed_n_sequence(s, float, 4)),
                      IntAreaTuple = IntAreaTuple,
                      FloatAreaTuple = FloatAreaTuple,
                      IntColor = IntColor,
                      FloatColor = FloatColor))


class ArgVProperties(tuple):

    def __new__(cls, argv,
                index=None, start=None, stop=None, position=0,
                pargs=None, kwargs=None, **extra):
        start, stop, _ = slice(start, stop, 1).indices(len(argv))
        if index is None:
            index = start
        if pargs is None:
            pargs = []
        if kwargs is None:
            kwargs = {}
        self = super().__new__(cls,
                               (argv,
                                index, start, stop, position,
                                pargs, kwargs))
        self._extra = extra
        return self

    @property
    def argv(self):
        return self[0]

    @property
    def index(self):
        return self[1]

    @property
    def start(self):
        return self[2]

    @property
    def stop(self):
        return self[3]

    @property
    def position(self):
        return self[4]

    @property
    def pargs(self):
        return self[5]

    @property
    def kwargs(self):
        return self[6]

    _extra = None

    @property
    def extra(self):
        return self._extra

    def __getattr__(self, name):
        try:
            return self._extra[name]
        except KeyError:
            pass
        raise AttributeError("AttributeError: {!r} object has no"
                             " attribute {!r}"
                             .format(type(self).__name__, name))

    def __dir__(self):
        return ('index', 'start', 'stop', 'position',
                'pargs', 'kwargs', *self._extra.keys())


class ArgVProcessor:

    def __new__(cls,
                arg_aliases={},
                operations=(),
                commands={},
                positional_keywords={},
                keyword_aliases={},
                positional_types={},
                keyword_types={},
                assop="=",
                keyword_replace=False,
                typed_keyword_only=True):
        if not (assop and type(assop) is str):
            raise ValueError("assop expects a non-empty str"
                             " but found {!r}".format(assop))
        self = super().__new__(cls)
        self.arg_aliases = dict(arg_aliases)
        self.operations = dict(operations)
        self.commands = dict(commands)
        self.positional_keywords = tuple(positional_keywords)
        self.keyword_aliases = dict(keyword_aliases)
        self.positional_types = tuple(positional_types)
        self.keyword_types = dict(keyword_types)
        self.assop = assop
        self.keyword_replace = bool(keyword_replace)
        self.typed_keyword_only = bool(typed_keyword_only)
        return self

    arg_aliases = None
    operations = None
    commands = None
    positional_keywords = None
    keyword_aliases = None
    positional_types = None
    keyword_types = None
    assop = None
    keyword_replace = None
    typed_keyword_only = None

    def process(self,
                argv, index, start=0, stop=None, start_position=0,
                pargs=None, kwargs=None, **extra):
        argc = len(argv)
        start, stop, _ = slice(start, stop, 1).indices(argc)
        if pargs is None:
            pargs = []
        if kwargs is None:
            kwargs = {}
        pos = start_position
        prev_index = None
        while start <= index < stop:
            if prev_index == index:
                raise RuntimeError("Argument Index not progressing"
                                   " at {!r} ({!r})"
                                   .format(index, argv[index]))
            arg, key, value = argv[index], None, None
            prev_index, index = index, index + 1
            arg = self.arg_aliases.get(arg, arg)
            snapshot = ArgVProperties(argv, index, start, stop, pos,
                                      pargs, kwargs, **extra)
            result = None
            if arg in self.operations:
                operator = self.operations[arg]
                try:
                    result = operator(snapshot)
                except SkipRequest:
                    continue
                except StopRequest:
                    index = prev_index
                    break
            elif arg in self.commands:
                try:
                    result = self.process_command(self.commands[arg],
                                                  snapshot)
                except SkipRequest:
                    continue
                except StopRequest:
                    index = prev_index
                    break
            if result is not None:
                if not isinstance(result, ArgVProperties):
                    raise TypeError("operators and commands must"
                                    " return None or an ArgVProperties"
                                    " instance but returned {!r}"
                                    .format(result))
                argv, index, start, stop, pos, pargs, kwargs = result
            try:
                i = arg.index(self.assop, 1)
            except ValueError:
                key = value = None
            else:
                key, value = arg[:i], arg[i+len(self.assop):]
            if key is None:
                try:
                    key = self.positional_keywords[pos]
                except IndexError:
                    try:
                        typer = self.positional_types[pos]
                    except IndexError:
                        pass
                    else:
                        arg = typer(arg)
                else:
                    value = arg
                pos += 1
            if key is not None:
                key = self.keyword_aliases.get(key, key)
                try:
                    assert key.isidentifier()
                    typer = self.keyword_types[key]
                except (KeyError, AssertionError):
                    if self.typed_keyword_only:
                        key = Value = None
                    else:
                        value = arg[i+len(self.assop):].strip()
                else:
                    value = typer(arg[i+len(self.assop):].strip())
            #print(arg, key, value)
            try:
                if key is None:
                    self.process_parg(arg, snapshot)

                else:
                    if not self.keyword_replace and key in kwargs:
                        raise ArgumentError("Duplicate keyword {!r},"
                                            " new={!r}, old={!r}"
                                            .format(key, value,
                                                    kwargs[key]))
                    self.process_kwarg(key, value, snapshot)
            except SkipRequest:
                continue
            except StopRequest:
                index = prev_index
                break
        return pargs, kwargs, index, pos

    def process_command(self, command, argvproc):
        raise StopRequest("Command {!r} at {!r}"
                         .format(command, argvproc.index))

    def process_parg(self, arg, argvproc):
        argvproc.pargs.append(arg)

    def process_kwarg(self, key, value, argvproc):
        argvproc.kwargs[key] = value


def mine_topics(docs=__doc__, library=None, prefix=None):
    topics = {}
    i, n, prev_name, prev_i, first = 0, len(docs), None, 0, None
    while i < n:
        i0 = i1 = i
        name = (docs[i0+2:i1]
                if (i0 := docs.find("\n[", i)) >= 0
                and (i1 := docs.find("]", i0+1)) >= 0 else None)
        if name and name.isprintable():
            topics[prev_name] = docs[prev_i:i0].strip()
            prev_name, prev_i = name, i1 + 1
        i = max(i0, i1, prev_i) + 1
    else:
        if prev_name is not None:
            topics[prev_name] = docs[prev_i:].strip()
    if library is not None:
        prefix = (prefix + "." if prefix else '')
        for name, content in topics.items():
            full_name = prefix + name
            if full_name in library:
                if content != library[full_name]:
                    warnings.warn("Topic collision for {!r},"
                                  " ignoring new content: {!r}"
                                  .format(full_name, content))
            else:
                library[full_name] = content
    return topics


"""
>>> def foo(a, b, c=None): pass
...
>>> dir(foo)
['__annotations__', '__builtins__', '__call__', '__class__', '__closure__', '__code__', '__defaults__', '__delattr__', '__dict__', '__dir__', '__doc__', '__eq__', '__format__', '__ge__', '__get__', '__getattribute__', '__globals__', '__gt__', '__hash__', '__init__', '__init_subclass__', '__kwdefaults__', '__le__', '__lt__', '__module__', '__name__', '__ne__', '__new__', '__qualname__', '__reduce__', '__reduce_ex__', '__repr__', '__setattr__', '__sizeof__', '__str__', '__subclasshook__']
>>> foo.__annotations__
{}
>>> foo.__qualname__
'foo'
>>> foo.__defaults__
(None,)
>>> foo.__kwdefaults__
>>> foo.__name__
'foo'
>>> foo.__code__
<code object foo at 0x000001A1502FD9A0, file "<stdin>", line 1>
>>> dir(foo.__code__)
['__class__', '__delattr__', '__dir__', '__doc__', '__eq__', '__format__', '__ge__', '__getattribute__', '__gt__', '__hash__', '__init__', '__init_subclass__', '__le__', '__lt__', '__ne__', '__new__', '__reduce__', '__reduce_ex__', '__repr__', '__setattr__', '__sizeof__', '__str__', '__subclasshook__', 'co_argcount', 'co_cellvars', 'co_code', 'co_consts', 'co_filename', 'co_firstlineno', 'co_flags', 'co_freevars', 'co_kwonlyargcount', 'co_lines', 'co_linetable', 'co_lnotab', 'co_name', 'co_names', 'co_nlocals', 'co_posonlyargcount', 'co_stacksize', 'co_varnames', 'replace']
>>> foo.__code__.co_argcount
3
>>> foo.__code__.co_cellvars
()
>>> foo.__code__.co_consts
(None,)
>>> foo.__code__.co_flags
67
>>> foo.__code__.co_freevars
()
>>> foo.__code__.co_kwonlyargcount
0
>>> foo.__code__.co_name
'foo'
>>> foo.__code__.co_names
()
>>> foo.__code__.co_nlocals
3
>>> foo.__code__.co_posonlyargcount
0
>>> foo.__code__.co_stacksize
1
>>> foo.__code__.co_varnames
('a', 'b', 'c')
"""


def command_noop(props):
    return props.index


def command_stop_request(props):
    raise StopRequest("Termination request")


def command_truncate(props):
    return ArgVProperties(props.argv, props.index,
                          props.start, props.index, 0)


def mine_functions(module,
                   func_prefix=None, func_suffix=None,
                   library=None, library_prefix=None):
    if module is None:
        module = sys.modules[__name__]
    functions = dict((name[len(func_prefix):].replace("_", "-"), thing)
                    for name, thing in vars(module).items()
                    if name.startswith(func_prefix) and callable(thing))
    if library is not None:
        library_prefix = (library_prefix
                          + ("." if library_prefix else ''))
        for name, func in functions.items():
            full_name = library_prefix + name
            if full_name in library:
                if func is not library[full_name]:
                    warnings.warn("Command collision for {!r},"
                                  " ignoring new function: {!r}"
                                  .format(full_name, func))
            else:
                library[full_name] = func
    return functions


if __name__ == "__main__":
    from pprint import pprint
    arg_aliases = {"-q": "verbosity=0",
                   "-v": "verbosity=2",
                   "-vv": "verbosity=3",
                   "-vvv": "verbosity=4",
                   "-vvvv": "verbosity=5"}
    operations = ()
    def command_dump_props(props):
        argv, index, start, stop, pos, _, _ = props
        print("  index", index)
        pargs, kwargs, index, pos = argvproc.process(argv, index)
        print("  pargs", pargs)
        print("  kwargs", kwargs)
        print("  index", index)
        print("  pos", pos)
        return ArgVProperties(argv, index, start, stop)
    def command_process_from_file(props):
        argv, index, start, stop, pos, _, _ = props
        if index >= stop:
            raise ArgumentError("Expected a options filepath")
        filepath = argv[index]
        index += 1
        if filepath == "-":
            content = sys.stdin.read()
        else:
            with open(filepath, "r") as f:
                content = f.read()
        child_argv = content_to_argv(content)
        main(child_argv)
        return index
    commands = {"--": command_stop_request,
                "noop": command_noop,
                "dump": command_dump_props,
                "process-file": command_process_from_file}
    keyword_aliases = {}
    positional_types = {}
    keyword_types = dict(number=ArgVTypes.Intish)
    keyword_replace = False
    positional_keywords = {}
    argvproc = ArgVProcessor(arg_aliases=arg_aliases,
                             operations=operations,
                             commands=commands,
                             keyword_aliases=keyword_aliases,
                             positional_types=positional_types,
                             keyword_types=keyword_types,
                             keyword_replace=keyword_replace,
                             positional_keywords=positional_keywords,
                             typed_keyword_only=False)
    def main(argv = sys.argv[1:]):
        start, stop, _ = slice(0, None, 1).indices(len(argv))
        print("argv", argv)
        pargs, kwargs, index, pos = argvproc.process(argv, start)
        if pargs:
            print("invalid command {!r}".format(pargs[0]))
            return
        global_kwargs = kwargs
        print("global_kwargs")
        pprint(global_kwargs)
        pargs, kwargs = [], {}
        prev_index, pos = None, 0
        while index < stop:
            if prev_index == index:
                raise IndexError("index not progressing at {}"
                                 .format(index))
            prev_index = index
            command_name = argv[index]
            index += 1
            if command_name not in commands:
                print("Command {!r} does not exist".format(command_name))
                return
            command = commands[command_name]
            print("Command {!r} ({})".format(command_name, command))
            try:
                result = command(ArgVProperties(argv, index, start, stop, pos))
                if result is not None:
                    if isinstance(result, ArgVProperties):
                        index, pos = result.index, result.position
                        start, stop = result.start, result.stop
                        pargs, kwargs = result.pargs, result.kwargs
                    elif isinstance(result, int):
                        index = result
            except SkipRequest:
                continue
            except StopRequest:
                break

    main()

