import os
import io
from os.path import join
import subprocess
from select import select

class DatagenBackend:
    proc = None
    stdin = None
    stdout = None
    stderr = None

    @classmethod
    def start_backend(cls):
        DatagenBackend.proc = subprocess.Popen(
            ['java', '-cp', './external/backend.jar:./external/lib/*', '-ea', 'datagen.Main', 'preverify_server'],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

        DatagenBackend.stdin = io.TextIOWrapper(DatagenBackend.proc.stdin, line_buffering=True)
        DatagenBackend.stdout = io.TextIOWrapper(DatagenBackend.proc.stdout)
        DatagenBackend.stderr = io.TextIOWrapper(DatagenBackend.proc.stderr)

    @classmethod
    def restart_backend(cls):
        if DatagenBackend.proc:
            if DatagenBackend.proc.poll() is None:
                DatagenBackend.proc.kill()
        DatagenBackend.start_backend()

    @classmethod
    def exit_backend(cls):
        if DatagenBackend.proc:
            if DatagenBackend.proc.poll() is None:
                DatagenBackend.proc.kill()
        # DatagenBackend.proc.kill()

    @classmethod
    def exec(cls, command, timeout=1, is_retry=False):
        try:
            # out,err = c.proc.communicate("{}\t{}".format(pred_line, exs_line).encode("utf-8"), timeout=2)
            DatagenBackend.stdin.write(command)
            # print(command)
            r_list, _, _ = select([DatagenBackend.stdout], [], [], 1)
            if r_list:
                out = DatagenBackend.stdout.readline()
            else:
                DatagenBackend.restart_backend()
                raise subprocess.TimeoutExpired(command, timeout)
        except BrokenPipeError as e:
            if is_retry:
                raise e
            else:
                DatagenBackend.restart_backend()
                return cls.exec(command, timeout, is_retry=True)
        except Exception as e:
            DatagenBackend.restart_backend()
            raise e
        return out

def check_regex_equiv(spec0, spec1):
    if spec0 == spec1:
        # print("exact", spec0, spec1)
        return True
    cmd_line = "CHECKEQUIV\t{}\t{}\n".format(spec0, spec1)
    try:
        out = DatagenBackend.exec(cmd_line, timeout=1)
    except subprocess.TimeoutExpired as e:
        return False
    except BrokenPipeError as e:
        return False
    out = out.rstrip()
    return out == "true"



def check_io_consistency(spec, examples):
    # pred_line = " ".join(preds)

    pred_line = "{} {}".format(spec, spec)
    exs_line = " ".join(["{},{}".format(x[1], x[0]) for x in examples])
    cmd_line = "{}\t{}\n".format(pred_line, exs_line)
    try:
        out = DatagenBackend.exec(cmd_line, timeout=1)
    except Exception as e:
        return False

    out = out.rstrip()

    return out == "true"

if __name__=="__main__":
    DatagenBackend.start_backend()
    print('EXPECTED TRUE', check_regex_equiv('or(<low>,<cap>)', 'or(<cap>,<low>)'))
    print('EXPECTED TRUE',check_regex_equiv('or(<low>,<cap>)', '<let>'))
    print('EXPECTED FALSE',check_regex_equiv('concat(<low>,<cap>)', 'concat(<cap>,<low>)'))

    spec = 'and(repeatatleast(or(<low>,or(<cap>,<^>)),1),and(not(startwith(<low>)),and(not(startwith(<^>)),not(contain(concat(notcc(<low>),<^>))))))'
    good_examples = [('ItrdY', '+'), ('JIQD', '+'), ('GAFXvIc^j^l^o^op', '+'), ('WZpg^y^eMrXSfXTqHw^', '+'), ('Y', '+'), ('Jw', '+'), ('cvZpBMcQKAqAXj', '-'), ('X^^mwwSbU^Wk^', '-'), ('ZHQgmLzM^', '-'), ('.-;-g', '-'), (':;A:', '-'), ('Ew^^B^Kcc^zR', '-')]
    bad_examples1 = [('123ItrdY', '+'), ('ItrdY', '+'), ('JIQD', '+'), ('GAFXvIc^j^l^o^op', '+'), ('WZpg^y^eMrXSfXTqHw^', '+'), ('Y', '+'), ('Jw', '+'), ('cvZpBMcQKAqAXj', '-'), ('X^^mwwSbU^Wk^', '-'), ('ZHQgmLzM^', '-'), ('.-;-g', '-'), (':;A:', '-'), ('Ew^^B^Kcc^zR', '-')]
    bad_examples2 = [('ItrdY', '-'), ('JIQD', '+'), ('GAFXvIc^j^l^o^op', '+'), ('WZpg^y^eMrXSfXTqHw^', '+'), ('Y', '+'), ('Jw', '+'), ('cvZpBMcQKAqAXj', '-'), ('X^^mwwSbU^Wk^', '-'), ('ZHQgmLzM^', '-'), ('.-;-g', '-'), (':;A:', '-'), ('Ew^^B^Kcc^zR', '-')]
    print('EXPECTED TRUE',check_io_consistency(spec, good_examples))
    print('EXPECTED FALSE',check_io_consistency(spec, bad_examples1))
    print('EXPECTED FALSE',check_io_consistency(spec, bad_examples2))
    DatagenBackend.exit_backend()
