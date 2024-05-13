from tqdm import tqdm
import sys
from tqdm.utils import disp_len

""" This tqdm class prints a newline after each iteration, very useful in combination with wandb logging. """


class NewlineTqdm(tqdm):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @staticmethod
    def status_printer(file):
        """
        Manage the printing and in-place updating of a line of characters.
        Note that if the string is longer than a line, then in-place
        updating may not work (it will print a new line at each refresh).
        """
        fp = file
        fp_flush = getattr(fp, "flush", lambda: None)  # pragma: no cover
        if fp in (sys.stderr, sys.stdout):
            getattr(sys.stderr, "flush", lambda: None)()
            getattr(sys.stdout, "flush", lambda: None)()

        def fp_write(s):
            fp.write(str(s))
            fp_flush()

        last_len = [0]

        def print_status(s):
            len_s = disp_len(s)
            # fp_write("\r" + s + (" " * max(last_len[0] - len_s, 0)))
            # fp_write(s + "\n")
            print(s, flush=True)
            last_len[0] = len_s

        return print_status


if __name__ == "__main__":
    import time

    for i in NewlineTqdm(range(10)):
        time.sleep(0.1)
