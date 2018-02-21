#!/usr/bin/env python3

# similar as subword-nmt/get_vocab.py
# similar as /u/peter/experiments/wmt/2017/2017-08-14_de-en/recipe/blocks/scripts/preprocess.py

import sys
from collections import Counter
import subprocess
from argparse import ArgumentParser


def iterate_seqs(txt_file, bpe_file):
  dump_corpus_proc = subprocess.Popen(["cat", txt_file], stdout=subprocess.PIPE)
  apply_bpe_proc = subprocess.Popen(
    ["./subword-nmt/apply_bpe.py", "-c", bpe_file], stdin=dump_corpus_proc.stdout, stdout=subprocess.PIPE)
  dump_corpus_proc.stdout.close()
  for line in apply_bpe_proc.stdout:
    yield line.decode("utf8")
  apply_bpe_proc.wait()


def main():
  arg_parser = ArgumentParser()
  arg_parser.add_argument("--txt", required=True)
  arg_parser.add_argument("--bpe", required=True)
  arg_parser.add_argument("--out", default="/dev/stdout")
  args = arg_parser.parse_args()

  symbol_counter = Counter()
  for line in iterate_seqs(bpe_file=args.bpe):
    for word in line.split():
      symbol_counter[word] += 1

  out = open(args.out, "w")
  out.write("{\n")
  out.write("'<s>': 0,\n")
  out.write("'</s>': 0,\n")
  out.write("'UNK': 1,\n")
  for i, (symbol, count) in enumerate(symbol_counter.most_common()):
    out.write("%r: %i,\n" % (symbol, i + 2))
  out.write("}\n")


if __name__ == "__main__":
  import better_exchook
  better_exchook.install()
  main()
