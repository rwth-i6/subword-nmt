#!/usr/bin/env python3

# similar as subword-nmt/get_vocab.py
# similar as /u/peter/experiments/wmt/2017/2017-08-14_de-en/recipe/blocks/scripts/preprocess.py

import os
import sys
from collections import Counter
import subprocess
from argparse import ArgumentParser

my_dir = os.path.dirname(os.path.abspath(__file__))


def iterate_seqs(txt_file, bpe_file):
  """
  :param str txt_file:
  :param str bpe_file:
  """
  dump_corpus_proc = subprocess.Popen(
    ["zcat" if txt_file.endswith(".gz") else "cat", txt_file], stdout=subprocess.PIPE)
  apply_bpe_proc = subprocess.Popen(
    ["%s/apply_bpe.py" % my_dir, "-c", bpe_file], stdin=dump_corpus_proc.stdout, stdout=subprocess.PIPE)
  dump_corpus_proc.stdout.close()
  for line in apply_bpe_proc.stdout:
    yield line.decode("utf8")
  apply_bpe_proc.wait()


def main():
  arg_parser = ArgumentParser()
  arg_parser.add_argument("--txt", required=True)
  arg_parser.add_argument("--bpe", required=True)
  arg_parser.add_argument("--out", default="/dev/stdout")
  arg_parser.add_argument("--unk", default="UNK")
  arg_parser.add_argument("--allow_special_labels", action="store_true")
  args = arg_parser.parse_args()

  symbol_counter = Counter()
  for line in iterate_seqs(txt_file=args.txt, bpe_file=args.bpe):
    for word in line.split():
      symbol_counter[word] += 1

  beginseq, endseq = "<s>", "</s>"
  unk = args.unk
  special_labels = [beginseq, endseq, unk]
  for l in special_labels:
    if args.allow_special_labels:
      if l in symbol_counter:
        del symbol_counter[l]
    else:
      assert l not in symbol_counter, "special token %r used by vocab" % l

  out = open(args.out, "w")
  out.write("{\n")
  out.write("%r: 0,\n" % beginseq)
  out.write("%r: 0,\n" % endseq)
  out.write("%r: 1,\n" % unk)
  # The order in most_common is non-deterministic, due to hashing.
  # Make it deterministic.
  syms = sorted([(-count, symbol) for (symbol, count) in symbol_counter.most_common()])
  for i, (_, symbol) in enumerate(syms):
    out.write("%r: %i,\n" % (symbol, i + 2))
  out.write("}\n")


if __name__ == "__main__":
  import better_exchook
  better_exchook.install()
  main()
