#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Converting the E2E Challenge dataset (http://www.macs.hw.ac.uk/InteractionLab/E2E/) to our data format.
"""

from __future__ import unicode_literals


import re
import argparse
import unicodecsv as csv
import codecs
from collections import OrderedDict, Counter
from copy import deepcopy

import os
import sys

# should be able to run from any location now
sys.path.insert(0, os.path.abspath('/home/henrye/downloads/tgen/'))

from nltk.tokenize import sent_tokenize
from tqdm import tqdm

from tgen.data import DA
from tgen.delex import delex_sent
from tgen.futil import tokenize

from tgen.debug import exc_info_hook

# Start IPdb on error in interactive mode
sys.excepthook = exc_info_hook


def filter_abst(abst, slots_to_abstract):
    """Filter abstraction instruction to only contain slots that are actually to be abstracted."""
    return [a for a in abst if a.slot in slots_to_abstract]


def convert(args):
    """Main function – read in the CSV data and output TGEN-specific files."""

    # find out which slots should be abstracted (from command-line argument)
    slots_to_abstract = set()
    if args.abstract is not None:
        slots_to_abstract.update(re.split(r'[, ]+', args.abstract))

    # initialize storage
    conc_das = []
    das = []  # abstracted DAs
    concs = []  # concrete sentences
    texts = []  # abstracted sentences
    absts = []  # abstraction descriptions
    original_sents = []
    delexicalised_sents = []
    sent_ids = []
    mrs_for_delex = []

    # statistics about different DAs
    da_keys = {}
    insts = 0
    find_apostrophes = r"([a-z])\s('[a-z]{1,2}\b)"

    def process_instance(da, conc, mr, multi_ref_id):
        original_da = deepcopy(da)
        # why do the das need to be sorted? This seems weird
        # Anyway, we checked it gets sorted in delex_sent anyway so nothing to
        # do about it until later
        da.sort()
        conc_das.append(da)

        text, da, abst = delex_sent(da, tokenize(conc), slots_to_abstract, args.slot_names, repeated=True)
        # Originall we didn't want to lower case because it will make things
        # easier for udpipe later on, however ...
        # we changed our mind on this because the upper case characters are
        # messing with udpipe's ability to properly sentence tokenize.
        # we need underscores instead of dashes or else udpipe breaks it apart
        # text = re.sub(r"X-", r"X_", text)
        # Again running into problems with leaving x and as a capital letter
        # and also with udpipe randomly segmenting it but sometimes not. We
        # really need to find a more reliable sentence tokenizer / word
        # tokenizer
        text = text.lower().replace('x-', 'x')
        # We're testing out making xnear upper case to see if it reduces the
        # incorrect dropping of it by the deep parser
        text = text.replace('xnear', 'Xnear')

        # detokenize some of the apostrophe stuff because udpipe does it
        # differently. Namely removing spaces between letters and apostrophes
        text = re.sub(find_apostrophes, r"\1\2", text)
        da.sort()

        da_keys[unicode(da)] = da_keys.get(unicode(da), 0) + 1
        das.append(da)
        concs.append(conc)
        absts.append(abst)
        texts.append(text)

        # now for our own bastardized sentence tokenization and human eval
        # required stuff
        this_conc_sents = sent_tokenize(conc)
        num_sents = len(this_conc_sents)
        this_delex_sents = []
        for i, this_conc_sent in enumerate(this_conc_sents):
            text, _, _ = delex_sent(original_da, tokenize(this_conc_sent), slots_to_abstract, args.slot_names, repeated=True)
            text = text.lower().replace('x-', 'x')
            # We're testing out making xnear upper case to see if it reduces the
            # incorrect dropping of it by the deep parser
            text = text.replace('xnear', 'Xnear')
            # detokenize some of the apostrophe stuff because udpipe does it
            # differently. Namely removing spaces between letters and apostrophes
            text = re.sub(find_apostrophes, r"\1\2", text)
            this_delex_sents.append(text)

            # start appending the sentence specific ones
            sent_ids.append('_'.join([mr.replace(' ', ''), str(multi_ref_id),
                                      str(i)]))
            mrs_for_delex.append(mr)

        # now we're onto something else
        original_sents.append('\n'.join(this_conc_sents))
        delexicalised_sents.append('\n'.join(this_delex_sents))

        # this_delex_sents = sent_tokenize(text)
        # num_sents = len(this_conc_sents)
        # if num_sents != len(this_delex_sents):
        #     # this is very bad if this happens!
        #     # import ipdb; ipdb.set_trace()
        #     print '\n'
        #     print this_conc_sents
        #     print this_delex_sents
        #     print '\nnext example'

        # original_sents.append('\n'.join(this_conc_sents))
        # delexicalised_sents.append('\n'.join(this_delex_sents))
        # for i in range(num_sents):
        #     sent_ids.append('_'.join([mr.replace(' ', ''), str(multi_ref_id),
        #                               str(i)]))
        #     mrs_for_delex.append(mr)

    # process the input data and store it in memory
    with open(args.in_file, 'r') as fh:
        csvread = csv.reader(fh, encoding='UTF-8')
        csvread.next()  # skip header
        multi_ref_count = Counter()
        for mr, text in tqdm(csvread):
            multi_ref_count[mr] += 1
            da = DA.parse_diligent_da(mr)
            process_instance(da, text, mr, multi_ref_count[mr])
            insts += 1

        print 'Processed', insts, 'instances.'
        print '%d different DAs.' % len(da_keys)
        print '%.2f average DAIs per DA' % (sum([len(d) for d in das]) / float(len(das)))
        print 'Max DA len: %d, max text len: %d' % (max([len(da) for da in das]),
                                                    max([text.count(' ') + 1 for text in texts]))

    # for multi-ref mode, group by the same conc DA
    if args.multi_ref:
        groups = OrderedDict()
        for conc_da, da, conc, text, abst in zip(conc_das, das, concs, texts, absts):
            group = groups.get(unicode(conc_da), {})
            group['da'] = da
            group['conc_da'] = conc_da
            group['abst'] = group.get('abst', []) + [abst]
            group['conc'] = group.get('conc', []) + [conc]
            group['text'] = group.get('text', []) + [text]
            groups[unicode(conc_da)] = group

        conc_das, das, concs, texts, absts = [], [], [], [], []
        for group in groups.itervalues():
            conc_das.append(group['conc_da'])
            das.append(group['da'])
            concs.append("\n".join(group['conc']) + "\n")
            texts.append("\n".join(group['text']) + "\n")
            absts.append("\n".join(["\t".join([unicode(a) for a in absts_])
                                    for absts_ in group['abst']]) + "\n")
    else:
        # convert abstraction instruction to string (coordinate output with multi-ref mode)
        absts = ["\t".join([unicode(a) for a in absts_]) for absts_ in absts]

    with codecs.open(args.out_name + '-das.txt', 'w', 'UTF-8') as fh:
        for da in das:
            fh.write(unicode(da) + "\n")

    with codecs.open(args.out_name + '-conc_das.txt', 'w', 'UTF-8') as fh:
        for conc_da in conc_das:
            fh.write(unicode(conc_da) + "\n")

    with codecs.open(args.out_name + '-conc.txt', 'w', 'UTF-8') as fh:
        for conc in concs:
            fh.write(conc + "\n")

    with codecs.open(args.out_name + '-abst.txt', 'w', 'UTF-8') as fh:
        for abst in absts:
            fh.write(abst + "\n")

    # We join on double new lines so that udpipe will read them out as
    # different paragraphs
    with codecs.open(args.out_name + '-text.txt', 'w', 'UTF-8') as fh:
        for text in texts:
            fh.write(text + "\n\n")

    # here are all our new ones
    with codecs.open(args.out_name + '-orig_sents.txt', 'w', 'UTF-8') as fh:
        for this in original_sents:
            fh.write(this + "\n")

    # again gets a double new lines for processing with udpipe
    with codecs.open(args.out_name + '-delex_sents.txt', 'w', 'UTF-8') as fh:
        for this in delexicalised_sents:
            fh.write(this + "\n\n")

    with codecs.open(args.out_name + '-sent_ids.txt', 'w', 'UTF-8') as fh:
        for this in sent_ids:
            fh.write(this + "\n")

    with codecs.open(args.out_name + '-mrs_for_delex.txt', 'w', 'UTF-8') as fh:
        for this in mrs_for_delex:
            fh.write(this + "\n")

if __name__ == '__main__':
    argp = argparse.ArgumentParser()
    argp.add_argument('in_file', help='Input CSV file')
    argp.add_argument('out_name', help='Output files name prefix')
    argp.add_argument('-a', '--abstract', help='Comma-separated list of slots to be abstracted')
    argp.add_argument('-m', '--multi-ref',
                      help='Multiple reference mode: relexicalize all possible references', action='store_true')
    argp.add_argument('-n', '--slot-names', help='Include slot names in delexicalized texts', action='store_true')
    args = argp.parse_args()
    convert(args)
