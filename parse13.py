#!/usr/bin/env python3
"""
Determine whether sentences are grammatical under a CFG, using Earley's algorithm.
(Starting from this basic recognizer, you should write a probabilistic parser
that reconstructs the highest-probability parse of each given sentence.)
"""

# Recognizer code by Arya McCarthy, Alexandra DeLucia, Jason Eisner, 2020-10, 2021-10.
# This code is hereby released to the public domain.

from __future__ import annotations
import argparse
import logging
import math
import pdb

import tqdm
from dataclasses import dataclass
from pathlib import Path
from collections import Counter
from typing import Counter as CounterType, Iterable, List, Optional, Dict, Tuple
import pdb

log = logging.getLogger(Path(__file__).stem)  # For usage, see findsim.py in earlier assignment.


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "grammar", type=Path, help="Path to .gr file containing a PCFG'"
    )
    parser.add_argument(
        "sentences", type=Path, help="Path to .sen file containing tokenized input sentences"
    )
    parser.add_argument(
        "-s",
        "--start_symbol",
        type=str,
        help="Start symbol of the grammar (default is ROOT)",
        default="ROOT",
    )

    verbosity = parser.add_mutually_exclusive_group()
    verbosity.add_argument(
        "-v",
        "--verbose",
        action="store_const",
        const=logging.DEBUG,
        default=logging.INFO,
    )
    verbosity.add_argument(
        "-q", "--quiet", dest="verbose", action="store_const", const=logging.WARNING
    )

    parser.add_argument(
        "--progress",
        action="store_true",
        help="Display a progress bar",
        default=False,
    )

    args = parser.parse_args()
    return args


class EarleyChart:
    """A chart for Earley's algorithm."""

    def __init__(self, tokens: List[str], grammar: Grammar, progress: bool = False) -> None:
        """Create the chart based on parsing `tokens` with `grammar`.
        `progress` says whether to display progress bars as we parse."""
        self.tokens = tokens
        self.grammar = grammar
        self.progress = progress
        self.profile: CounterType[str] = Counter()
        self.maxWeights = float("inf");


        self.cols: List[Agenda]
        self._run_earley()  # run Earley's algorithm to construct self.cols

    def accepted(self) -> bool:
        """Was the sentence accepted?
        That is, does the finished chart contain an item corresponding to a parse of the sentence?
        This method answers the recognition question, but not the parsing question."""
        pdb.set_trace()
       
        for item in self.cols[-1].all():  # the last column
            if (item.rule.lhs == self.grammar.start_symbol  # a ROOT item in this column
                    and item.next_symbol() is None  # that is complete
                    and item.start_position == 0):  # and started back at position 0
                max(self.maxWeights, self.weights[item])
                
                
                return True
        return False  # we didn't find any appropriate item

    def returnMaxProbability(self) -> float:
        """Was the sentence accepted?
                That is, does the finished chart contain an item corresponding to a parse of the sentence?
                This method answers the recognition question, but not the parsing question."""

        for item in self.cols[-1].all():  # the last column
            if (item[0].rule.lhs == self.grammar.start_symbol  # a ROOT item in this column
                    and item[0].next_symbol() is None  # that is complete
                    and item[0].start_position == 0):  # and started back at position 0
                self.maxWeights = item[2];
                self.maxWeights = min(self.maxWeights, item[2])
               
        if (self.maxWeights == float("inf")):
            self.maxWeights = 0;
       
        return self.maxWeights  # returns maxProb of Parse.

    def printItem(self, item):
        print(item)
        self.printItem(item[1])
        if type(item) is tuple:
            print(item[0].rule)
            if type(item[1]) is tuple:
                self.printItem(item[1][0])
            else:
                self.printItem(item[1])
      
        ##print rule of this item 
        # if item is None:
        #     return
        # print(item.rule)
        # if type(item) is Item:
        #     print(item.rule)
        #     self.printItem(item)
        # else:
        #     print(item[0].rule)

        # if type(item[1]) is not tuple:
        #     print(item)
        #     self.printItem(item[1])
        # else:
        #     self.printItem(item[1][0])
        #     self.printItem(item[1][1][0])
        # print("Rule: {}".format(item[0]))
        # # if item[1] is not None:
        # #     print(item[1])
        # #     self.printItem(item[1][0][0])
        # # ##print the backpointer of this item
        # print("backpointer: {}".format(item[1]))
        # item = item[1]
        # if type(item) is not tuple:
        #     self.printItem(item[1])
        # else:
        #     self.printItem(item[0])
        #     self.printItem(item[1][0])
            # if item[0] is not None:
            #     if len(item) == 1:
            #        # print(last_item[0].rule)
            #         self.printItem(item[0][0])
            #         return 
            #     else:
            #         #last_item = last_item[0][1].backpointer
            #        # print("last item : {}".format(last_item))
            #         print(f"({last_item[0][1].rule.lhs}({last_item[0][1].rule.rhs})") 
            #         self.printItem(last_item[0][1].backpointer) 
            #         #print(f"({last_item[1].rule.lhs}")
            #         print(f"({last_item[1].rule.lhs}({last_item[1].rule.rhs}))")
                   # print(last_item[0][1].rule.lhs )
                    #print(last_item[0][1].rule.rhs)

                   ##### self.printItem(last_item[0][1].backpointer) #######

                   ##### print(f"({last_item[1].rule.lhs}({last_item[1].rule.rhs}") ######

            #         self.printItem(last_item[1].backpointer)
            #         return 
            # else:
            #     print(f"({last_item[1].rule.lhs} (")
            #     #print(f"({last_item[1].rule.lhs}({last_item[1].rule.rhs}))")
            #     self.printItem(last_item[1].backpointer)
            #     return 
            
        # ###get the index of the backpointer of this item 
        ##index = self.cols[item[1][0][0].start_position].getIndex(item[1])
        # # #index = self.cols[item[1][0][0].start_position].getIndex(item[0][0][0])
        # # print(index)
        # # ###get that item by index
        # item_tuple = self.cols[item[0].start_position].getItem(index)
        # print("Item tuple: {}".format(item_tuple))
        # # ##call recursion on that item 
        # self.printItem(item_tuple[0][0])
       
        
       # if not item[1]:
            #return
       # index = self.cols[item[1][0]
        #item = item[1][0]
        

        #print(item[1][0][0])
       # index = self.cols[item[1][0][0].start_position].getIndex(item[1])
        # print(index)
        # item = self.cols[item[1][0][0].start_position].getItem(index)
        # print(item)
        #pdb.set_trace()
       # self.printItem(item)
        # recursively print the parse tree from the chart of backpointers

        return;
    
    def helper_print(self) :
        for item in self.cols[-1].all():  # the last column
            if (item[0].rule.lhs == self.grammar.start_symbol  # a ROOT item in this column
                    and item[0].next_symbol() is None  # that is complete
                    and item[0].start_position == 0):  # and started back at position 0
                    #print(item)
                    print(self.printItem(item))


    def _run_earley(self) -> None:
        """Fill in the Earley chart"""
        # Initially empty column for each position in sentence
        self.cols = [Agenda() for _ in range(len(self.tokens) + 1)]

        # Start looking for ROOT at position 0
        self._predict(self.grammar.start_symbol, 0)

        # We'll go column by column, and within each column row by row.
        # Processing earlier entries in the column may extend the column
        # with later entries, which will be processed as well.
        #
        # The iterator over numbered columns is `enumerate(self.cols)`.
        # Wrapping this iterator in the `tqdm` call provides a progress bar.
        from tqdm import tqdm

        for i, column in tqdm(enumerate(self.cols),
                              total=len(self.cols),
                              disable=not self.progress):
            log.debug("")
            log.debug(f"Processing items in column {i}")
            while column:  # while agenda isn't empty
                item = column.pop()  # dequeue the next unprocessed item
                next = item[0].next_symbol();
                if next is None:  # if there is nothing u find its customer and attach.
                    # Attach this complete constituent to its customers
                    log.debug(f"{item} => ATTACH")
                    self._attach(item, i)
                elif self.grammar.is_nonterminal(next):
                    # Predict the nonterminal after the dot
                    log.debug(f"{item} => PREDICT")
                    self._predict(next, i)
                else:
                    # Try to scan the terminal after the dot
                    log.debug(f"{item} => SCAN")
                    self._scan(item, i)

    def _predict(self, nonterminal: str, position: int) -> None:
        """Start looking for this nonterminal at the given position."""
        for rule in self.grammar.expansions(nonterminal):
            new_item = Item(rule, dot_position=0, start_position=position)
            self.cols[position].push(new_item, None, rule.weight)
            log.debug(f"\tPredicted: {new_item} in column {position}")
            self.profile["PREDICT"] += 1

    def _scan(self, item: Item, position: int) -> None:
        """Attach the next word to this item that ends at position,
        if it matches what this item is looking for next."""
        # keeps the backpointer of the rules
        if position < len(self.tokens) and self.tokens[position] == item[0].next_symbol():
            new_item = item[0].with_dot_advanced()
            self.cols[position + 1].push(new_item, (None,item[0]), item[2])
            #trying reprocessing
            log.debug(f"\tScanned to get: {new_item} in column {position + 1}")
            self.profile["SCAN"] += 1

    def _attach(self, item: Item, position:int) -> None:
        """Attach this complete item to its customers in previous columns, advancing the
        customers' dots to create new items in this column.  (This operation is sometimes
        called "complete," but actually it attaches an item that was already complete.)
        """
        # backponter
        # backpointer datastrcture
        mid = item[0].start_position  # start position of this item = end position of item to its left

        for customer in self.cols[mid].all():  # could you eliminate this inefficient linear search?
            if customer[0].next_symbol() == item[0].rule.lhs:
                new_item = customer[0].with_dot_advanced()
                index = self.cols[position].checkExist(new_item);
                if(index > 0):
                    if min(self.cols[position].getWeights(index), item[2] + customer[2]) == (item[2] + customer[2]):
                        self.cols[position].reprocess(new_item, (customer,item), item[2] + customer[2]);
                else:
                    self.cols[position].push(new_item, (customer, item), item[2] + customer[2])
                    log.debug(f"\tAttached to get: {new_item} in column {position}")
                    self.profile["ATTACH"] += 1

class Agenda:

    def __init__(self) -> None:
        self._items: List[(Item, Item, float)] = []  # list of all items that were *ever* pushed
        self._next = 0  # index of first item that has not yet been popped
        self._index: Dict[Item, int] = {}  # stores index of an item if it has been pushed before

        # Note: There are other possible designs.  For example, self._index doesn't really
        # have to store the index; it could be changed from a dictionary to a set.
        #
        # However, we provided this design because there are multiple reasonable ways to extend
        # this design to store weights and backpointers.  That additional information could be
        # stored either in self._items or in self._index.

    def __len__(self) -> int:
        """Returns number of items that are still waiting to be popped.
        Enables `len(my_agenda)`."""
        return len(self._items) - self._next

    def __eq__(self, other):
        return;
    def __hash__(self):
        return;

    def getWeights(self, index):
        return self._items[index][2]

    def checkExist(self, item):
        if (item in self._index):
            return self._index.get(item)
        else:
            return 0;

    def getIndex(self, item):
        if (item in self._index):
            return self._index.get(item)
        else:
            return 0;

    def getItem(self, index):
       
        return self._items[index]
        

    def push(self, item, bp, weight) -> None:
        """Add (enqueue) the item, unless it was previously added."""
        if item not in self._index:  # O(1) lookup in hash table
            self._items.append((item, bp, weight))
            self._index[item] = len(self._items) - 1

    def reprocess(self, item, bp, weight):
        self._items[self.checkExist(item)] = ((item, bp, weight))

    def pop(self) -> (Item, Item, float):
        """Returns one of the items that was waiting to be popped (dequeued).
        Raises IndexError if there are no items waiting."""
        if len(self) == 0:
            raise IndexError
        item = self._items[self._next]
        self._next += 1
        return item

    def all(self) -> Iterable[(Item, Item, float)]:
        """Collection of all items that have ever been pushed, even if
        they've already been popped."""
        return self._items

    def __repr__(self):
        """Provide a REPResentation of the instance for printing."""
        next = self._next
        return f"{self.__class__.__name__}({self._items[:next]}; {self._items[next:]})"


class Grammar:
    """Represents a weighted context-free grammar."""

    def __init__(self, start_symbol: str, *files: Path) -> None:
        """Create a grammar with the given start symbol,
        adding rules from the specified files if any."""
        self.start_symbol = start_symbol
        self._expansions: Dict[str, List[Rule]] = {}  # maps each LHS to the list of rules that expand it
        # Read the input grammar files
        for file in files:
            self.add_rules_from_file(file)

    def add_rules_from_file(self, file: Path) -> None:
        """Add rules to this grammar from a file (one rule per line).
        Each rule is preceded by a normalized probability p,
        and we take -log2(p) to be the rule's weight."""
        with open(file, "r") as f:
            for line in f:
                # remove any comment from end of line, and any trailing whitespace
                line = line.split("#")[0].rstrip()
                # skip empty lines
                if line == "":
                    continue
                # Parse tab-delimited linfore of format <probability>\t<lhs>\t<rhs>
                _prob, lhs, _rhs = line.split("\t")
                prob = float(_prob)
                rhs = tuple(_rhs.split())
                rule = Rule(lhs=lhs, rhs=rhs, weight=-math.log2(prob))

                # rule.weight holds the weight for each rule
                # rule.weight

                if lhs not in self._expansions:
                    self._expansions[lhs] = []
                self._expansions[lhs].append(rule)

    def expansions(self, lhs: str) -> Iterable[Rule]:
        """Return an iterable collection of all rules with a given lhs"""
        return self._expansions[lhs]

    def is_nonterminal(self, symbol: str) -> bool:
        """Is symbol a nonterminal symbol?"""
        return symbol in self._expansions


# A dataclass is a class that provides some useful defaults for you. If you define
# the data that the class should hold, it will automatically make things like an
# initializer and an equality function.  This is just a shortcut.
# More info here: https://docs.python.org/3/library/dataclasses.html
# Using a dataclass here lets us specify that instances are "frozen" (immutable),
# and therefore can be hashed and used as keys in a dictionary.
@dataclass(frozen=True)
class Rule:
    """
    Convenient abstraction for a grammar rule.
    A rule has a left-hand side (lhs), a right-hand side (rhs), and a weight.
    """
    lhs: str
    rhs: Tuple[str, ...]
    weight: float = 0.0

    def __repr__(self) -> str:
        """Complete string used to show this rule instance at the command line"""
        return f"{self.lhs} → {' '.join(self.rhs)}, {self.weight}"


# We particularly want items to be immutable, since they will be hashed and
# used as keys in a dictionary (for duplicate detection).
@dataclass(frozen=True)
class Item:
    """An item in the Earley parse table, representing one or more subtrees
    that could yield a particular substring."""
    rule: Rule
    # ,weights: {}
    dot_position: int
    start_position: int

    # We don't store the end_position, which corresponds to the column
    # that the item is in, although you could store it redundantly for
    # debugging purposes if you wanted.

    def next_symbol(self) -> Optional[str]:
        """What's the next, unprocessed symbol (terminal, non-terminal, or None) in this partially matched rule?"""
        assert 0 <= self.dot_position <= len(self.rule.rhs)
        if self.dot_position == len(self.rule.rhs):
            return None
        else:
            return self.rule.rhs[self.dot_position]

    def with_dot_advanced(self) -> Item:
        # update item with new weights?
        if self.next_symbol() is None:
            raise IndexError("Can't advance the dot past the end of the rule")
        return Item(rule=self.rule, dot_position=self.dot_position + 1, start_position=self.start_position)

    def __repr__(self) -> str:
        """Complete string used to show this item at the command line"""
        DOT = "·"
        rhs = list(self.rule.rhs)  # Make a copy.
        rhs.insert(self.dot_position, DOT)
        # self.weights[(self.start_position, self.rule.lhs.join(rhs))] = self.rule.weight;
        dotted_rule = f"{self.rule.lhs} → {' '.join(rhs)}"
        return f"({self.start_position}, {dotted_rule})"  # matches notation on slides


def main():
    # Parse the command-line arguments
    args = parse_args()
    #logging.basicConfig(level=args.verbose)  # Set logging level appropriately
    logging.basicConfig(filename="all3.log", level=args.verbose, filemode='w')
    grammar = Grammar(args.start_symbol, args.grammar)
    import pdb;

    with open(args.sentences) as f:
        for sentence in f.readlines():
            sentence = sentence.strip()
            if sentence != "":  # skip blank lines
                # analyze the sentence
                log.debug("=" * 70)
                log.debug(f"Parsing sentence: {sentence}")
                chart = EarleyChart(sentence.split(), grammar, progress=args.progress)
                # print the result
                print(
                    "Sentence Max Probability:" + str(chart.returnMaxProbability())
                )
                log.debug(f"Profile of work done: {chart.profile}")
                print(chart.helper_print())
               


if __name__ == "__main__":
    import doctest

    doctest.testmod(verbose=False)  # run tests
    main()