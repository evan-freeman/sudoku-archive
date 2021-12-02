"""
Sudoku Solver v2.1 by Evan Freeman
This module contains various classes and functions used to solve a Sudoku puzzle. Currently only supports 9x9.
In general you will want to use the analyze function to solve your puzzle and output some interesting statistics.
Please format your puzzle as an 81 character string.
Conventions:
1) "region" denotes the set of either all columns, rows, or boxes.
2) "element" denotes one particular column, row, or box.
3) Two cells are said to "intersect" if they appear in the same column, row, or box.
Upcoming Features
1) More Strategies
2) More Statistics:
    - Contradiction Depth
    - Amount of backtracking for brute force
3) Generalize to nxn
4) Accept int as well as str for input
    - This is simple, just pad front with zeros if the input is too short
5) Error handling
"""

import itertools as it
import time
from typing import Union

import numpy as np


class Cell:
    """ Represents a particular cell of a Sudoku, along with all information about it."""

    def __init__(self, x: int, y: int, value: int = 0, n: int = 9):
        """
        self.x is the cell's x-coordinate
        self.y is the cell's y-coordinate
        self.value is the cell's number. 0 represents unknown.
        self.poss is the list of possible values for that cell.
        self.column is the column number of that cell, which is the same as the x-coordinate.
        self.row is the row number of that cell, which is the same as the y-coordinate.
        self.box is the box number of that cell. The box numbers are 0-8, going in 3x3 sections from left to right.
        The coordinate convention is positive x and negative y, beginning in the upper left corner, i.e. left then down.
        TODO Find a mathematical solution to identifying the box number of a given cell, not a handcrafted dict.
        """

        self.x = x
        self.y = y
        self.value = value
        if self.value == 0:
            self.poss = list(range(1, n + 1))
        else:
            self.poss = []
        self.column = self.x
        self.row = self.y
        box_dict = {(0, 0): 0, (0, 3): 1, (0, 6): 2,
                    (3, 0): 3, (3, 3): 4, (3, 6): 5,
                    (6, 0): 6, (6, 3): 7, (6, 6): 8}
        self.box = box_dict[self.x // 3 * 3, self.y // 3 * 3]


class Grid:
    """
    Represents a Sudoku puzzle, composed of 81 Cell object, along with all information about it and all functions
        needed to solve it.
    """

    # =============================================================================
    # init Helper Functions
    # =============================================================================

    @staticmethod
    def int_except(x: Union[int, str]) -> int:
        """
        Returns x as an int if possible, else returns 0. Used for parsing the input puzzle.
        If x can't be made into an int, like '.', then it represents a blank.
        Args:
            x (object): One element of the input, a string or int.
        Returns:
            int: The value of that cell as an int, or 0, which represents unknown.
        """

        try:
            return int(x)
        except:
            return 0

    def generate_region_list(self, region: str) -> list:
        """
        Generates a list of elements containing the Cell objects that belong to a given region.
        Args:
            region: A string that denotes the desired region, either 'columns', 'rows', or 'boxes'.
        Returns:
            list: This is a list of 9 elements, each of which is a list containing
                the 9 cells that belong to that particular element.
        """

        return [[cell for cell in self.cells.values() if getattr(cell, region) == i]
                for i in range(self.n)]

    def generate_blank_region_list(self, region: str) -> list:
        """
        Generates a list of elements containing the BLANK Cell objects that belong to a given region.
        Args:
            region: A string that denotes the desired region, either 'columns', 'rows', or 'boxes'.
        Returns:
            list: This is a list of 9 elements, each of which is a list containing the BLANK cells that
                belong to a particular region. The lists may be empty.
        """

        return [[cell for cell in self.cells.values() if getattr(cell, region) == i and
                 cell.value == 0] for i in range(self.n)]

    # =============================================================================
    # init and properties
    # =============================================================================

    def __init__(self, puzzle: str, description: str = 'N/A'):
        """
        self.description is an optional description of the puzzle.
        self.list is a list of the 81 elements in the original puzzle, used for generating other attributes.
        self.input is the standardized input puzzle, with all blanks replaced by 0
        self.length is the length of the puzzle, 81 for 9x9 puzzles.
        self.n is the dimension of the puzzle, 9 for 9x9 puzzles.
        self.arr is a nxn numpy array of the puzzle. This is used to find the coordinates of each cell.
        self.cells is a dict of the Cell objects in the puzzle, keyed by their coordinate for fast lookup.
            Note that we must reverse the order of the coordinates in the array, because it uses matrix coordinates.
        self.columns is a list of 9 columns, each a list containing 9 Cell objects.
        self.rows is a list of 9 rows, each a list containing 9 Cell objects.
        self.boxes is a list of 9 rows, each a list containing 9 Cell objects.
        self.blanks is a list of all the blank Cell objects, which have a 0 value.
        self.column_blanks is a list of 9 columns, each a list containing only the blank Cell objects in that element.
            The columns may be blank.
        self.row_blanks is a list of 9 rows, each a list containing only the blank Cell objects in that element.
            The rows may be blank.
        self.box_blanks = is a list of 9 rows, each a list containing only the blank Cell objects in that element.
            The rows may be blank.
        self.strategy_counts is a dict of the number of successful applications of various strategies.
            'ns' is naked single
            'hs' is hidden single
            'nd' is naked double
            'hd' is hidden double
            'nt' is naked triple
            'ht' is hidden triple
            'nq' is naked quad
            'hq' is hidden quad
            'r' is reduction, both pointing pair and box line
        self.total_strategy_counts is the sum of all values in self.strategy_counts.
        self.output is an 81 char string of the values in each cell object.
        self.output_grid is a list of 9 strings, representing the 9 rows, in the output.
        Args:
            puzzle (str): 81 char string representing the puzzle
            description (str): optional description of the puzzle, e.g. 'Naked Quad Example'
        """

        self.description = description
        self.list = [self.int_except(x) for x in puzzle]
        self.input = ''.join(str(x) for x in self.list)
        self.length = len(puzzle)
        self.n = int(self.length ** .5)
        self.arr = np.array(self.list).reshape((self.n, self.n))
        # Note that we must reverse the order of the coordinates in the array, because it uses matrix coordinates
        self.cells = {(x, y): Cell(x, y, self.arr[y, x], self.n) for y in range(self.n) for x in range(self.n)}
        self.columns = self.generate_region_list('column')
        self.rows = self.generate_region_list('row')
        self.boxes = self.generate_region_list('box')
        self.blanks = [cell for cell in self.cells.values() if cell.value == 0]
        self.column_blanks = self.generate_blank_region_list('column')
        self.row_blanks = self.generate_blank_region_list('row')
        self.box_blanks = self.generate_blank_region_list('box')
        self.strategy_counts = {
            'ns': 0,
            'hs': 0,
            'nd': 0,
            'hd': 0,
            'nt': 0,
            'ht': 0,
            'nq': 0,
            'hq': 0,
            'r': 0
        }

    @property
    def total_strategy_count(self):
        return sum(count for count in self.strategy_counts.values())

    @property
    def output(self):
        output = [self.cells[(i, j)].value for j in range(self.n) for i in range(self.n)]
        return ''.join(str(x) for x in output)

    @property
    def output_grid(self):
        return [self.output[i:i + self.n] for i in range(0, self.length, self.n)]

    # =============================================================================
    # DISPLAY FUNCTIONS
    # =============================================================================

    def display_grid(self):
        """
        Displays the puzzle as a 9x9 grid of strings.
        """

        for row in self.output_grid:
            print(row)

    # =============================================================================
    # STRATEGY HELPER FUNCTIONS
    # =============================================================================

    @staticmethod
    def intersect(cell1: object, cell2: object) -> bool:
        """
        Returns a boolean for whether two cells intersect. Note that every cell intersects with itself, trivially.
        Args:
            cell1: a cell
            cell2: a cell
        Returns:
            bool: The truth value of "cell1 intersects cell2"
        """

        return cell1.column == cell2.column or cell1.row == cell2.row or cell1.box == cell2.box

    def intersecting_values(self, cell: object) -> list:
        """ 
        Returns a list of values that intersect a given cell. The list is cleared of duplicates.
        Args:
            cell: a cell
        Returns:
            list: list of intersecting values
        """

        return list(set([other_cell.value for other_cell in self.cells.values() if self.intersect(cell, other_cell)]))

    def update_poss(self):
        """ Updates all lists of possibilities for blank cells with new information in the Sudoku."""

        for cell in self.blanks:
            for poss in cell.poss[:]:  # Here we iterate over a copy, so we don't have issues with modifying inplace.
                if poss in self.intersecting_values(cell):
                    cell.poss.remove(poss)

    @staticmethod
    def same_region(cell1: object, cell2: object, region: str) -> bool:
        """ Returns whether two cells are in the same element of the given region.
        Args:
            cell1 (object): a cell
            cell2 (object): a cell
            region (str): 'column', 'row', or 'box'
        Returns:
            bool: whether two cells are in the same element of the given region
        """
        return getattr(cell1, region) == getattr(cell2, region)

    def intersecting_blank_cells(self, cell: object, region: str) -> list:
        """
        Returns a list of blank cells which intersect the given cell, in the given region, OTHER THAN the given cell.
        Args:
            cell (object): a cell
            region (str): 'column', 'row', or 'box'
        Returns:
            list: list of blank cells which intersect the given cell, in the given region, OTHER THAN the given cell
        """

        return [other_cell for other_cell in self.blanks if
                other_cell != cell and self.same_region(cell, other_cell, region)]

    def generate_other_blanks(self, cell: object) -> list:
        """
        Returns blanks in OTHER cells in the same column, row, and box, as a list of lists.
        Args:
            cell: a cell
        Returns:
            list: list of 3 lists - other blanks in the given cell's column, row, and box
        """

        return [self.intersecting_blank_cells(cell, region) for region in ('column', 'row', 'box')]

    @staticmethod
    def check_no_dupes(list_of_cells: list) -> bool:
        """
        Returns true if a given list of cells has no duplicate values, else false.
        Args:
            list_of_cells: a list of Cell objects
        Returns:
            bool: True if the list had no dupes, else false.
        """

        # First, remove any 0s, which are placeholders for unknowns
        clean_list = [cell.value for cell in list_of_cells if cell.value != 0]

        # Now check for duplicates. Return True if there were no duplicates, else False.
        return len(clean_list) == len(set(clean_list))

    def check_consistency(self, cell: object) -> bool:
        """
        Checks whether a cell is a member of a contradictory region. i.e there is a repeated value.
        Args:
            cell: a cell
        Returns:
            object: True if there are no intersecting duplicate values, else False.
        """

        return (
                self.check_no_dupes(self.columns[cell.column]) and
                self.check_no_dupes(self.rows[cell.row]) and
                self.check_no_dupes(self.boxes[cell.box])
        )

    @staticmethod
    def extract_possibilities(list_of_cells: list) -> set:
        """ Returns the set of all possibilities in a given list of cells.
        Args:
            list_of_cells: list of cell objects
        Returns:
            set: set of all possibilities in a given list of cells
        """

        return set(poss for cell in list_of_cells for poss in cell.poss)

    def check_for_naked_set(self, element: list, n: int) -> tuple:
        """
        Returns a tuple of a naked set of size n in a given element, if one exists, along with the naked cells.
            If not, returns a tuple of an empty tuple and empty list, which each evaluate to False in Python.
        Args:
            element (list): an element of blanks
            n (int): the size of the desired naked set to check for (i.e. 2->naked pair, 3->naked triple, etc.)
        Returns:
            tuple: Tuple of tuple and list, representing the naked set and the naked cells.
            """

        all_poss = self.extract_possibilities(element)
        for potential_naked_set in it.combinations(all_poss, n):
            potential_naked_cells = [cell for cell in element if set(cell.poss).issubset(set(potential_naked_set))]
            if len(potential_naked_cells) == n:
                return potential_naked_set, potential_naked_cells
        return (), []

    def remove_from_other_cells(self, unchanged_cells: list, set_to_remove: set, element: list, n: int) -> bool:
        """
        This function removes a set of numbers from the possibilities of cells in a given element,
            other than a list of unchanged cells. It also counts any progress that is made for the strategy used,
            which will be one of the naked strategies or reduction.
        Note: The list of cell possibilities is modified inplace as it is iterated over, but that is not an issue
            because if any changes are made the function will immediately return and not keep iterating. This also
            has the benefit of using each strategy the minimal amount possible. This may decrease solve speed, but it
            more clearly delineates which strategies are used, and how much.
        Args:
            unchanged_cells (list): list of cells in element that will not be altered
            set_to_remove (tuple): tuple of number to remove from possibilities in element
            element (list): list of blank cells whose possibilities will be altered
            n (int): size of naked strategy being used. reduction is labeled 5, though it is not exactly a naked strat
        Returns:
            bool: True if a number was removed from a cell's possibilities, else False.
        """
        naked_count_dict = {2: 'nd', 3: 'nt', 4: 'nq', 5: 'r'}

        to_check = [cell for cell in element if cell not in unchanged_cells]
        for cell in to_check:
            for i, poss in enumerate(cell.poss):
                if poss in set_to_remove:
                    del cell.poss[i]
                    self.strategy_counts[naked_count_dict[n]] += 1
                    return True
        return False

    def check_for_hidden_set(self, element: list, n: int) -> tuple:
        """
        Returns a tuple of a hidden set of size n in a given element, if one exists, along with the hidden cells.
            If not, returns a tuple of an empty tuple and empty list, which each evaluate to False in Python.
        Args:
            element (list): an element of blanks
            n (int): the size of the desired hidden set to check for (i.e. 2->hidden pair, 3->hidden triple, etc.)
        Returns:
            tuple: Tuple of tuple and list, representing the hidden cells and the hidden set.
        """
        all_poss = self.extract_possibilities(element)
        for potential_hidden_set in it.combinations(all_poss, n):
            potential_hidden_cells = [cell for cell in element if
                                      any([poss in cell.poss for poss in potential_hidden_set])]
            if len(potential_hidden_cells) == n:
                return potential_hidden_set, potential_hidden_cells
        return (), []

    def reduce_cells(self, hidden_cells: list, hidden_set: tuple, n: int) -> bool:
        """ This function reduces a set of cells to only the possibilities they contain from a given set of numbers.
        Note: As in remove_from_other_cells, the list of cell possibilities is modified inplace as it is iterated over,
            but that is not an issue because if any changes are made the function will immediately return and not keep
            iterating.
        Args:
            hidden_cells (list): list of cells which can be reduced
            hidden_set (tuple): tuple of number to use for reducing cells
            n (int): size of hidden strategy being used
        Returns:
            bool: True if a number was removed from a cell's possibilities, else False.
        """

        hidden_count_dict = {2: 'hd', 3: 'ht', 4: 'hq'}

        for cell in hidden_cells:
            for poss in cell.poss:
                if poss not in hidden_set:
                    cell.poss.remove(poss)
                    self.strategy_counts[hidden_count_dict[n]] += 1
                    return True
        return False

    @staticmethod
    def extract_region_numbers(cells: list, region: str) -> set:
        """
        Extracts the region numbers that appear among a list of cells.
        Args:
            cells (list): list of cells
            region (str): 'column', 'row', or 'box'
        Returns:
            set: the numbers that were extracted
        """

        return set(getattr(cell, region) for cell in cells)

    def remove_from_blank_lists_and_clear_possibilities(self, cell: object):
        """
        Takes in a cell who's value has just been set because of naked or hidden single and clears the possibilities
        of that cell. Then removes it from the following lists:
        self.blanks
        the element of self.column_blanks that contains the cell
        the element of self.row_blanks that contains the cell
        the element of self.box_blanks that contains the cell
        Note: The ONLY possible way for a number's value to be set is by naked or hidden single.
        Args:
            cell: Cell object that has just had it's value set and needs to be cleaned up.
        """

        cell.poss = []
        self.blanks.remove(cell)
        self.column_blanks[cell.column].remove(cell)
        self.row_blanks[cell.row].remove(cell)
        self.box_blanks[cell.box].remove(cell)

    def check_for_intersection(self, element: list, region: str) -> tuple:
        """
        Checks whether there is a possible number in a given element such that all occurrences of that number
        intersect another element in the given region.
        FOR EXAMPLE - If all 2s in a box occurred in the same column, then 2 could be removed from everywhere else in
            that column.
        Args:
            element (list): list of blank cells to check for intersection
            region (str): 'column', 'row', or 'box'. Indicates the region of the input element so the function
                only looks for intersection in other regions.
        Returns:
            tuple: three lists -    cells to remain unchanged,
                                    singleton possibility that defines intersection,
                                    element that intersects
        """

        other_regions = [reg for reg in ('column', 'row', 'box') if reg != region]
        intersecting_region_blanks = {'column': 'column_blanks', 'row': 'row_blanks', 'box': 'box_blanks'}

        for poss in range(1, 10):
            cells_with_poss = [cell for cell in element if poss in cell.poss]
            region_nums = {reg: self.extract_region_numbers(cells_with_poss, reg) for
                           reg in other_regions}
            for reg in region_nums:
                if len(region_nums[reg]) == 1:
                    return cells_with_poss, [poss], getattr(self,
                                                            intersecting_region_blanks[reg])[list(region_nums[reg])[0]]
        return [], [], []

    # =============================================================================
    # STRATEGY FUNCTIONS
    # =============================================================================

    def naked_single(self) -> bool:
        """
        Fill in a blank cell if there is only a single possibility in that cell.
        Returns:
            bool: True if a blank cell was filled in, else False .
        """
        self.update_poss()

        for cell in self.blanks:
            if len(cell.poss) == 1:
                cell.value = cell.poss[0]
                self.remove_from_blank_lists_and_clear_possibilities(cell)
                self.strategy_counts['ns'] += 1
                return True
        return False

    def hidden_single(self) -> bool:
        """
        Fill in a blank cell when there is only one remaining place for a number in an element.
        TODO rewrite this strategy to iterate over each number 1-9 instead of each cell?
            Possible speed / readability improvements
        Returns:
            bool: True if a blank cell was filled in, else False.
        """

        self.update_poss()

        for cell in self.blanks:
            # Generate the subset of blanks that are in the same column, row, or box as our current blank
            blank_column, blank_row, blank_box = self.generate_other_blanks(cell)

            # Extract the possible numbers from the intersecting regions
            other_column_poss = self.extract_possibilities(blank_column)
            other_row_poss = self.extract_possibilities(blank_row)
            other_box_poss = self.extract_possibilities(blank_box)

            # If one of the possibilities is the only occurrence in a given region, fill it in
            for poss in cell.poss:
                if poss not in other_column_poss or poss not in other_row_poss or poss not in other_box_poss:
                    cell.value = poss
                    self.remove_from_blank_lists_and_clear_possibilities(cell)
                    self.strategy_counts['hs'] += 1
                    return True
        return False

    def general_naked(self, n: int) -> bool:
        """ Performs the logic of naked double, or triple, or quad for a given region.
        Strategy explanation - If there is an element which contains n cells which are each a subset of a set of
            n possibilities, then those numbers may be removed from every other cell in that element.
        Args:
            n (int): Size of naked set to search for (either 2, 3, or 4).
        Returns:
            bool: True if a possibility was removed, else False
        """

        for region in ('column_blanks', 'row_blanks', 'box_blanks'):
            for element in getattr(self, region):
                naked_set, naked_cells = self.check_for_naked_set(element, n)
                if naked_cells:
                    if self.remove_from_other_cells(naked_cells, naked_set, element, n):
                        return True
        return False

    def general_hidden(self, n: int) -> bool:
        """
        This will execute a hidden pair, triple, or quad for a given region.
        Strategy explanation - If there is an element and a set of n possibilities which only appear among n cells in
            that element, then no other possibilities are allowed among those cells.
        Args:
            n (int): Size of hidden set to search for (either 2, 3, or 4)
        Returns:
            bool: True if a possibility was removed, else False
        """

        for region in ('column_blanks', 'row_blanks', 'box_blanks'):
            for element in getattr(self, region):
                hidden_set, hidden_cells = self.check_for_hidden_set(element, n)
                if hidden_cells:
                    if self.reduce_cells(hidden_cells, hidden_set, n):
                        return True
        return False

    def naked_double(self):
        """ See general_naked"""
        return self.general_naked(2)

    def naked_triple(self):
        """ See general_naked"""
        return self.general_naked(3)

    def naked_quad(self):
        """ See general_naked"""
        return self.general_naked(4)

    def hidden_double(self):
        """ See general_hidden"""
        return self.general_hidden(2)

    def hidden_triple(self):
        """ See general_hidden"""
        return self.general_hidden(3)

    def hidden_quad(self):
        """ See general_hidden"""
        return self.general_hidden(4)

    def reduction(self):
        """
        Performs both pointing pair and box line reduction.
        Strategy Explanation - If there is a possibility such that all occurrences of that possibility in element1
            intersect with element2, where elements 1 and 2 are of different types, then that possibility may be
            removed from all non-intersecting cells of element2.
        Returns:
            bool: True if a possibility was removed, else False.
        """

        for region_blanks, region in (('column_blanks', 'column'), ('row_blanks', 'row'), ('box_blanks', 'box')):
            for element in getattr(self, region_blanks):
                intersecting_cells, intersecting_set, intersecting_element = (
                    self.check_for_intersection(element, region))
                if intersecting_cells:
                    if self.remove_from_other_cells(intersecting_cells, intersecting_set, intersecting_element, 5):
                        return True
        return False


# =============================================================================
# SOLVERS
# =============================================================================


class Solver:
    """ This is a class of general attributes and methods for my 3 Sudoku Solvers."""

    def __init__(self, puzzle: str, description: str = 'N/A'):
        """
        self.sudoku is the Grid object that represents the given puzzle
        self.count is the count of how many loops of brute force, or limited brute force, have been executed
        self.type is the type of solver, either 'BruteForce', 'LimitedBruteForce', or 'StrategySolve
        self.start_time is the time at which the solver began solving
        self.total_time is the time at which the solver finished solving
        self.sudoku.description is an optional description of the sudoku
        Args:
            puzzle (str): 81 char string representing the puzzle
            description (str): optional description of the puzzle, e.g. 'Naked Quad Example'
        """

        self.sudoku = Grid(puzzle)
        self.count = 0
        self.type = 'None'
        self.start_time = 0
        self.total_time = 0
        self.sudoku.description = description

    def begin_timing(self):
        """ Begin timing the solver"""
        self.start_time = time.time()

    def end_timing(self):
        """ Finish timing the solver"""
        self.total_time = time.time() - self.start_time

    def general_brute_force(self, use_poss: bool = True):
        """
        This is the general brute force function, which will iterate back and forth through the blanks
            in the puzzle until it finds a non-contradictory set of values.
        Args:
            use_poss (bool): decides whether we will brute force over 1-9 in each blank, or just over the
            possibilities in that cell. This will be set to False for BruteForce, but True for LimitedBruteForce and
            StrategySolve.
        """

        # If we're going to use information about possibilities, let's first update that information.
        if use_poss:
            self.sudoku.update_poss()

        # i is our index, which will keep track of our position as we step back and forth through the list of blanks
        i = 0
        while i != len(self.sudoku.blanks):
            self.count += 1
            blank = self.sudoku.blanks[i]

            # Scenario 1: The blank's value is 0. That means we should try the first possibility.
            if blank.value == 0:
                blank.value = blank.poss[0]

            # Scenario 2: The blank's value is the final possibility. So we've already tried all the possibilities.
            # So we need to clear it out and step back. Also we skip the rest of the loop,
            # because we don't need to check for consistency. In fact, it would be bad to check for consistency,
            # as we are guaranteed to trivially be consistent. This would lead to stepping forward,
            # canceling out our step back, and ending up in an infinite loop.
            elif blank.value == blank.poss[-1]:
                blank.value = 0
                i -= 1
                continue

            # Scenario 3: The blank's value is some other non-last possibility. So we step forward by one.
            else:
                blank.value = blank.poss[blank.poss.index(blank.value) + 1]

            # Now we check for consistency. If consistent, step forward. Else run through this same spot again.
            if self.sudoku.check_consistency(blank):
                i += 1


class BruteForce(Solver):
    """
    Here is a class of solver that uses full brute force. So we don't even limit our candidates for brute force,
    we just go numerically through them all.
    """

    def __init__(self, puzzle, description='N/A'):
        """ See Solver.__init__"""
        super().__init__(puzzle, description)
        self.type = 'BruteForce'

    def solve(self):
        """ See Solver.general_brute_force"""
        self.begin_timing()
        self.general_brute_force(False)
        self.end_timing()


class LimitedBruteForce(Solver):
    """ This class uses the same logic as brute force, but first reduces the possible candidates to brute force over."""

    def __init__(self, puzzle, description='N/A'):
        """ See Solver.__init__"""
        super().__init__(puzzle, description)
        self.type = 'LimitedBruteForce'

    def solve(self):
        """ See Solver.general_brute_force"""
        self.begin_timing()
        self.general_brute_force()
        self.end_timing()


class StrategySolve(Solver):
    """
    This solver applies the 9 strategies within the Grid object in order to solve the puzzle,
        then finishes with limited brute force if necessary.
    """

    def __init__(self, puzzle, description='N/A'):
        """ See Solver.__init__"""
        super().__init__(puzzle, description)
        self.type = 'StrategySolve'

    def solve(self):
        """
        Iterates through each of the 9 strategies in the Grid class. If any progress is made by any strategy,
            then it goes back to the beginning of the strategy list. If all strategies fail in a given loop,
            then it finished with limited brute force.
        Note: each strategy stops as soon as ANY progress is made with that strategy. This ensures that the minimal
            amount of advanced strategies are used.
        """
        self.begin_timing()

        strategy_list = [
            self.sudoku.naked_single,
            self.sudoku.hidden_single,
            self.sudoku.naked_double,
            self.sudoku.hidden_double,
            self.sudoku.naked_triple,
            self.sudoku.hidden_triple,
            self.sudoku.naked_quad,
            self.sudoku.hidden_quad,
            self.sudoku.reduction
        ]

        progress = True
        while progress:
            for strategy in strategy_list:
                progress = strategy()
                if progress:
                    break
        self.general_brute_force()
        self.end_timing()


def analyse(puzzle: str, description: str = 'N/A') -> dict:
    """
    Solves a given Sudoku with all 3 solvers: BruteForce, LimitedBruteForce, and StrategySolve. Then returns a dict of
        statistics about the Sudoku and the solvers.
    Args:
        puzzle (str): 81 char string representing the puzzle
        description (str): optional description of the puzzle, e.g. 'Naked Quad Example'
    Returns:
        dict:   'description': optional description of the Sudoku
                'input': the input Sudoku string
                'output': the solved Sudoku as a string
                'bf_time': time for BruteForce to solve
                'lbf_time': time for LimitedBruteForce to solve
                'strat_time': time for StrategySolve to solve
                'bf_loops': number of brute force loops to solve
                'lbf_loops': number of limited brute force loops to solve
                'strat_lbf_loops': number of limited brute force loops to solve AFTER applying all strategies
                'ns_count': number of successful applications of naked singles
                'hs_count': number of successful applications of hidden singles
                'nd_count': number of successful applications of naked doubles
                'hd_count': number of successful applications of hidden doubles
                'nt_count': number of successful applications of naked triples
                'ht_count': number of successful applications of hidden triples
                'nq_count': number of successful applications of naked quads
                'hq_count': number of successful applications of hidden quads
                'r_count': number of successful applications of reduction
    """

    bf = BruteForce(puzzle, description)
    bf.solve()

    lbf = LimitedBruteForce(puzzle, description)
    lbf.solve()

    strat = StrategySolve(puzzle, description)
    strat.solve()

    return {
        'description': strat.sudoku.description,
        'input': strat.sudoku.input,
        'output': strat.sudoku.output,
        'bf_time': bf.total_time,
        'lbf_time': lbf.total_time,
        'strat_time': strat.total_time,
        'bf_loops': bf.count,
        'lbf_loops': lbf.count,
        'strat_lbf_loops': strat.count,
        'ns_count': strat.sudoku.strategy_counts['ns'],
        'hs_count': strat.sudoku.strategy_counts['hs'],
        'nd_count': strat.sudoku.strategy_counts['nd'],
        'hd_count': strat.sudoku.strategy_counts['hd'],
        'nt_count': strat.sudoku.strategy_counts['nt'],
        'ht_count': strat.sudoku.strategy_counts['ht'],
        'nq_count': strat.sudoku.strategy_counts['nq'],
        'hq_count': strat.sudoku.strategy_counts['hq'],
        'r_count': strat.sudoku.strategy_counts['r']
    }


# =============================================================================
# SAMPLE SUDOKU
# =============================================================================


if __name__ == '__main__':
    example_puzzle = '.94...13..............76..2.8..1.....32.........2...6.....5.4.......8..7..63.4..8'
    example_description = 'Fast Example'
    solution = '794582136268931745315476982689715324432869571157243869821657493943128657576394218'
    analyze_example = analyse(example_puzzle, example_description)
    for key in analyze_example:
        print(f'{key}: {analyze_example[key]}')
    success = analyze_example['output'] == solution
    print(f'It is {success} that we got the right answer.')
    if success:
        print("Aren't we smart?")

# =============================================================================
# POSTSCRIPT
# =============================================================================

# For Megan, who will always be better than me at Sudoku.
