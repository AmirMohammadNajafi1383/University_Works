import tkinter
import numpy as np
from tkinter import filedialog
import json
import copy
import math

# code is forked from : https://github.com/kubicodes/Backtracking-Sudoku-Solver-with-Python.git


class GUI:
    def find_first_empty_position(self , board):
        for i in range(9):
            for j in range(9):
                if board[i][j] == 0:
                    return (i, j)
        return None

    def is_valid(self , board):
        # check row
        for row in board:
            for number in range(1, 10):
                if row.count(number) > 1:
                    return False

        # check column
        for col_index in range(9):
            col = []
            for row in board:
                col.append(row[col_index])
            for number in range(1,10):
                if col.count(number) > 1:
                    return False

        # check cube
        positions = [
            [0, 0],
            [0, 3],
            [0, 6],
            [3, 0],
            [3, 3],
            [3, 6],
            [6, 0],
            [6, 3],
            [6, 6],
        ]

        for number in range(1, 10):
            for position in positions:
                cube_x = position[1] // 3
                cube_y = position[0] // 3
                cube_cells = []
                for i in range(cube_y * 3, cube_y * 3 + 3):
                    for j in range(cube_x * 3, cube_x * 3 + 3):
                        cube_cells.append(board[i][j])
                if cube_cells.count(number) > 1:
                    return False

        return True

    def can_be_put(self , board, position, number):
        # in that row and col and cube that position is located in there is not any
        # other cell with number= that number we want to put in this position
        t = copy.deepcopy(board)
        t[position[0]][position[1]] = number
        return self.is_valid(t)

    def validate_board(self, board):
        # returns true if board is solveable
        if not self.is_valid(board):
            return False
        clone = copy.deepcopy(board)
        return self.solve_board(clone)

    def calc_empty_percent(self , board):
        cells = []
        for row in range(9):
            for col in range(9):
                cells.append(board[row][col])
        return round((cells.count(0) / 81) * 100)

    def solve_board(self , board):
        empty_position = self.find_first_empty_position(board)

        # base case
        if not empty_position:
            return True
        else:
            row, col = empty_position

        for i in range(1, 10):
            if self. can_be_put(board, (row, col), i):
                board[row][col] = i
                # backtrack and try again
                if self.solve_board(board):
                    return True
                board[row][col] = 0

        return False

    def gen_random_board(self , number_of_empty_cells=30):
        board = [[0 for j in range(9)] for i in range(9)]
        board[np.random.randint(9)][np.random.randint(9)] = np.random.randint(1, 10)
        self.solve_board(board)

        cells_to_empty = []

        def gen_random_cell():
            return [np.random.randint(9), np.random.randint(9)]

        # add `number_of_empty_cells` unique cells to cells_to_empty
        for i in range(number_of_empty_cells):
            while True:
                t = gen_random_cell()
                if not t in cells_to_empty:
                    cells_to_empty.append(t)
                    break
        for cell in cells_to_empty:
            board[cell[0]][cell[1]] = 0

        return board

    @property
    def labels_data(self):
        t = [[] for i in range(9)]
        for i in range(9):
            for j in range(9):
                t[i].append(
                    int(float(self.entry_controlled_variables[i * 9 + j].get() or 0))
                )
        return t

    @labels_data.setter
    def labels_data(self, new_value):
        for row in range(9):
            for col in range(9):
                self.entry_controlled_variables[row * 9 + col].set(
                    str(new_value[row][col])
                )

    def __init__(self):
        self.window = tkinter.Tk()
        self.window.geometry("500x500")
        self.entry_controlled_variables = [
            tkinter.StringVar(self.window, 0) for i in range(81)
        ]
        # creating label components
        self.labels = []
        for i in range(81):
            l = tkinter.Entry(
                self.window, textvariable=self.entry_controlled_variables[i], width=3
            )
            cor = self.index_to_cordinates(i)

            l.grid(row=cor[0], column=cor[1])
            self.labels.append(l)

        # create difficulty radio buttons
        self.radio_buttons_variable = tkinter.IntVar(self.window, 30)

        t = tkinter.Radiobutton(
            self.window, text="easy", value=30, variable=self.radio_buttons_variable
        )
        t.grid(row=10, column=1)
        t = tkinter.Radiobutton(
            self.window, text="medium", value=40, variable=self.radio_buttons_variable
        )
        t.grid(row=10, column=2)
        t = tkinter.Radiobutton(
            self.window, text="hard", value=50, variable=self.radio_buttons_variable
        )
        t.grid(row=10, column=3)

        # create control menu
        t = tkinter.Button(self.window, text="create", command=self.create)
        t.grid(row=11, column=1)
        t = tkinter.Button(self.window, text="load", command=self.load)
        t.grid(row=11, column=2)
        t = tkinter.Button(self.window, text="solve", command=self.solve)
        t.grid(row=11, column=3)
        t = tkinter.Button(self.window, text="check", command=self.check)
        t.grid(row=11, column=4)

        self.window.mainloop()

    def load(self):
        filename = filedialog.askopenfilename(title="select a file")
        new_labels_data = []
        for line in open(filename, "r"):
            new_labels_data.append([int(cell) for cell in line.split(",")])
        self.labels_data = new_labels_data

    def create(self):
        t = self. gen_random_board(self.radio_buttons_variable.get())
        self.labels_data = t

    def solve(self):
        clone = copy.deepcopy(self.labels_data)
        t = self.solve_board(clone)

        if t == True:
            self.labels_data = clone
        else:
            tkinter.messagebox.showerror(
                title="board validation result", message="this board is not solvable "
            )

    def check(self):
        if self . validate_board(self.labels_data) == True:
            tkinter.messagebox.showinfo(
                title="board validation result", message="board is valid"
            )
        else:
            tkinter.messagebox.showerror(
                title="board validation result", message="board is invalid"
            )

    def index_to_cordinates(self, index):
        return [index // 9, index % 9]


GUI()
