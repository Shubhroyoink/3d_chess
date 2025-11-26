import os
models ={}
import sys
from enum import Enum, auto
import copy
from net import NetMode, start_host, start_client, send_move



from OpenGL.GL import *
from OpenGL.GLU import *
from OpenGL.GLUT import *

from OpenGL.GLUT import GLUT_BITMAP_TIMES_ROMAN_24, glutBitmapCharacter


# ============================================================
# === STUB / PLACEHOLDER GAME & MODEL IMPLEMENTATION ========
# ============================================================

class PieceColor(Enum):
    WHITE = auto()
    BLACK = auto()

class PieceType(Enum):
    PAWN = auto()
    ROOK = auto()
    KNIGHT = auto()
    BISHOP = auto()
    QUEEN = auto()
    KING = auto()

class MoveType(Enum):
    NORMAL = auto()
    CAPTURE = auto()
    EN_PASSANT = auto()
    CASTLING = auto()
# Networking globals
net_mode = NetMode.LOCAL
player_color = PieceColor.WHITE  # will be WHITE or BLACK depending on mode
sock = None
recv_thread = None
incoming_moves = []  # list used as a simple queue from net thread
host_ip = "127.0.0.1"
NET_PORT = 5000
class Move:
    def __init__(self, dst_row, dst_col, move_type=MoveType.NORMAL):
        self._dst = (dst_row, dst_col)
        self._type = move_type

    def getDestinationPosition(self):
        return self._dst

    def getType(self):
        return self._type

class BoardStub:
    # used only for promotion row info
    MIN_ROW_INDEX = 1
    MAX_ROW_INDEX = 8

class Piece:
    def __init__(self, piece_type, color):
        self._type = piece_type
        self._color = color
        self.has_moved = False

    def getType(self):
        return self._type

    def setType(self, t):
        self._type = t

    def getColor(self):
        return self._color

class Game:
    """
    Proper chess engine:
    - standard initial position
    - legal move generation
    - en-passant
    - castling
    - check / checkmate detection
    - promotion (type set via promote())
    """

    def __init__(self):
        self.board = BoardStub()
        # 1-based indexing: 1..8, we ignore index 0
        self._grid = [[None for _ in range(9)] for _ in range(9)]
        self._turn = PieceColor.WHITE
        self.en_passant_target = None  # (row, col) or None
        self._setup_start_position()

    # -------------- basic helpers -----------------

    def inside(self, r, c):
        return 1 <= r <= 8 and 1 <= c <= 8

    def getPiece(self, row, col):
        return self._grid[row][col]

    def isSquareOccupied(self, row, col):
        return self._grid[row][col] is not None

    def getPieceColor(self, row, col):
        p = self._grid[row][col]
        return p.getColor() if p else None

    def getTurnColor(self):
        return self._turn

    def getBoard(self):
        return self.board

    

    # -------------- initial position --------------

    def _setup_start_position(self):
        # pawns
        for c in range(1, 9):
            self._grid[2][c] = Piece(PieceType.PAWN, PieceColor.WHITE)
            self._grid[7][c] = Piece(PieceType.PAWN, PieceColor.BLACK)

        # back rank: R N B Q K B N R
        back_rank = [
            PieceType.ROOK,
            PieceType.KNIGHT,
            PieceType.BISHOP,
            PieceType.QUEEN,
            PieceType.KING,
            PieceType.BISHOP,
            PieceType.KNIGHT,
            PieceType.ROOK,
        ]

        # white pieces row 1
        for c, ptype in enumerate(back_rank, start=1):
            self._grid[1][c] = Piece(ptype, PieceColor.WHITE)

        # black pieces row 8
        for c, ptype in enumerate(back_rank, start=1):
            self._grid[8][c] = Piece(ptype, PieceColor.BLACK)

    # -------------- attack / check helpers --------

    def _find_king(self, color):
        for r in range(1, 9):
            for c in range(1, 9):
                p = self._grid[r][c]
                if p and p.getColor() == color and p.getType() == PieceType.KING:
                    return r, c
        return None

    def _is_square_attacked(self, row, col, by_color):
        # pawns
        dir_ = 1 if by_color == PieceColor.WHITE else -1
        for dc in (-1, 1):
            r = row - dir_
            c = col - dc
            if self.inside(r, c):
                p = self._grid[r][c]
                if p and p.getColor() == by_color and p.getType() == PieceType.PAWN:
                    return True

        # knights
        knight_offsets = [(2, 1), (1, 2), (-1, 2), (-2, 1),
                          (-2, -1), (-1, -2), (1, -2), (2, -1)]
        for dr, dc in knight_offsets:
            r, c = row + dr, col + dc
            if self.inside(r, c):
                p = self._grid[r][c]
                if p and p.getColor() == by_color and p.getType() == PieceType.KNIGHT:
                    return True

        # bishops / rooks / queens (sliding)
        directions = [
            (1, 0), (-1, 0), (0, 1), (0, -1),
            (1, 1), (1, -1), (-1, 1), (-1, -1)
        ]
        for dr, dc in directions:
            r, c = row + dr, col + dc
            while self.inside(r, c):
                p = self._grid[r][c]
                if p:
                    if p.getColor() == by_color:
                        t = p.getType()
                        if (dr == 0 or dc == 0) and t in (PieceType.ROOK, PieceType.QUEEN):
                            return True
                        if (dr != 0 and dc != 0) and t in (PieceType.BISHOP, PieceType.QUEEN):
                            return True
                    break
                r += dr
                c += dc

        # king (adjacent squares)
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                if dr == 0 and dc == 0:
                    continue
                r, c = row + dr, col + dc
                if self.inside(r, c):
                    p = self._grid[r][c]
                    if p and p.getColor() == by_color and p.getType() == PieceType.KING:
                        return True

        return False

    def inCheckState(self):
        king_pos = self._find_king(self._turn)
        if not king_pos:
            return False
        kr, kc = king_pos
        enemy = PieceColor.BLACK if self._turn == PieceColor.WHITE else PieceColor.WHITE
        return self._is_square_attacked(kr, kc, enemy)

    def inCheckMateState(self):
        if not self.inCheckState():
            return False
        # if side to move has no legal moves -> checkmate
        for r in range(1, 9):
            for c in range(1, 9):
                p = self._grid[r][c]
                if p and p.getColor() == self._turn:
                    if self.getValidMoves(r, c):
                        return False
        return True

    # -------------- move generation ---------------

    def _pseudo_moves_for_piece(self, row, col):
        """Generate pseudo-legal moves (not checking self-check)."""
        p = self._grid[row][col]
        if not p:
            return []

        moves = []
        color = p.getColor()
        enemy = PieceColor.BLACK if color == PieceColor.WHITE else PieceColor.WHITE
        ptype = p.getType()

        if ptype == PieceType.PAWN:
            dir_ = 1 if color == PieceColor.WHITE else -1
            start_row = 2 if color == PieceColor.WHITE else 7
            # one step
            r1 = row + dir_
            if self.inside(r1, col) and not self._grid[r1][col]:
                moves.append((r1, col, MoveType.NORMAL))
                # double step
                r2 = row + 2 * dir_
                if row == start_row and self.inside(r2, col) and not self._grid[r2][col]:
                    moves.append((r2, col, MoveType.NORMAL))
            # captures
            for dc in (-1, 1):
                rc = row + dir_
                cc = col + dc
                if self.inside(rc, cc):
                    target = self._grid[rc][cc]
                    if target and target.getColor() == enemy:
                        moves.append((rc, cc, MoveType.CAPTURE))
            # en passant
            if self.en_passant_target:
                er, ec = self.en_passant_target
                if er == row + dir_ and abs(ec - col) == 1:
                    moves.append((er, ec, MoveType.EN_PASSANT))

        elif ptype == PieceType.KNIGHT:
            offsets = [(2, 1), (1, 2), (-1, 2), (-2, 1),
                       (-2, -1), (-1, -2), (1, -2), (2, -1)]
            for dr, dc in offsets:
                r, c = row + dr, col + dc
                if not self.inside(r, c):
                    continue
                target = self._grid[r][c]
                if not target:
                    moves.append((r, c, MoveType.NORMAL))
                elif target.getColor() == enemy:
                    moves.append((r, c, MoveType.CAPTURE))

        elif ptype in (PieceType.BISHOP, PieceType.ROOK, PieceType.QUEEN):
            directions = []
            if ptype in (PieceType.ROOK, PieceType.QUEEN):
                directions += [(1, 0), (-1, 0), (0, 1), (0, -1)]
            if ptype in (PieceType.BISHOP, PieceType.QUEEN):
                directions += [(1, 1), (1, -1), (-1, 1), (-1, -1)]
            for dr, dc in directions:
                r, c = row + dr, col + dc
                while self.inside(r, c):
                    target = self._grid[r][c]
                    if not target:
                        moves.append((r, c, MoveType.NORMAL))
                    else:
                        if target.getColor() == enemy:
                            moves.append((r, c, MoveType.CAPTURE))
                        break
                    r += dr
                    c += dc

        elif ptype == PieceType.KING:
            # normal king moves
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    if dr == 0 and dc == 0:
                        continue
                    r, c = row + dr, col + dc
                    if not self.inside(r, c):
                        continue
                    target = self._grid[r][c]
                    if not target:
                        moves.append((r, c, MoveType.NORMAL))
                    elif target.getColor() == enemy:
                        moves.append((r, c, MoveType.CAPTURE))

            # castling (very basic, assumes standard positions)
            if not p.has_moved and not self._is_square_attacked(row, col, enemy):
                # king side
                if color == PieceColor.WHITE and row == 1 and col == 5:
                    if (self._grid[1][6] is None and self._grid[1][7] is None):
                        rook = self._grid[1][8]
                        if rook and rook.getColor() == color and rook.getType() == PieceType.ROOK and not rook.has_moved:
                            if (not self._is_square_attacked(1, 6, enemy) and
                                    not self._is_square_attacked(1, 7, enemy)):
                                moves.append((1, 7, MoveType.CASTLING))
                if color == PieceColor.BLACK and row == 8 and col == 5:
                    if (self._grid[8][6] is None and self._grid[8][7] is None):
                        rook = self._grid[8][8]
                        if rook and rook.getColor() == color and rook.getType() == PieceType.ROOK and not rook.has_moved:
                            if (not self._is_square_attacked(8, 6, enemy) and
                                    not self._is_square_attacked(8, 7, enemy)):
                                moves.append((8, 7, MoveType.CASTLING))
                # queen side
                if color == PieceColor.WHITE and row == 1 and col == 5:
                    if (self._grid[1][2] is None and self._grid[1][3] is None and self._grid[1][4] is None):
                        rook = self._grid[1][1]
                        if rook and rook.getColor() == color and rook.getType() == PieceType.ROOK and not rook.has_moved:
                            if (not self._is_square_attacked(1, 3, enemy) and
                                    not self._is_square_attacked(1, 4, enemy)):
                                moves.append((1, 3, MoveType.CASTLING))
                if color == PieceColor.BLACK and row == 8 and col == 5:
                    if (self._grid[8][2] is None and self._grid[8][3] is None and self._grid[8][4] is None):
                        rook = self._grid[8][1]
                        if rook and rook.getColor() == color and rook.getType() == PieceType.ROOK and not rook.has_moved:
                            if (not self._is_square_attacked(8, 3, enemy) and
                                    not self._is_square_attacked(8, 4, enemy)):
                                moves.append((8, 3, MoveType.CASTLING))

        return moves

    def _apply_move_on_grid(self, grid, sr, sc, dr, dc, move_type, color):
        """Apply move to a copy of grid (used for legality check)."""
        enemy = PieceColor.BLACK if color == PieceColor.WHITE else PieceColor.WHITE

        piece = grid[sr][sc]
        grid[sr][sc] = None

        # en-passant capture
        if move_type == MoveType.EN_PASSANT:
            dir_ = 1 if color == PieceColor.WHITE else -1
            # captured pawn is behind destination
            grid[dr - dir_][dc] = None

        # castling: move rook in test grid as well
        if move_type == MoveType.CASTLING and piece.getType() == PieceType.KING:
            if color == PieceColor.WHITE and sr == 1:
                if dc == 7:  # king side
                    rook = grid[1][8]
                    grid[1][8] = None
                    grid[1][6] = rook
                elif dc == 3:  # queen side
                    rook = grid[1][1]
                    grid[1][1] = None
                    grid[1][4] = rook
            if color == PieceColor.BLACK and sr == 8:
                if dc == 7:
                    rook = grid[8][8]
                    grid[8][8] = None
                    grid[8][6] = rook
                elif dc == 3:
                    rook = grid[8][1]
                    grid[8][1] = None
                    grid[8][4] = rook

        grid[dr][dc] = piece

    def _is_move_legal(self, sr, sc, dr, dc, move_type):
        p = self._grid[sr][sc]
        if not p:
            return False
        color = p.getColor()
        enemy = PieceColor.BLACK if color == PieceColor.WHITE else PieceColor.WHITE

        grid_copy = copy.deepcopy(self._grid)

        # find king pos before move if needed
        self._apply_move_on_grid(grid_copy, sr, sc, dr, dc, move_type, color)

        # find own king after move
        kr, kc = None, None
        for r in range(1, 9):
            for c in range(1, 9):
                piece = grid_copy[r][c]
                if piece and piece.getColor() == color and piece.getType() == PieceType.KING:
                    kr, kc = r, c
                    break
            if kr is not None:
                break

        if kr is None:
            return False

        # is king attacked?
        return not self._is_square_attacked(kr, kc, enemy)

    def getValidMoves(self, row, col):
        p = self._grid[row][col]
        if not p or p.getColor() != self._turn:
            return []
        pseudo = self._pseudo_moves_for_piece(row, col)
        legal_moves = []
        for dr, dc, mtype in pseudo:
            if self._is_move_legal(row, col, dr, dc, mtype):
                legal_moves.append(Move(dr, dc, mtype))
        return legal_moves

    # -------------- making a move -----------------

    def move(self, sr, sc, dr, dc):
        """Attempt a move for side to move. Returns True if legal & applied."""
        p = self._grid[sr][sc]
        if not p or p.getColor() != self._turn:
            return False

        legal_moves = self.getValidMoves(sr, sc)
        chosen = None
        for mv in legal_moves:
            r, c = mv.getDestinationPosition()
            if r == dr and c == dc:
                chosen = mv
                break
        if not chosen:
            return False

        move_type = chosen.getType()
        color = p.getColor()
        enemy = PieceColor.BLACK if color == PieceColor.WHITE else PieceColor.WHITE

        # update en-passant target
        self.en_passant_target = None

        # en-passant
        if move_type == MoveType.EN_PASSANT:
            dir_ = 1 if color == PieceColor.WHITE else -1
            self._grid[dr - dir_][dc] = None

        # castling rook move
        if move_type == MoveType.CASTLING and p.getType() == PieceType.KING:
            if color == PieceColor.WHITE and sr == 1:
                if dc == 7:
                    rook = self._grid[1][8]
                    self._grid[1][8] = None
                    self._grid[1][6] = rook
                    if rook:
                        rook.has_moved = True
                elif dc == 3:
                    rook = self._grid[1][1]
                    self._grid[1][1] = None
                    self._grid[1][4] = rook
                    if rook:
                        rook.has_moved = True
            if color == PieceColor.BLACK and sr == 8:
                if dc == 7:
                    rook = self._grid[8][8]
                    self._grid[8][8] = None
                    self._grid[8][6] = rook
                    if rook:
                        rook.has_moved = True
                elif dc == 3:
                    rook = self._grid[8][1]
                    self._grid[8][1] = None
                    self._grid[8][4] = rook
                    if rook:
                        rook.has_moved = True

        # double pawn move -> set en-passant target
        if p.getType() == PieceType.PAWN and abs(dr - sr) == 2:
            mid_row = (sr + dr) // 2
            self.en_passant_target = (mid_row, sc)

        # move piece
        self._grid[sr][sc] = None
        self._grid[dr][dc] = p
        p.has_moved = True

        return True

    def nextTurn(self):
        self._turn = PieceColor.BLACK if self._turn == PieceColor.WHITE else PieceColor.WHITE

    # -------------- promotion ---------------------

    def promote(self, row, col, new_type):
        p = self._grid[row][col]
        if p and p.getType() == PieceType.PAWN:
            p.setType(new_type)
import numpy as np
from OpenGL.GL import *

import numpy as np
from OpenGL.GL import *

import numpy as np
from OpenGL.GL import *
import ctypes

class Model:
    """
    OBJ loader using a single VBO (fast).
    Supports:
      - v (vertices)
      - vn (normals)
      - f  (faces with triangles/quads)
    """

    def __init__(self, path):
        self.vertices = np.array([], dtype="float32")
        self.normals = np.array([], dtype="float32")
        self.vbo = None
        self.vertex_count = 0

        self.load_obj(path)
        self.create_vbo()
        print(f"Loaded {path} with {self.vertex_count} vertices")

    def load_obj(self, path):
        verts = []
        norms = []
        faces = []

        with open(path, "r") as f:
            for raw in f:
                line = raw.strip()
                if not line or line.startswith("#"):
                    continue

                parts = line.split()
                tag = parts[0]

                if tag == "v":
                    if len(parts) < 4:
                        continue
                    x, y, z = map(float, parts[1:4])
                    verts.append([x, y, z])

                elif tag == "vn":
                    if len(parts) < 4:
                        continue
                    x, y, z = map(float, parts[1:4])
                    norms.append([x, y, z])

                elif tag == "f":
                    face_verts = []
                    for p in parts[1:]:
                        # supports v, v//n, v/t/n, v/t
                        tokens = (p + "//").split("/")[:3]
                        v_idx = int(tokens[0]) - 1 if tokens[0] else None
                        n_idx = int(tokens[2]) - 1 if tokens[2] else None
                        if v_idx is not None:
                            face_verts.append((v_idx, n_idx))
                    if len(face_verts) >= 3:
                        faces.append(face_verts)

        final_vertices = []
        final_normals = []

        # triangulate faces
        for face in faces:
            v0 = face[0]
            for i in range(1, len(face) - 1):
                v1 = face[i]
                v2 = face[i + 1]
                for v_idx, n_idx in (v0, v1, v2):
                    final_vertices.append(verts[v_idx])
                    if n_idx is not None and 0 <= n_idx < len(norms):
                        final_normals.append(norms[n_idx])
                    else:
                        final_normals.append([0.0, 0.0, 1.0])

        self.vertices = np.array(final_vertices, dtype="float32")
        self.normals = np.array(final_normals, dtype="float32")
        self.vertex_count = len(self.vertices)

    def create_vbo(self):
        if self.vertex_count == 0:
            return

        data = np.hstack((self.vertices, self.normals)).astype("float32")
        self.vbo = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, self.vbo)
        glBufferData(GL_ARRAY_BUFFER, data.nbytes, data, GL_STATIC_DRAW)
        glBindBuffer(GL_ARRAY_BUFFER, 0)

    def Draw(self):
        if self.vbo is None or self.vertex_count == 0:
            return

        glBindBuffer(GL_ARRAY_BUFFER, self.vbo)

        stride = 6 * 4  # 3 floats pos + 3 floats normal

        glEnableClientState(GL_VERTEX_ARRAY)
        glVertexPointer(3, GL_FLOAT, stride, ctypes.c_void_p(0))

        glEnableClientState(GL_NORMAL_ARRAY)
        glNormalPointer(GL_FLOAT, stride, ctypes.c_void_p(12))

        glDrawArrays(GL_TRIANGLES, 0, self.vertex_count)

        glDisableClientState(GL_VERTEX_ARRAY)
        glDisableClientState(GL_NORMAL_ARRAY)
        glBindBuffer(GL_ARRAY_BUFFER, 0)


WINDOW_WIDTH = 800
WINDOW_HEIGHT = 600
WINDOW_POS_X = 51
WINDOW_POS_Y = 51

# Button defines not used in original code, kept for completeness
BUTTON_X = -100
BUTTON_Y = -100
BUTTON_WIDTH = 200
BUTTON_HEIGHT = 75

# LookAt variables
eyeX, eyeY, eyeZ = 5.0, 0.0, -5.0
centerX, centerY, centerZ = 0.0, 0.0, 0.0
upX, upY, upZ = 0.0, 0.0, -1.0

# Perspective variables
fovy, zNear, zFar = 50.0, 0.1, 20.0

# Light variables
position = [0.0, 0.0, 100.0, 0.0]
diffusion = [1.0, 1.0, 1.0, 1.0]
normal_board = [0.0, 0.0, 1.0]
normal_valid_move = [0.0, 0.0, -1.0]
ang = 0.0
mat_diffusion = [0.8, 0.8, 0.8, 1.0]
mat_specular = [0.1, 0.1, 0.1, 1.0]

# View management
screen_ratio = 1.0
zoomOut = 2.0


# Pre-start
pressed = False

# Game loading
chess = None  # will be Game instance later

# Real-time variables
inGame = False
verify = False
selectedRow = 1
selectedCol = 1
moveToRow = 1
moveToCol = 1
selected = False
board_rotating = True
rotation = 0
check = False
checkMate = False
closeGame = False
needPromote = False

# Chess board vertices (same as C++ array)
chessBoard = [
    [-4.0, -4.0, 0.5],
    [-4.0,  4.0, 0.5],
    [ 4.0,  4.0, 0.5],
    [ 4.0, -4.0, 0.5],

    [-4.5, -4.5, 0.5],
    [-4.5,  4.5, 0.5],
    [ 4.5,  4.5, 0.5],
    [ 4.5, -4.5, 0.5],

    [-5.0, -5.0, 0.0],
    [-5.0,  5.0, 0.0],
    [ 5.0,  5.0, 0.0],
    [ 5.0, -5.0, 0.0],
]


# ============================================================
# === PORTED DRAWING / INPUT FUNCTIONS ======================
# ============================================================

def showWord(x, y, word: str):
    glMatrixMode(GL_PROJECTION)
    glLoadIdentity()
    glOrtho(-WINDOW_WIDTH / 2, WINDOW_WIDTH / 2,
            -WINDOW_HEIGHT / 2, WINDOW_HEIGHT / 2,
            0, 1)

    glMatrixMode(GL_MODELVIEW)
    glLoadIdentity()

    glRasterPos2i(x, y)
    glColor3f(1.0, 1.0, 0.0)
    for ch in word:
        glutBitmapCharacter(GLUT_BITMAP_TIMES_ROMAN_24, ord(ch))


def drawMoveToSquare():
    global moveToRow, moveToCol, selected

    r = 1.0 * (moveToRow - 5)
    c = 1.0 * (moveToCol - 5)
    if selected:
        glPushMatrix()
        glColor3f(0.5, 1.0, 0.0)
        glTranslatef(r, c, 0.502)
        glScalef(0.98, 0.98, 1.0)
        glBegin(GL_TRIANGLES)
        glNormal3fv(normal_valid_move)
        glVertex3f(0.0, 0.0, 0.0)
        glVertex3f(1.0, 1.0, 0.0)
        glVertex3f(0.0, 1.0, 0.0)
        glVertex3f(0.0, 0.0, 0.0)
        glVertex3f(1.0, 1.0, 0.0)
        glVertex3f(1.0, 0.0, 0.0)
        glEnd()
        glPopMatrix()
    glColor3f(0.0, 0.0, 0.0)


def drawChessBoard():
    glPushMatrix()
    # bottom
    glNormal3fv(normal_valid_move)
    glBegin(GL_QUADS)
    glColor3f(1.0, 0.0, 0.0)
    for i in range(8, 12):
        glVertex3fv(chessBoard[i])
    glEnd()

    # top quads (same color grad logic)
    glBegin(GL_QUADS)
    glColor3f(0.803, 0.522, 0.247)
    glVertex3fv(chessBoard[0])
    glColor3f(0.545, 0.271, 0.075)
    glVertex3fv(chessBoard[4])
    glColor3f(0.545, 0.271, 0.075)
    glVertex3fv(chessBoard[5])
    glColor3f(0.803, 0.522, 0.247)
    glVertex3fv(chessBoard[1])
    glEnd()

    glBegin(GL_QUADS)
    glColor3f(0.803, 0.522, 0.247)
    glVertex3fv(chessBoard[1])
    glColor3f(0.545, 0.271, 0.075)
    glVertex3fv(chessBoard[5])
    glColor3f(0.545, 0.271, 0.075)
    glVertex3fv(chessBoard[6])
    glColor3f(0.803, 0.522, 0.247)
    glVertex3fv(chessBoard[2])
    glEnd()

    glBegin(GL_QUADS)
    glColor3f(0.803, 0.522, 0.247)
    glVertex3fv(chessBoard[2])
    glColor3f(0.545, 0.271, 0.075)
    glVertex3fv(chessBoard[6])
    glColor3f(0.545, 0.271, 0.075)
    glVertex3fv(chessBoard[7])
    glColor3f(0.803, 0.522, 0.247)
    glVertex3fv(chessBoard[3])
    glEnd()

    glBegin(GL_QUADS)
    glColor3f(0.803, 0.522, 0.247)
    glVertex3fv(chessBoard[3])
    glColor3f(0.545, 0.271, 0.075)
    glVertex3fv(chessBoard[7])
    glColor3f(0.545, 0.271, 0.075)
    glVertex3fv(chessBoard[4])
    glColor3f(0.803, 0.522, 0.247)
    glVertex3fv(chessBoard[0])
    glEnd()

    # side quads
    glBegin(GL_QUADS)
    glColor3f(1.0, 0.95, 0.9)
    glColor3f(0.545, 0.271, 0.075)
    glVertex3fv(chessBoard[4])
    glColor3f(1.0, 1.0, 1.0)
    glVertex3fv(chessBoard[8])
    glColor3f(1.0, 1.0, 1.0)
    glVertex3fv(chessBoard[9])
    glColor3f(0.545, 0.271, 0.075)
    glVertex3fv(chessBoard[5])
    glEnd()

    glBegin(GL_QUADS)
    glColor3f(0.545, 0.271, 0.075)
    glVertex3fv(chessBoard[5])
    glColor3f(1.0, 1.0, 1.0)
    glVertex3fv(chessBoard[9])
    glColor3f(1.0, 1.0, 1.0)
    glVertex3fv(chessBoard[10])
    glColor3f(0.545, 0.271, 0.075)
    glVertex3fv(chessBoard[6])
    glEnd()

    glBegin(GL_QUADS)
    glColor3f(0.545, 0.271, 0.075)
    glVertex3fv(chessBoard[6])
    glColor3f(1.0, 1.0, 1.0)
    glVertex3fv(chessBoard[10])
    glColor3f(1.0, 1.0, 1.0)
    glVertex3fv(chessBoard[11])
    glColor3f(0.545, 0.271, 0.075)
    glVertex3fv(chessBoard[7])
    glEnd()

    glBegin(GL_QUADS)
    glColor3f(0.545, 0.271, 0.075)
    glVertex3fv(chessBoard[7])
    glColor3f(1.0, 1.0, 1.0)
    glVertex3fv(chessBoard[11])
    glColor3f(1.0, 1.0, 1.0)
    glVertex3fv(chessBoard[8])
    glColor3f(0.545, 0.271, 0.075)
    glVertex3fv(chessBoard[4])
    glEnd()

    glPopMatrix()
    glColor3f(0.0, 0.0, 0.0)


def drawBoardSquares():
    global selectedRow, selectedCol, selected
    for row in range(1, 9):
        for col in range(1, 9):
            r = 1.0 * (row - 5)
            c = 1.0 * (col - 5)

            if row == selectedRow and col == selectedCol:
                if selected:
                    glColor3f(0.33, 0.420, 0.184)
                elif chess.isSquareOccupied(selectedRow, selectedCol):
                    if chess.getPieceColor(selectedRow, selectedCol) == chess.getTurnColor():
                        glColor3f(0.0, 0.5, 0.0)
                    else:
                        glColor3f(1.0, 0.0, 0.0)
                else:
                    glColor3f(0.3, 0.7, 0.5)
            else:
                if (row + col) & 1:
                    glColor3f(1.0, 1.0, 1.0)
                else:
                    glColor3f(0.0, 0.0, 0.0)

            glPushMatrix()
            glTranslatef(r, c, 0.5)
            glBegin(GL_TRIANGLES)
            glNormal3fv(normal_board)
            glVertex3f(0.0, 0.0, 0.0)
            glVertex3f(1.0, 1.0, 0.0)
            glVertex3f(0.0, 1.0, 0.0)
            glVertex3f(0.0, 0.0, 0.0)
            glVertex3f(1.0, 1.0, 0.0)
            glVertex3f(1.0, 0.0, 0.0)
            glEnd()
            glPopMatrix()

    glColor3f(0.0, 0.0, 0.0)


def drawValidMoves():
    global selected
    if selected:
        valid_moves = chess.getValidMoves(selectedRow, selectedCol)
        for mv in valid_moves:
            row, col = mv.getDestinationPosition()
            move_type = mv.getType()
            if move_type == MoveType.NORMAL:
                glColor3f(0.8, 1.0, 0.6)
            elif move_type == MoveType.CAPTURE:
                glColor3f(1.0, 0.0, 0.0)
            elif move_type == MoveType.EN_PASSANT:
                glColor3f(0.8, 1.0, 0.6)
            elif move_type == MoveType.CASTLING:
                glColor3f(0.196, 0.804, 0.196)

            r = 1.0 * (row - 5)
            c = 1.0 * (col - 5)
            glPushMatrix()
            glTranslatef(r, c, 0.501)
            glScalef(0.99, 0.99, 1.0)
            glBegin(GL_TRIANGLES)
            glNormal3fv(normal_valid_move)
            glVertex3f(0.0, 0.0, 0.0)
            glVertex3f(1.0, 1.0, 0.0)
            glVertex3f(0.0, 1.0, 0.0)
            glVertex3f(0.0, 0.0, 0.0)
            glVertex3f(1.0, 1.0, 0.0)
            glVertex3f(1.0, 0.0, 0.0)
            glEnd()
            glPopMatrix()
    glColor3f(0.0, 0.0, 0.0)


def drawChessPieces():
    for row in range(1, 9):
        for col in range(1, 9):
            if chess.isSquareOccupied(row, col):
                glPushMatrix()
                if selected and row == selectedRow and col == selectedCol:
                    z = 1.0
                else:
                    z = 0.5
                glTranslatef((row - 5) * 1.0 + 0.5, (col - 5) * 1.0 + 0.5, z)
                glScalef(0.01, 0.01, 0.01)

                piece = chess.getPiece(row, col)
                color = piece.getColor()
                if color == PieceColor.WHITE:
                    glRotatef(90, 0.0, 0.0, 1.0)
                    glColor3f(0.9, 0.9, 0.9)
                else:
                    glRotatef(-90, 0.0, 0.0, 1.0)
                    glColor3f(0.1, 0.1, 0.1)

                ptype = piece.getType()
                if ptype == PieceType.PAWN:
                    models["pawn"].Draw()
                elif ptype == PieceType.ROOK:
                    models["rook"].Draw()
                elif ptype == PieceType.KNIGHT:
                    models["knight"].Draw()
                elif ptype == PieceType.BISHOP:
                    models["bishop"].Draw()
                elif ptype == PieceType.QUEEN:
                    models["queen"].Draw()
                elif ptype == PieceType.KING:
                    models["king"].Draw()

                glPopMatrix()

    glColor3f(0.0, 0.0, 0.0)


def key_W_pressed(color):
    global selectedRow, moveToRow, selected
    if color == PieceColor.WHITE:
        if not selected and selectedRow < 8:
            selectedRow += 1
        if selected and moveToRow < 8:
            moveToRow += 1
    else:  # BLACK
        if not selected and selectedRow > 1:
            selectedRow -= 1
        if selected and moveToRow > 1:
            moveToRow -= 1


def key_D_pressed(color):
    global selectedCol, moveToCol, selected
    if color == PieceColor.WHITE:
        if not selected and selectedCol < 8:
            selectedCol += 1
        if selected and moveToCol < 8:
            moveToCol += 1
    else:  # BLACK
        if not selected and selectedCol > 1:
            selectedCol -= 1
        if selected and moveToCol > 1:
            moveToCol -= 1


def key_S_pressed(color):
    global selectedRow, moveToRow, selected
    if color == PieceColor.WHITE:
        if not selected and selectedRow > 1:
            selectedRow -= 1
        if selected and moveToRow > 1:
            moveToRow -= 1
    else:  # BLACK
        if not selected and selectedRow < 8:
            selectedRow += 1
        if selected and moveToRow < 8:
            moveToRow += 1


def key_A_pressed(color):
    global selectedCol, moveToCol, selected
    if color == PieceColor.WHITE:
        if not selected and selectedCol > 1:
            selectedCol -= 1
        if selected and moveToCol > 1:
            moveToCol -= 1
    else:  # BLACK
        if not selected and selectedCol < 8:
            selectedCol += 1
        if selected and moveToCol < 8:
            moveToCol += 1


def updateTurn(color):
    global selectedRow, selectedCol
    if color == PieceColor.WHITE:
        selectedRow = 1
        selectedCol = 8
    else:
        selectedRow = 8
        selectedCol = 1


def doRotationBoard(color):
    global rotation, board_rotating
    if color == PieceColor.WHITE:
        if rotation < 180:
            rotation += 2
        else:
            board_rotating = False
    else:  # BLACK
        if rotation < 360:
            rotation += 2
        else:
            rotation = 0
            board_rotating = False

def get_view_rotation():
    """
    Returns the board rotation angle for this client.
    - In LOCAL mode: use animated rotation (old behavior).
    - In HOST/CLIENT online mode: lock board to player_color.
    """
    # we read these globals:
    global net_mode, player_color, rotation

    if net_mode == NetMode.LOCAL:
        # keep old animated rotation when playing on same PC
        return rotation
    else:
        # online: lock POV based on who this player is
        if player_color == PieceColor.WHITE:
            return 180.0      # white at bottom
        else:
            return 0.0    # black at bottom


def endOfTurn():
    global selected, needPromote, check, checkMate, board_rotating
    selected = False
    needPromote = False
    check = False

    chess.nextTurn()
    if chess.inCheckMateState():
        globals()['checkMate'] = True
    elif chess.inCheckState():
        globals()['check'] = True

    if net_mode == NetMode.LOCAL:
        board_rotating = True  # only animate in offline mode
    else:
        board_rotating = False  # no animation in online mode

    updateTurn(chess.getTurnColor())

def process_incoming_moves():
    """
    Apply any moves received from the network.
    """
    global incoming_moves

    # Process all queued messages
    while incoming_moves:
        msg = incoming_moves.pop(0)
        if msg.get("type") == "move":
            sr = msg["sr"]
            sc = msg["sc"]
            dr = msg["dr"]
            dc = msg["dc"]
            # Apply the move for the remote player
            # (we assume move is legal; Game will enforce rules anyway)
            if chess.move(sr, sc, dr, dc):
                endOfTurn()


def displayFunction():
    global inGame, zoomOut, check, checkMate, closeGame

    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

    if inGame:
        # Projection
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        gluPerspective(fovy, screen_ratio, zNear, zoomOut * zFar)

        # ModelView
        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()
        gluLookAt(zoomOut * eyeX, zoomOut * eyeY, zoomOut * eyeZ,
                  centerX, centerY, centerZ,
                  upX, upY, upZ)

        # Only animate rotation in LOCAL (same-PC) mode
        if net_mode == NetMode.LOCAL and board_rotating:
            doRotationBoard(chess.getTurnColor())

        ambient_model = [0.5, 0.5, 0.5, 1.0]
        glLightModelfv(GL_LIGHT_MODEL_AMBIENT, ambient_model)
        glMaterialfv(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE, mat_diffusion)
        glMaterialfv(GL_FRONT_AND_BACK, GL_SPECULAR, mat_specular)
        glLightfv(GL_LIGHT0, GL_DIFFUSE, diffusion)

        glScalef(1.0, 1.0, -1.0)
        glLightfv(GL_LIGHT0, GL_POSITION, position)
        # Use locked POV in online mode, animated in local mode
        view_rot = get_view_rotation()
        glRotatef(view_rot, 0.0, 0.0, 1.0)

        drawChessBoard()
        drawBoardSquares()
        drawChessPieces()
        drawMoveToSquare()
        drawValidMoves()

        if needPromote:
            showWord(-200, WINDOW_HEIGHT // 2 - 24,
                     "Promote to: (Q) Queen | (R) Rook | (B) Bishop | (K) Knight")
        elif verify:
            showWord(-200, WINDOW_HEIGHT // 2 - 24,
                     "Are you sure to retry? Yes (O)  or  No (X)")
        else:
            if check:
                s = "BLACK PIECE" if chess.getTurnColor() == PieceColor.BLACK else "WHITE PIECE"
                showWord(-150, WINDOW_HEIGHT // 2 - 24, f"{s} CHECKED!")
            if checkMate:
                s = "WHITE PLAYER" if chess.getTurnColor() == PieceColor.BLACK else "BLACK PLAYER"
                showWord(-100, WINDOW_HEIGHT // 2 - 50, "CHECK MATE!")
                showWord(-140, WINDOW_HEIGHT // 2 - 75, f"{s} WIN!")
                showWord(-150, -WINDOW_HEIGHT // 2 + 50, "Do you want to play again?")
                showWord(-120, -WINDOW_HEIGHT // 2 + 25, "Yes (O)  or  No (X)")
    else:
        showWord(-150, 0, "- - Press N to Start The Game - -")

    # Process network moves (if online)
    if net_mode != NetMode.LOCAL and inGame and chess is not None:
        process_incoming_moves()

    if closeGame:
        sys.exit(0)

    glutSwapBuffers()
    glutPostRedisplay()


def reshapeFunction(width, height):
    global screen_ratio
    if height == 0:
        height = 1
    glViewport(0, 0, width, height)
    screen_ratio = float(width) / float(height)


def specialFunction(key, x, y):
    global zoomOut, ang

    if key == GLUT_KEY_UP:
        zoomOut += 0.2
    elif key == GLUT_KEY_DOWN:
        zoomOut -= 0.2
        if zoomOut < 0.1:
            zoomOut = 0.1
    elif key == GLUT_KEY_LEFT:
        ang += 5
    elif key == GLUT_KEY_RIGHT:
        ang -= 5


def keyFunction(key, x, y):
    global selected, selectedRow, selectedCol, moveToRow, moveToCol
    global inGame, verify, checkMate, closeGame, needPromote

    # In PyOpenGL, 'key' is bytes
    if isinstance(key, bytes):
        key = key.decode('utf-8')

    k = key.lower()

    # 🔒 If online, only allow input when it's THIS player's turn
    if net_mode != NetMode.LOCAL and inGame and chess is not None:
        if chess.getTurnColor() != player_color:
            # allow ESC still
            if key == '\x1b':
                sys.exit(0)
            return

    if k == 'w':
        if not needPromote and not checkMate and not verify and inGame and not board_rotating:
            key_W_pressed(chess.getTurnColor())

    elif k == 'a':
        if not needPromote and not checkMate and not verify and inGame and not board_rotating:
            key_A_pressed(chess.getTurnColor())

    elif k == 's':
        if not needPromote and not checkMate and not verify and inGame and not board_rotating:
            key_S_pressed(chess.getTurnColor())

    elif k == 'd':
        if not needPromote and not checkMate and not verify and inGame and not board_rotating:
            key_D_pressed(chess.getTurnColor())

    elif key == ' ':
        if not needPromote and not checkMate and not verify and inGame and not board_rotating:
            if selected:
                if chess.move(selectedRow, selectedCol, moveToRow, moveToCol):

                    # 🌐 If online, send this move to the other side
                    if net_mode != NetMode.LOCAL:
                        send_move(sock, selectedRow, selectedCol, moveToRow, moveToCol)

                    movedPiece = chess.getPiece(moveToRow, moveToCol)
                    if (movedPiece.getType() == PieceType.PAWN and
                        ((movedPiece.getColor() == PieceColor.BLACK and
                          moveToRow == chess.getBoard().MIN_ROW_INDEX) or
                         moveToRow == chess.getBoard().MAX_ROW_INDEX)):
                        needPromote = True
                    if needPromote:
                        return
                    endOfTurn()
                selected = False
            else:
                if (chess.isSquareOccupied(selectedRow, selectedCol) and
                        chess.getPieceColor(selectedRow, selectedCol) == chess.getTurnColor()):
                    selected = not selected
                    if selected:
                        moveToRow = selectedRow
                        moveToCol = selectedCol

    elif k == 'n':
        if not inGame:
            newGame()
        else:
            verify = True

    elif k == 'o':
        if checkMate or verify:
            # reset game
            newGame()
            verify = False

    elif k == 'x':
        if checkMate:
            closeGame = True
        if verify:
            verify = False

    elif k == 'q':
        if needPromote:
            chess.promote(moveToRow, moveToCol, PieceType.QUEEN)
            endOfTurn()

    elif k == 'r':
        if needPromote:
            chess.promote(moveToRow, moveToCol, PieceType.ROOK)
            endOfTurn()

    elif k == 'b':
        if needPromote:
            chess.promote(moveToRow, moveToCol, PieceType.BISHOP)
            endOfTurn()

    elif k == 'k':
        if needPromote:
            chess.promote(moveToRow, moveToCol, PieceType.KNIGHT)
            endOfTurn()

    elif key == '\x1b':  # ESC
        sys.exit(0)



def initialize():
    glClearColor(0.2, 0.6, 0.5, 1.0)
    glClearDepth(1.0)
    glEnable(GL_DEPTH_TEST)
    # glEnable(GL_SMOOTH)   # ❌ REMOVE THIS LINE
    glDepthFunc(GL_LEQUAL)
    glShadeModel(GL_SMOOTH)     # ✅ this is correct
    glEnable(GL_NORMALIZE)
    glHint(GL_PERSPECTIVE_CORRECTION_HINT, GL_NICEST)

    glEnable(GL_COLOR_MATERIAL)
    glEnable(GL_LIGHTING)
    glEnable(GL_LIGHT0)


def newGame():
    global chess, selectedRow, selectedCol, moveToRow, moveToCol
    global selected, board_rotating, rotation, inGame, check, checkMate, needPromote, verify
    global net_mode
    
    chess = Game()
    selectedRow = 1
    selectedCol = 1
    moveToRow = 1
    moveToCol = 1
    selected = False
    
    rotation = 0
    inGame = True
    check = False
    checkMate = False
    needPromote = False
    verify = False
    
    # ✅ Only rotate the board in LOCAL (two-players-on-same-PC) mode
    if net_mode == NetMode.LOCAL:
        board_rotating = True
    else:
        board_rotating = False
    updateTurn(chess.getTurnColor())

def main():
    global models, net_mode, player_color, sock, recv_thread, host_ip

    # --- Choose mode before creating window ---
    print("=== 3D Chess Mode Selection ===")
    print("1) Offline (two players on same PC)")
    print("2) Host online game (you are WHITE)")
    print("3) Join online game (you are BLACK)")
    choice = input("Enter 1 / 2 / 3 [default 1]: ").strip()

    if choice == "2":
        net_mode = NetMode.HOST
        player_color = PieceColor.WHITE
    elif choice == "3":
        net_mode = NetMode.CLIENT
        player_color = PieceColor.BLACK
        host = input("Enter host IP [default 127.0.0.1]: ").strip()
        if host:
            host_ip = host
    else:
        net_mode = NetMode.LOCAL
        player_color = PieceColor.WHITE

    # --- OpenGL / GLUT setup ---
    glutInit(sys.argv)
    glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGB | GLUT_DEPTH)
    glutInitWindowSize(WINDOW_WIDTH, WINDOW_HEIGHT)
    glutInitWindowPosition(WINDOW_POS_X, WINDOW_POS_Y)
    glutCreateWindow(b"Chess - PyOpenGL")

    initialize()

    # Load models AFTER context creation
    models["pawn"]   = Model("model/Pawn.obj")
    models["rook"]   = Model("model/Rook.obj")
    models["knight"] = Model("model/Knight.obj")
    models["bishop"] = Model("model/Bishop.obj")
    models["queen"]  = Model("model/Queen.obj")
    models["king"]   = Model("model/King.obj")

    # --- Networking init (if online) ---
    if net_mode == NetMode.HOST:
        sock, recv_thread = start_host(NET_PORT, incoming_moves)
    elif net_mode == NetMode.CLIENT:
        sock, recv_thread = start_client(host_ip, NET_PORT, incoming_moves)

    # GLUT callbacks
    glutDisplayFunc(displayFunction)
    glutReshapeFunc(reshapeFunction)
    glutKeyboardFunc(keyFunction)
    glutSpecialFunc(specialFunction)

    glutMainLoop()
    
if __name__ == "__main__":
    main()

