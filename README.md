![Demo GIF](media/demo.gif)

# ♟️ 3D Chess – PyOpenGL Multiplayer Chess Game

This project is a fully interactive **3D Chess Game** built using **Python**, **PyOpenGL**, and **GLUT**, with support for:

- ✔️ Offline 2-player mode (same PC)  
- ✔️ Online multiplayer (Host & Client)  
- ✔️ Full legal chess rules  
- ✔️ Real 3D chess pieces (OBJ models)  
- ✔️ Smooth rendering and lighting  
- ✔️ Move validation, check & checkmate  
- ✔️ Pawn promotion, castling, en-passant  
- ✔️ Board rotation locked per-player in online mode  

The goal of the project is to provide a simple but visually appealing 3D chess engine that works both locally and over LAN.

---

# 📁 Folder Structure

```
3D-Chess/
│── main.py                # Main OpenGL rendering + game logic
│── net.py                 # Networking (Host/Client communication)
│── requirements.txt       # Dependencies
│── model/                 # 3D chess piece models (OBJ)
│   ├── Pawn.obj
│   ├── Rook.obj
│   ├── Knight.obj
│   ├── Bishop.obj
│   ├── Queen.obj
│   ├── King.obj
```

---

# 🚀 Features

### 🎮 Game Mechanics
- Full chess engine:
  - Legal move generation
  - Check & checkmate detection
  - En-passant
  - Castling
  - Promotion
- Smooth 3D animations
- Select / move tiles using keyboard  
- Board orientation automatically adjusts to player color

### 🌐 Online Play
The game supports **LAN multiplayer** using Python sockets.

In `main.py`:

```python
print("1) Offline (two players on same PC)")
print("2) Host online game (you are WHITE)")
print("3) Join online game (you are BLACK)")
```

Hosting uses:

```python
sock, recv_thread = start_host(NET_PORT, incoming_moves)
```

Joining uses:

```python
sock, recv_thread = start_client(host_ip, NET_PORT, incoming_moves)
```

All moves are synchronized across both players in real-time.

---

# 🖼️ 3D Models

The `model/` folder contains OBJ models for all chess pieces:

```
Pawn.obj
Rook.obj
Knight.obj
Bishop.obj
Queen.obj
King.obj
```

They are loaded using the custom loader:

```python
models["pawn"] = Model("model/Pawn.obj")
```

Rendering uses VBOs for high performance:

```python
glBindBuffer(GL_ARRAY_BUFFER, self.vbo)
glDrawArrays(GL_TRIANGLES, 0, self.vertex_count)
```

---

# 🎮 Controls

| Key | Action |
|-----|--------|
| **W / A / S / D** | Move cursor on board |
| **Space** | Select / move piece |
| **N** | New game |
| **O** | Confirm (restart / menu) |
| **X** | Cancel / exit prompt |
| **Q / R / B / K** | Promote pawn |
| **ESC** | Quit game |

---

# 🔧 Installation

### 1️⃣ Create a virtual environment
```
python -m venv venv
```

### 2️⃣ Activate it  
Windows:
```
venv\Scripts\activate
```

### 3️⃣ Install dependencies
```
pip install -r requirements.txt
```

---

# ▶️ Running the Game

```
python main.py
```

Choose one of:

- `1` → offline mode  
- `2` → host online game  
- `3` → join online game  

On the **client side**, enter the host’s LAN IP.

---

# 🧠 How Networking Works

Movement messages are exchanged as JSON-like objects:

```python
{
  "type": "move",
  "sr": 1, "sc": 2,
  "dr": 3, "dc": 2
}
```

`process_incoming_moves()` applies remote moves:

```python
if chess.move(sr, sc, dr, dc):
    endOfTurn()
```

---

# 👤 Author
This project is created and maintained by Shubhro Shekhar Das , designed to make learning PyOpenGL and game architecture easier for beginners.

Feel free to modify, extend, or share this project.

---

# 📄 License

MIT License


