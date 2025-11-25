# net.py
import socket
import threading
import json
from enum import Enum, auto


class NetMode(Enum):
    LOCAL = 0   # no networking, both players on same PC
    HOST = 1    # create server, play as WHITE
    CLIENT = 2  # connect to server, play as BLACK


def _network_listener(sock, incoming_moves):
    """
    Runs in a background thread.
    Reads newline-separated JSON messages from the socket and
    appends them to the incoming_moves list.
    """
    buffer = ""
    try:
        while True:
            data = sock.recv(1024)
            if not data:
                print("Connection closed by remote.")
                break
            buffer += data.decode("utf-8")
            # messages are newline-separated JSON
            while "\n" in buffer:
                line, buffer = buffer.split("\n", 1)
                line = line.strip()
                if not line:
                    continue
                try:
                    msg = json.loads(line)
                    incoming_moves.append(msg)
                except json.JSONDecodeError:
                    print("Failed to decode message:", line)
    except OSError as e:
        print("Network error:", e)
    finally:
        try:
            sock.close()
        except OSError:
            pass


def start_host(port, incoming_moves):
    """
    Start as host (server). Blocks until a client connects.
    Returns (conn_socket, listener_thread).
    """
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    s.bind(("", port))
    s.listen(1)
    print(f"[HOST] Waiting for client on port {port}...")
    conn, addr = s.accept()
    print(f"[HOST] Client connected from {addr}")
    t = threading.Thread(target=_network_listener, args=(conn, incoming_moves), daemon=True)
    t.start()
    return conn, t


def start_client(host, port, incoming_moves):
    """
    Start as client. Connects to given host:port.
    Returns (socket, listener_thread).
    """
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    print(f"[CLIENT] Connecting to {host}:{port}...")
    s.connect((host, port))
    print("[CLIENT] Connected!")
    t = threading.Thread(target=_network_listener, args=(s, incoming_moves), daemon=True)
    t.start()
    return s, t


def send_move(sock, sr, sc, dr, dc):
    """
    Send a move over the network: source row/col, dest row/col.
    """
    if sock is None:
        return
    msg = {
        "type": "move",
        "sr": sr,
        "sc": sc,
        "dr": dr,
        "dc": dc,
    }
    data = (json.dumps(msg) + "\n").encode("utf-8")
    try:
        sock.sendall(data)
    except OSError as e:
        print("Send error:", e)
