import socket
import json
import struct
import constants
import torch
import threading
import signal
import traceback
from utils import Message, State, Bot
import utils






class Server:

    MESSAGE_IN_UPDATED = False
    MESSAGE_IN = Message("temp")
    MESSAGE_OUT = Message("waiting_for_connection")
    MESSAGE_IN_UPDATED = False
    SOCKET_OPEN = True
    STATE = State.WAIT_FOR_CONNECTION


    def start():

        def server_handler(signum, frame):
            Server.SOCKET_OPEN = False
            exit(1)

        signal.signal(signal.SIGINT, server_handler)
        server_thread = threading.Thread(target = Server._start)
        server_thread.start()


    def _start():
        server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server_socket.bind((constants.HOST, constants.PORT))
        server_socket.listen()


        while Server.SOCKET_OPEN:
            client_socket, addr = server_socket.accept()
            length_bytes = client_socket.recv(2)

            if len(length_bytes) == 2:
                data_size = struct.unpack('>H', length_bytes)[0]

                #print(f"Connected by {addr}")
                data = client_socket.recv(data_size)
                while(len(data) < data_size):
                    chunk = client_socket.recv(data_size - len(data))
                    if not chunk:
                        break

                    data += chunk

            else:
                print("FIX THIS!")


            Server.MESSAGE_IN = Message(**json.loads(data.decode('utf-8')))
            Server.MESSAGE_IN_UPDATED = True
            Server.manage_state(Server.MESSAGE_IN)

            #print(Server.MESSAGE_OUT.command)
            #print(Server.MESSAGE_OUT.info)
            response = Server.MESSAGE_OUT.to_json_out()
            
            
            client_socket.send(response)
            client_socket.close()


    def manage_state(message):
        last_response = message
        if Server.STATE == State.RESET_EPISDOE:
            return
        elif message.command == "Connected":
            Server.STATE = State.SPAWN_BOTS
            print("State: Spawn Bots")
        elif message.command in ["Wait", "Success: spawn_bots"]:
            Server.STATE = State.WAIT_FOR_DATA
            print("State: Waiting for DATA")
        elif message.command == "json":
            Server.last_response = json.loads(message.info)
            Server.STATE = State.SEND_ACTION

    def step(data):
        Server.MESSAGE_OUT = Message("json", data)
        
    
    def close():
        Server.SOCKET_OPEN = False

        



    def update_message():
        if Server.STATE == State.WAIT_FOR_CONNECTION:
            Server.MESSAGE_OUT = Message("waiting_for_connection")
        if Server.STATE == State.WAIT_FOR_DATA:
            Server.MESSAGE_OUT = Message("server_waiting")
        if Server.STATE in [State.SPAWN_BOTS, State.RESET_EPISDOE]:
            Server.STATE = State.SPAWN_BOTS
            botinfo = Bot({"task":"woodcutting", "nodesRange": constants.NODES_RANGE})
            botinfo = json.dumps([botinfo.info])
            Server.MESSAGE_OUT = Message(f"spawn_bots {constants.SPAWN_LOCATION[0]} {constants.SPAWN_LOCATION[1]} {constants.NUM_BOTS}", botinfo )


        Server.MESSAGE_IN_UPDATED = False



 
        
    











 




#batch_size = 128
#epochs = 5
#lr = 0.0001


#AGENT = QLearningAgent(constants.STATE_SIZE, constants.ACTION_SIZE, lr=lr)
#REPLAY_MEMORY = ReplayMemory()
#EPISODE_NUM_STEPS = constants.EPISODE_NUM_STEPS_MIN
#STEPS_THIS_EPISODE = 0




    