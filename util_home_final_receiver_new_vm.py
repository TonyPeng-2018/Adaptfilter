# receive tcp package from the device
import socket
from struct import unpack
import time
import sys
import os

class Server:
    def __init__(self, host, port, model, sampler):
        # host = 'localhost'
        # host = '100.64.0.2'
        # host = '100.64.0.4'
        host = "127.0.0.1"
        self.i_stop = 1000
        self.port = port
        self.model = model
        self.sampler = sampler
        self.s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

        self.s.bind((host, port))
        self.s.listen(10)

    def receiver(self, rate):
        try:
            i_stop = self.i_stop

            client_socket, addr = self.s.accept()
            print("Got a connection from %s" % str(addr))

            experiment_root = "data/experiment/"
            avg_receive_time = []
            threds = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6 ,0.7, 0.8, 0.9, 0.95, 1.0]
            

            for thred in threds:
                network_latency = [0] * (i_stop)
                data_path = experiment_root + self.model + "_" + self.sampler + "_" + str(thred) + "/"
                if not os.path.exists(data_path):
                    return 0

                packet_receive_time = 0
                received_bytes = 0

                f = open(data_path + f"network_latency_{rate}.txt", "w")
                
                for i in range(i_stop):
                    if i%10 == 0:
                        print(i)
                    t1 = time.time()
                    data = client_socket.recv(8)  # length of encoded img
                    length = unpack(">Q", data)[0]
                    data = b""
                    while len(data) < length:
                        data += client_socket.recv(
                            1024 if length - len(data) > 1024 else length - len(data)
                        )
                    t2 = time.time()
                    received_bytes += len(data)
                    # send received to the client
                    client_socket.sendall(b"1")
                    network_latency[i] = t2 - t1
                    packet_receive_time += t2 - t1
                    f.write(f"{i}: " + str(t2 - t1) + "\n")
                    #
                packet_receive_time = [x * 1000 / i_stop for x in network_latency]
                received_bytes = [x / i_stop for x in received_bytes]
                received_bytes = ['%.2f' % elem for elem in received_bytes]
                packet_receive_time = ['%.2f' % elem for elem in packet_receive_time]
                print(
                    data_path,
                    " avg received bytes:",
                    received_bytes,
                    " avg receive time:",
                    packet_receive_time,
                )
                avg_receive_time.append(packet_receive_time)
                # store the latency matrix
                np.save(data_path + "network_latency.npy", network_latency)
                f.write(str(avg_receive_time) + "\n")
            self.s.close()
        except Exception as e:
            print(e)
            self.s.close()


if __name__ == "__main__":
    model = sys.argv[1]   
    sampler = sys.argv[2]
    rate = sys.argv[3]
    # ip_ind = sys.argv[4]
    # thred = float(sys.argv[5])
    # target_ip = f"127.0.{ip_ind}.1"
    target_ip = "127.0.0.1"
    port = 5567

    # print(ip_ind)

    # write command in sudo
    # sudo python3 util_home_final_receiver_new.py
    # import os
    # os.system(f"sudo ifconfig lo:{ip_ind} 127.0.{ip_ind}.1 up")
    # os.system(f"sudo tc qdisc add dev lo:{ip_ind} root netem delay 20ms rate {rate}bit")
    # threds = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6 ,0.7, 0.8, 0.9, 0.95, 1.0]
    # for thred in threds:
    #     os.system(f"python util_home_final_sender_new.py {model} {sampler} {target_ip} {port} {thred} &")
    #     print('running receiver')
    server = Server(f"127.0.0.1", port, model, sampler)
    server.receiver(rate)
    # os.system(f"sudo tc qdisc del dev lo:{ip_ind} root")
    # os.system(f"sudo ifconfig lo:{ip_ind} down")
