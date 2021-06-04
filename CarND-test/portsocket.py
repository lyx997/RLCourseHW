from socket import *
import _thread
def getData(tcpCliSock):
    while True:
        try:
            print('hello')
        except:
            break
    tcpCliSock.close()
    _thread.stop()

if __name__ == '__main__':
    HOST = ""
    PORT = 8588
    BUFSIZ = 1024    #缓冲区大小
    ADDR = (HOST,PORT)
    tcpSerSock = socket(AF_INET,SOCK_STREAM)
    tcpSerSock.bind(ADDR)
    tcpSerSock.listen(60)    #连接被转接或者被拒绝之前，传入连接请求的最大数
    while True:
        tcpCliSock, addr = tcpSerSock.accept()
        print("waiting for connect ...")
        print("...connect from:", addr)
        _thread.start_new_thread(getData, (tcpCliSock, ))
        continue
