import zmq
import pickle

class ZeroMQClient:
    """
    A client class for communicating with a ZeroMQ server using REQ/REP pattern.
    """

    def __init__(self, addr: str, port: int = 7000, timeout: int = 15000):
        """
        Initializes the ZeroMQ client.

        Args:
            addr (str): The server's IP address or hostname.
            port (int): The port to connect to (default is 7000).
            timeout (int): The timeout period in milliseconds for receiving data (default is 15000).
        """
        self.addr = addr
        self.port = port
        self.timeout = timeout
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.REQ)
        self.socket.setsockopt(zmq.RCVTIMEO, self.timeout)

    def connect(self):
        """
        Connects the client to the ZeroMQ server.

        This method establishes a connection to the server at the specified address and port.
        """
        self.socket.connect(f'tcp://{self.addr}:{self.port}')
        print(f"Connected to {self.addr}:{self.port}")

    def send_request(self, request: str):
        """
        Sends a request to the server and waits for the response.

        Args:
            request (str): The message to be sent to the server.

        Returns:
            str: The response from the server if successful, None if timeout occurs.
        """
        try:
            # Serialize the request
            self.socket.send(pickle.dumps(request))
            print("Request sent")
            
            # Wait for the response
            data = self.socket.recv()
            response = pickle.loads(data)
            print(f"Data received: {response}")
            return response

        except zmq.Again:
            print("Timeout reached, no response received.")
            return None
        except Exception as e:
            print(f"Error sending or receiving data: {e}")
            return None

    def close(self):
        """
        Closes the ZeroMQ client connection.

        This method terminates the connection and releases any resources held by the client.
        """
        self.socket.close()
        self.context.term()
        print("Connection closed")


if __name__ == "__main__":
    zmq_client = ZeroMQClient(addr='161.72.210.177', timeout=15000)    
    zmq_client.connect()

    while True:
        try:
            request = "give me data."
            response = zmq_client.send_request(request)
            print(f"Response from server: {response}")
            

        except KeyboardInterrupt:
            print("Interrupted by user. Closing connection.")
            break
    zmq_client.close()

