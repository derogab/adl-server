class Reply:

    def __init__(self):
        raise TypeError("cannot create 'Reply' instances")
    
    @staticmethod
    def handshake():

        reply = {"status": "OK", "type": "handshake"}
        reply_str = str(reply) + "\n" # \n is essential
        reply_bytes = bytes(reply_str, 'UTF-8')

        return reply_bytes

    @staticmethod
    def goodbye():

        reply = {"status": "OK", "type": "goodbye"}
        reply_str = str(reply) + "\n" # \n is essential
        reply_bytes = bytes(reply_str, 'UTF-8')

        return reply_bytes

    @staticmethod
    def ack():

        reply = {"status": "OK", "type": "ack"}
        reply_str = str(reply) + "\n" # \n is essential
        reply_bytes = bytes(reply_str, 'UTF-8')

        return reply_bytes

    @staticmethod
    def close():

        reply = {"status": "OK", "type": "close"}
        reply_str = str(reply) + "\n" # \n is essential
        reply_bytes = bytes(reply_str, 'UTF-8')

        return reply_bytes

    @staticmethod
    def destroy():

        reply = {"status": "OK", "type": "destroy"}
        reply_str = str(reply) + "\n" # \n is essential
        reply_bytes = bytes(reply_str, 'UTF-8')

        return reply_bytes

    @staticmethod
    def prediction(activity):

        reply = {"status": "OK", "type": "prediction", "activity": activity}
        reply_str = str(reply) + "\n" # \n is essential
        reply_bytes = bytes(reply_str, 'UTF-8')

        return reply_bytes
