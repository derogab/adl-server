class Reply:
    def __init__(self):

        pass

    def handshake():

        reply = {"status": "OK", "type": "handshake"}
        reply_str = str(reply) + "\n" # \n is essential
        reply_bytes = bytes(reply_str, 'UTF-8')

        return reply_bytes

    def ack():

        reply = {"status": "OK", "type": "ack"}
        reply_str = str(reply) + "\n" # \n is essential
        reply_bytes = bytes(reply_str, 'UTF-8')

        return reply_bytes

    def close():

        reply = {"status": "OK", "type": "close"}
        reply_str = str(reply) + "\n" # \n is essential
        reply_bytes = bytes(reply_str, 'UTF-8')

        return reply_bytes


# create Reply static methods
Reply.handshake = staticmethod(Reply.handshake)
Reply.ack       = staticmethod(Reply.ack)
Reply.close     = staticmethod(Reply.close)