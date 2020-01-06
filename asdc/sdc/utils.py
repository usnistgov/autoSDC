def encode(message):
    message = message + '\r\n'
    return message.encode()

def decode(message):
    """ bytes to str; strip carriage return """

    if type(message) is list:
        return [decode(msg) for msg in message]

    return message.decode().strip()
