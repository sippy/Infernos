class SipSessInfo():
    call_id: str
    from_number: str
    from_name: str

    def __init__(self, call_id, from_number, from_name):
        self.call_id = call_id
        self.from_number = from_number
        self.from_name = from_name