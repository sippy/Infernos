from threading import Thread

class InfernRTPIngest(Thread):
    def __init__(self):
        Thread.__init__(self)
        self.setDaemon(True)
#        self.start()

    def run(self):
        print("InfernRTPIngest started")

    def stop(self):
        print("InfernRTPIngest stopped")

    def rtp_received(self, data, address, udp_server, rtime):
        print(f"InfernRTPIngest.rtp_received called (data={data}, address={address}, rtime={rtime})")
