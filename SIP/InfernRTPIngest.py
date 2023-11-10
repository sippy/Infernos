from queue import Queue, Empty as QueueEmpty

from .InfernWrkThread import InfernWrkThread, RTPWrkTRun

class InfernRTPIngest(InfernWrkThread):
    pkt_queue: Queue = None
    def __init__(self):
        super().__init__()
        self.pkt_queue = Queue()
#        self.start()

    def run(self):
        super().thread_started()
        print("InfernRTPIngest started")
        data, address, rtime = (None, None, None)
        npkts = 0
        while self.get_state() == RTPWrkTRun:
            try:
                data, address, rtime = self.pkt_queue.get(timeout=0.03)
            except QueueEmpty:
                continue
            npkts += 1
            if npkts == 1:
                print(f"InfernRTPIngest.run: address={address}, rtime={rtime}, len(data) = {len(data)} data={data[:40]}")
        print(f"InfernRTPIngest.run: last packet: address={address}, rtime={rtime}, len(data) = {len(data)} data={data[:40]}")
        print(f"InfernRTPIngest.run: exiting, total packets received: {npkts}")

    def stop(self):
        super().stop()
        print("InfernRTPIngest stopped")

    def rtp_received(self, data, address, udp_server, rtime):
        self.pkt_queue.put((data, address, rtime))
