from queue import Queue, Empty as QueueEmpty

from rtpsynth.RtpJBuf import RtpJBuf, RTPParseError

from Core.InfernWrkThread import InfernWrkThread, RTPWrkTRun

class InfernRTPIngest(InfernWrkThread):
    debug = True
    pkt_queue: Queue = None
    jb_size: int = 8
    input_sr: int = 8000
    output_sr: int
    codec: str = "PCMU"
    def __init__(self, output_sr: int = 16000):
        super().__init__()
        self.pkt_queue = Queue()
        self.output_sr = output_sr
#        self.start()

    def dprint(self, *args):
        if self.debug:
            print(*args)

    def run(self):
        super().thread_started()
        self.dprint("InfernRTPIngest started")
        data, address, rtime = (None, None, None)
        npkts = 0
        jbuf = RtpJBuf(self.jb_size)
        while self.get_state() == RTPWrkTRun:
            try:
                data, address, rtime = self.pkt_queue.get(timeout=0.03)
            except QueueEmpty:
                continue
            try:
                res = jbuf.udp_in(data)
            except RTPParseError as e:
                self.dprint(f"InfernRTPIngest.run: RTPParseError: {e}")
                continue
            npkts += 1
            if npkts == 1:
                self.dprint(f"InfernRTPIngest.run: address={address}, rtime={rtime}, len(data) = {len(data)} data={data[:40]}")
            if npkts < 10 and len(res) > 0:
                self.dprint(f"InfernRTPIngest.run: res = {res}")
        if data is not None:
            self.dprint(f"InfernRTPIngest.run: last packet: address={address}, rtime={rtime}, len(data) = {len(data)} data={data[:40]}")
        self.dprint(f"InfernRTPIngest.run: exiting, total packets received: {npkts}")

    def stop(self):
        super().stop()
        self.dprint("InfernRTPIngest stopped")

    def rtp_received(self, data, address, udp_server, rtime):
        #self.dprint(f"InfernRTPIngest.rtp_received: len(data) = {len(data)}")
        self.pkt_queue.put((data, address, rtime))

    def __del__(self):
        self.dprint("InfernRTPIngest.__del__")
