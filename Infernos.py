from getopt import getopt, GetoptError
import os, sys

from sippy.misc import daemonize
from sippy.Core.EventDispatcher import ED2
from sippy.SipLogger import SipLogger

sys.path.append('.')

from SIP.InfernUAS import InfernSIP, InfernUASConf

def patch_signals():
    import threading
    import signal

    def _start_new_thread(*args):
        allsigs = list(signal.valid_signals())

        old_sigset = signal.pthread_sigmask(signal.SIG_BLOCK, allsigs)
        ret = _old_start_new_thread(*args)
        signal.pthread_sigmask(signal.SIG_SETMASK, old_sigset)
        return ret

    _old_start_new_thread = threading._start_new_thread
    threading._start_new_thread = _start_new_thread

if __name__ == '__main__':
    try:
        opts, args = getopt(sys.argv[1:], 'fl:p:n:L:s:u:P:i:')
    except GetoptError:
        print('usage: Infernos.py [-l addr] [-p port] [-n addr] [-f] [-L logfile] [-u authname [-P authpass]]\n' \
          '                   [-i pidfile]')
        sys.exit(1)
    sdev = None
    idir = os.path.realpath(sys.argv[0])
    idir = os.path.dirname(idir)
    sys.path.append(idir)
    authname = None
    authpass = None
    logfile = '/var/log/Infernos.log'
    pidfile = None
    iuac = InfernUASConf()
    foreground = False
    for o, a in opts:
        if o == '-f':
            foreground = True
        elif o == '-l':
            iuac.laddr = a
        elif o == '-p':
            iuac.lport = int(a)
        elif o == '-L':
            logfile = a
        elif o == '-n':
            if a.startswith('['):
                parts = a.split(']', 1)
                iuac.nh_addr = [parts[0] + ']', 5060]
                parts = parts[1].split(':', 1)
            else:
                parts = a.split(':', 1)
                iuac.nh_addr = [parts[0], 5060]
            if len(parts) == 2:
                iuac.nh_addr[1] = int(parts[1])
        elif o == '-s':
            sdev = a
        elif o == '-u':
            authname = a
        elif o == '-P':
            authpass = a
        elif o == '-i':
            pidfile = a

    if not foreground:
        daemonize(logfile)

    patch_signals()

    if logfile == '-':
        lfile = sys.stdout
    else:
        lfile = open(logfile, 'a')

    if pidfile != None:
        open(pidfile, 'w').write('%d' % os.getpid())

    iuac.logger = SipLogger('Infernos')

    iuac.authname = authname
    iuac.authpass = authpass
    iuac.cli = iuac.cld = authname
    iua = InfernSIP(iuac)
    #pio = PELIO(lfile)
    #if sdev != None:
    #    pio.sdev = sdev
    #pio.sstart_cb = pua.sess_started
    #pio.send_cb = pua.sess_ended
    #pio.start()
    exit(ED2.loop())

