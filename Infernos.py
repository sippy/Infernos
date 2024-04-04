from getopt import getopt, GetoptError
import os, sys

import ray

from sippy.misc import daemonize
from sippy.SipLogger import SipLogger

sys.path.append('.')

from SIP.InfernUA import InfernUAConf
from Cluster.InfernSIPActor import InfernSIPActor

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
    iua_c = InfernUAConf()
    foreground = False
    for o, a in opts:
        if o == '-f':
            foreground = True
        elif o == '-l':
            iua_c.laddr = a
        elif o == '-p':
            iua_c.lport = int(a)
        elif o == '-L':
            logfile = a
        elif o == '-n':
            if a.startswith('['):
                parts = a.split(']', 1)
                iua_c.nh_addr = [parts[0] + ']', 5060]
                parts = parts[1].split(':', 1)
            else:
                parts = a.split(':', 1)
                iua_c.nh_addr = [parts[0], 5060]
            if len(parts) == 2:
                iua_c.nh_addr[1] = int(parts[1])
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

    default_resources = InfernSIPActor.default_resources
    try:
        ray.init(num_gpus=1, resources = default_resources)
    except ValueError as ex:
        if str(ex).index('connecting to an existing cluster') < 0: raise ex
        ray.init()

    if pidfile != None:
        open(pidfile, 'w').write('%d' % os.getpid())

    iua_c.logger = SipLogger('Infernos',  logfile = os.path.expanduser('~/.Infernos.log'))

    iua_c.authname = authname
    iua_c.authpass = authpass
    iua_c.cli = iua_c.cld = authname
    iua = InfernSIPActor.options(max_concurrency=2).remote(iua_c)
    try:
        exit(ray.get(iua.loop.remote()))
    except KeyboardInterrupt:
        ray.get(iua.stop.remote())
        raise
