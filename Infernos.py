from getopt import getopt, GetoptError
import os, sys

import ray

from sippy.misc import daemonize
from sippy.SipLogger import SipLogger

sys.path.append('.')

from SIP.InfernUA import InfernSIPConf
from Cluster.InfernSIPActor import InfernSIPActor
from Core.InfernConfig import InfernConfig

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

def usage():
    print('usage: Infernos.py [-f] [-L logfile] [-i pidfile] [myconfig.yaml]')
    sys.exit(1)

if __name__ == '__main__':
    try:
        opts, args = getopt(sys.argv[1:], 'fL:i:')
    except GetoptError:
        usage()

    if len(args) > 1:
        usage()

    cfile = 'config.yaml' if len(args) == 0 else args[0]

    idir = os.path.realpath(sys.argv[0])
    idir = os.path.dirname(idir)
    sys.path.append(idir)
    authname = None
    authpass = None
    logfile = '/var/log/Infernos.log'
    pidfile = None
    foreground = False
    for o, a in opts:
        if o == '-f':
            foreground = True
        elif o == '-L':
            logfile = a
#        elif o == '-n':
#            if a.startswith('['):
#                parts = a.split(']', 1)
#                iua_c.nh_addr = [parts[0] + ']', 5060]
#                parts = parts[1].split(':', 1)
#            else:
#                parts = a.split(':', 1)
#                iua_c.nh_addr = [parts[0], 5060]
#            if len(parts) == 2:
#                iua_c.nh_addr[1] = int(parts[1])
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
    default_resources['live_translator'] = 1
    default_resources['tts'] = 2
    default_resources['stt'] = 1
    try:
        ray.init(num_gpus=1, resources = default_resources)
    except ValueError as ex:
        if str(ex).index('connecting to an existing cluster') < 0: raise ex
        ray.init()

    inf_c = InfernConfig(cfile)

    if pidfile != None:
        open(pidfile, 'w').write('%d' % os.getpid())

    inf_c.sip_conf.logger = SipLogger('Infernos',  logfile = os.path.expanduser('~/.Infernos.log'))

    if inf_c.sip_actr is None:
        ray.shutdown()
        exit(0)

    try:
        exit(ray.get(inf_c.sip_actr.loop.remote(inf_c)))
    except KeyboardInterrupt:
        ray.get(inf_c.sip_actr.stop.remote())
        raise
