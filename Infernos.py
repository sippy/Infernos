from getopt import getopt, GetoptError
import os, sys

from sippy.misc import daemonize
from sippy.Core.EventDispatcher import ED2
from sippy.SipConf import SipConf
from sippy.SipLogger import SipLogger

sys.path.append('.')

from SIP.InfernUAS import InfernUAS, InfernUASConf

if __name__ == '__main__':
    try:
        opts, args = getopt(sys.argv[1:], 'fl:p:n:L:s:u:P:i:')
    except GetoptError:
        print('usage: pel_collect.py [-l addr] [-p port] [-n addr] [-f] [-L logfile] [-u authname [-P authpass]]\n' \
          '                   [-i pidfile]')
        sys.exit(1)
    laddr = None
    lport = None
    sdev = None
    authname = None
    authpass = None
    logfile = '/var/log/pel_collect.log'
    pidfile = None
    global_config = {'nh_addr':['192.168.0.102', 5060]}
    foreground = False
    for o, a in opts:
        if o == '-f':
            foreground = True
        elif o == '-l':
            laddr = a
        elif o == '-p':
            lport = int(a)
        elif o == '-L':
            logfile = a
        elif o == '-n':
            if a.startswith('['):
                parts = a.split(']', 1)
                global_config['nh_addr'] = [parts[0] + ']', 5060]
                parts = parts[1].split(':', 1)
            else:
                parts = a.split(':', 1)
                global_config['nh_addr'] = [parts[0], 5060]
            if len(parts) == 2:
                global_config['nh_addr'][1] = int(parts[1])
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
    if logfile == '-':
        lfile = sys.stdout
    else:
        lfile = open(logfile, 'a')

    if pidfile != None:
        open(pidfile, 'w').write('%d' % os.getpid())

    global_config['_sip_address'] = SipConf.my_address
    global_config['_sip_port'] = SipConf.my_port
    if laddr != None:
        global_config['_sip_address'] = laddr
    if lport != None:
        global_config['_sip_port'] = lport
    global_config['_sip_logger'] = SipLogger('pel_collect')
    #print(global_config)

    iuac = InfernUASConf()
    iuac.global_config = global_config
    iuac.authname = authname
    iuac.authpass = authpass
    iuac.cli = iuac.cld = authname
    iua = InfernUAS(iuac)
    #pio = PELIO(lfile)
    #if sdev != None:
    #    pio.sdev = sdev
    #pio.sstart_cb = pua.sess_started
    #pio.send_cb = pua.sess_ended
    #pio.start()
    ED2.loop()

