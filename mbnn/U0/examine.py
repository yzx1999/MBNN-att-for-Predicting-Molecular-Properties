import os
import numpy as np
import time

def examine_one( ):
    nlines = []
    dirs = ['../../S2S4/U0/continue/lasp.out','lasp.out']
    for d in dirs:
    #for d in ['origin', 'charge_0.0', 'charge_0.2', 'charge_0.4']:
        nlines.append( int( os.popen('grep step {} -c'.format(d)).read().split()[0] ) - 1 )

    minlines = np.min( nlines ) -1

    for i,d in enumerate(dirs):
        if i == 0 :
            xline = minlines #+ 1015
        else:
            xline = minlines
        print( d )
        os.system(
            "grep ' {:4d} C = ' {}  -A 15".format( xline, d, )
            )

        print('-'*60)

    print('\n')

while True:
    examine_one()
    time.sleep( 10 )
