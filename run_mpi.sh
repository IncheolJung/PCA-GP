#!/bin/bash
unset DISPLAY
/home/incheol/Software/openmpi-4.1.4/bin/mpiexec --hostfile hosts.txt --mca btl_tcp_if_include tailscale0 --bind-to none --map-by node -x SSH_AUTH_SOCK --prefix /home/incheol/Software/openmpi-4.1.4/ -n 2 $1
