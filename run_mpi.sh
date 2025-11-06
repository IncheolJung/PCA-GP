#!/bin/bash
unset DISPLAY
$HOME/Software/openmpi-4.1.4/bin/mpiexec --hostfile hosts.txt --mca oob_tcp_if_include tailscale0 --mca btl_tcp_if_include tailscale0 --bind-to none --map-by node -x SSH_AUTH_SOCK --prefix /home/incheol/Software/openmpi-4.1.4/ -n 4 $1
