#!/bin/bash
if [ -n "$1" ]; then
    filename="$HOME/recording/$(date +"%Y_%m_%d_%H_%M").log"
    ssh $1 "date --set=\"@$(date +%s)\""
    ssh $1 < print_temps.sh >> $filename
fi