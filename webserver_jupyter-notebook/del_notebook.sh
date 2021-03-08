#!/bin/bash
ports=`jupyter notebook list | awk -F[:/] '{print $5}'`
for port in $ports
do
    PORT=$(echo $port | xargs)
    if [ "$PORT" != "8888" ]; then
        kill -9 $(lsof -n -i4TCP:$port | cut -f 2 -d " ")
    else
        echo "Notebooks on port '$PORT' are still running ..."
    fi
done
