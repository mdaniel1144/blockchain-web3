#!/bin/bash

# Start server and redirect its output to a log file
python server/dp_server.py > server.log 2>&1 &
SERVER_PID=$!

# Wait for server to start
sleep 5

# Start clients
NUM_CLIENTS=${1:-4}  # Default to 4 clients if not specified

for i in $(seq 1 $NUM_CLIENTS)
do
    # Redirect each client's output to a separate log file
    python client/dp_client.py > "client_$i.log" 2>&1 &
    sleep 2
done

# Wait for all background processes to finish
wait

# Kill the server process
kill $SERVER_PID
