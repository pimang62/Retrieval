#!/bin/bash

start_port=49152
end_port=49162

for ((port=$start_port; port<=$end_port; port++)); do
  if ! ss -tuln | grep -q ":$port " && ! netstat -tuln | grep -q ":$port "; then
    echo "Port $port is available"
  fi
done
