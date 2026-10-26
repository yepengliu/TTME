# shellcheck shell=bash
# Helpers shared by the scripts that start vLLM servers; source this file, do not run it.
# Ports are never fixed: the kernel picks free ones, and a server's URL is used only once the port is
# its own, because vLLM binds with SO_REUSEPORT and a second server of the same user could otherwise
# share the port and answer in its place.

# Print $1 distinct TCP ports that are free now (all bound to port 0 at once, then released).
free_ports() {
  "${PYTHON:-python3}" -c '
import socket, sys
socks = [socket.socket() for _ in range(int(sys.argv[1]))]
for s in socks:
    s.bind(("", 0))
print(" ".join(str(s.getsockname()[1]) for s in socks))' "$1"
}

# Inodes of the TCP sockets (IPv4 and IPv6) listening on port $1; state 0A is LISTEN.
listen_inodes() {
  awk -v hex="$(printf '%04X' "$1")" \
    '$4 == "0A" { n = split($2, a, ":"); if (a[n] == hex) print $10 }' \
    /proc/net/tcp /proc/net/tcp6 2> /dev/null || true
}

# PIDs of process $1 and all of its descendants.
tree_pids() {
  ps -e -o pid= -o ppid= | awk -v root="$1" '
    { parent[$1] = $2 }
    END {
      keep[root] = 1
      do {
        added = 0
        for (p in parent) if (!(p in keep) && (parent[p] in keep)) { keep[p] = 1; added = 1 }
      } while (added)
      for (p in keep) print p
    }'
}

# Succeed when port $2 has a listener and every listener belongs to process $1 or its descendants.
owns_port() {
  local listening held inode p
  listening=$(listen_inodes "$2")
  [[ -n $listening ]] || return 1
  held=$(for p in $(tree_pids "$1"); do readlink /proc/"$p"/fd/* 2> /dev/null; done |
    sed -n 's/^socket:\[\([0-9]*\)\]$/\1/p' || true)
  for inode in $listening; do
    grep -qx "$inode" <<< "$held" || return 1
  done
}

# One probe of the vLLM server with PID $1 on local port $2. Prints `ready`, `starting`, `exited`,
# or `foreign` when the port answers but another process also listens on it.
probe_server() {
  if ! kill -0 "$1" 2> /dev/null; then
    echo exited
  elif ! curl -sf -o /dev/null --max-time 5 --noproxy '*' "http://127.0.0.1:$2/health"; then
    echo starting
  elif [[ ! -r /proc/net/tcp ]] || owns_port "$1" "$2"; then
    echo ready
  else
    echo foreign
  fi
}

# Write the line $2 to file $1 through a temporary file, so readers never see a partial file.
write_atomic() {
  local tmp
  tmp=$(mktemp "$(dirname "$1")/.$(basename "$1").XXXXXX")
  printf '%s\n' "$2" > "$tmp"
  mv -f "$tmp" "$1"
}
