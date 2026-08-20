#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 SQLite Cloud, Inc.
"""
range_server.py — a throwaway HTTP server for testing fetch_weights.sh.

The standard library's http.server answers 200 to a Range request and
sends the whole file, which is exactly the case the download script must
not silently accept. This serves a directory with real 206 responses, or
with Range support switched off, so both branches of the worker's resume
logic can be exercised without touching the network.

  python3 tests/range_server.py DIR PORT [--no-range] [--flaky N] [--port-file F]

--flaky N answers the first N GETs for each path with 503 and serves it
normally afterwards, which is what a transient CDN failure looks like from
curl's side. get_small used to lose a file to exactly this and report the
run as successful (#35), so the retry that replaced it needs something that
fails and then stops failing.

PORT 0 asks the kernel for a free one, which is what the caller should
use: a hardcoded port is a check that fails on any machine already running
something there, and it fails as "resume" rather than as "that port is
taken". --port-file is how the caller learns which port it got.
"""

import http.server
import os
import re
import socketserver
import sys


class Server(http.server.HTTPServer):
    """HTTPServer without the reverse DNS lookup in its constructor.

    HTTPServer.server_bind calls socket.getfqdn() on the address it just
    bound, and TCPServer.__init__ runs it *between* bind() and the listen()
    in server_activate. For 127.0.0.1 the answer is worth nothing — nothing
    here reads server_name — but on a macOS CI runner it blocked for about
    fifteen seconds, and until it returns the port is bound and not yet
    listening, so a caller connecting in that window is refused rather than
    queued. It also delayed the port file below past the caller's patience:
    the first server of a run failed to report in time, the second was fast
    because the resolver had cached the answer by then.
    """

    def server_bind(self):
        socketserver.TCPServer.server_bind(self)
        self.server_name, self.server_port = self.server_address[:2]


def make_handler(root, ranges, flaky):
    # Per-path GET counter. Single-threaded server, so no lock is needed.
    seen = {}

    class H(http.server.BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def _flaked(self):
            """True if this GET is one of the first `flaky` for its path."""
            if not flaky:
                return False
            n = seen.get(self.path, 0) + 1
            seen[self.path] = n
            if n <= flaky:
                self.send_error(503)
                return True
            return False

        def _path(self):
            # No traversal guard: this serves one temp dir to one test on
            # loopback and is killed straight after.
            return os.path.join(root, self.path.lstrip("/"))

        def do_HEAD(self):
            try:
                n = os.path.getsize(self._path())
            except OSError:
                return self.send_error(404)
            self.send_response(200)
            self.send_header("Content-Length", str(n))
            if ranges:
                self.send_header("Accept-Ranges", "bytes")
            self.end_headers()

        def do_GET(self):
            if self._flaked():
                return
            try:
                n = os.path.getsize(self._path())
            except OSError:
                return self.send_error(404)
            m = ranges and re.match(r"bytes=(\d+)-", self.headers.get("Range", ""))
            start = int(m.group(1)) if m else 0
            self.send_response(206 if m else 200)
            if m:
                self.send_header("Content-Range", f"bytes {start}-{n-1}/{n}")
            self.send_header("Content-Length", str(n - start))
            if ranges:
                self.send_header("Accept-Ranges", "bytes")
            self.end_headers()
            with open(self._path(), "rb") as f:
                f.seek(start)
                self.wfile.write(f.read())

    return H


def main():
    argv = sys.argv[1:]
    port_file = None
    if "--port-file" in argv:
        i = argv.index("--port-file")
        port_file = argv[i + 1]
        del argv[i:i + 2]
    flaky = 0
    if "--flaky" in argv:
        i = argv.index("--flaky")
        flaky = int(argv[i + 1])
        del argv[i:i + 2]
    ranges = "--no-range" not in argv
    root, port = [a for a in argv if not a.startswith("--")][:2]
    srv = Server(("127.0.0.1", int(port)), make_handler(root, ranges, flaky))

    # Hand the real port back only after the constructor has bound and
    # listened, so a caller that sees this file can connect immediately —
    # anything arriving before serve_forever() waits in the backlog rather
    # than being refused. Written under a temp name and renamed, so a
    # caller polling for the file never reads half a number.
    if port_file:
        with open(port_file + ".tmp", "w") as f:
            f.write(str(srv.server_address[1]))
        os.replace(port_file + ".tmp", port_file)

    srv.serve_forever()


if __name__ == "__main__":
    main()
