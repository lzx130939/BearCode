"""Disconnected clients and active streams must not outlive Web shutdown."""
from __future__ import annotations

import http.client
from http.server import BaseHTTPRequestHandler
import os
from pathlib import Path
import socket
import subprocess
import sys
import tempfile
import textwrap
import threading
import time
import unittest
from unittest.mock import Mock, patch

from agents import session
from agents.web import WebApp, WebError, WebHandler, WebServer


class WebDisconnectTests(unittest.TestCase):
    def handler(self):
        handler = WebHandler.__new__(WebHandler)
        handler.path = "/api/config"
        handler.close_connection = False
        handler.send_json = Mock()
        return handler

    def test_get_and_post_disconnects_do_not_attempt_an_error_response(self):
        for method in ("do_GET", "do_POST"):
            for error in (ConnectionAbortedError, ConnectionResetError, BrokenPipeError, TimeoutError):
                with self.subTest(method=method, error=error):
                    handler = self.handler()
                    handler.guard = Mock(side_effect=error("client disconnected"))
                    getattr(handler, method)()
                    self.assertTrue(handler.close_connection)
                    handler.send_json.assert_not_called()

    def test_disconnect_while_sending_a_rejection_does_not_escape_the_handler(self):
        for method in ("do_GET", "do_POST"):
            with self.subTest(method=method):
                handler = self.handler()
                handler.guard = Mock(side_effect=WebError("denied", 403))
                handler.send_json.side_effect = ConnectionAbortedError(10053, "client disconnected")
                with patch.object(BaseHTTPRequestHandler, "handle", lambda current: getattr(current, method)()):
                    handler.handle()
                self.assertTrue(handler.close_connection)
                handler.send_json.assert_called_once_with(403, {"error": "denied"})


class WebShutdownTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        patcher = patch.object(session, "SESSION_DIR", Path(temporary.name))
        patcher.start()
        self.addCleanup(patcher.stop)
        self.app = WebApp(options={"api_key": "offline", "custom_system_prompt": "test",
                                   "enable_mcp": False, "enable_online_evolution": False})
        self.addCleanup(self.app.close)

    def start_server(self):
        server = WebServer(("127.0.0.1", 0), self.app)
        self.addCleanup(server.server_close)
        server.handle_error = Mock()
        worker = threading.Thread(target=server.serve_forever, kwargs={"poll_interval": .01}, daemon=True)
        worker.start()
        return server, worker

    def connect(self, server, path):
        client = http.client.HTTPConnection("127.0.0.1", server.server_port, timeout=3)
        self.addCleanup(client.close)
        client.request("GET", path, headers={"Cookie": "bear_web=" + self.app.auth})
        response = client.getresponse()
        self.addCleanup(response.close)
        self.assertEqual(response.status, 200)
        return response

    def open_stream(self, server):
        state = self.app.call(self.app.dispatch("POST", "/api/sessions", {}))
        response = self.connect(server, "/api/events?session=" + state["id"])
        self.assertEqual(response.readline(), b"event: state\n")
        while True:
            line = response.readline()
            self.assertTrue(line, "Stream ended before the initial event was complete")
            if line == b"\n":
                break
        return self.app.conversations[state["id"]], response

    def test_app_close_wakes_and_joins_all_stream_threads_before_closing_the_loop(self):
        server, worker = self.start_server()
        with patch("agents.web.STREAM_HEARTBEAT_SECONDS", 60):
            views = [self.open_stream(server)[0] for _ in range(2)]
            handlers = list(server._threads)
            self.assertEqual(len(handlers), 2)
            started = time.monotonic()
            self.app.close()
            self.assertLess(time.monotonic() - started, 5)
        self.assertTrue(self.app.loop.is_closed())
        self.assertFalse(self.app.thread.is_alive())
        self.assertFalse(worker.is_alive())
        self.assertTrue(all(not handler.is_alive() for handler in handlers))
        self.assertTrue(all(view.stream_clients == 0 for view in views))
        self.assertFalse(server._requests)
        server.handle_error.assert_not_called()

    def test_server_close_drains_streams_and_idle_keep_alive_without_closing_the_app(self):
        server, worker = self.start_server()
        idle = self.connect(server, "/api/config")
        idle.read()
        with patch("agents.web.STREAM_HEARTBEAT_SECONDS", 60):
            view, response = self.open_stream(server)
            handlers = list(server._threads)
            started = time.monotonic()
            server.server_close()
            self.assertLess(time.monotonic() - started, 8)
        self.assertEqual(response.read(), b"")
        self.assertEqual(view.stream_clients, 0)
        self.assertTrue(all(not handler.is_alive() for handler in handlers))
        self.assertFalse(worker.is_alive())
        self.assertFalse(server._requests)
        self.assertFalse(self.app.stopping)
        self.assertTrue(self.app.loop.is_running())
        server.handle_error.assert_not_called()

    def test_server_close_interrupts_an_incomplete_post_body(self):
        server, worker = self.start_server()
        client = socket.create_connection(("127.0.0.1", server.server_port), timeout=3)
        self.addCleanup(client.close)
        accepted = threading.Event()
        original = WebHandler.guard

        def guard(handler, **kwargs):
            original(handler, **kwargs)
            accepted.set()

        request = (f"POST /api/sessions HTTP/1.1\r\nHost: 127.0.0.1:{server.server_port}\r\n"
                   f"Cookie: bear_web={self.app.auth}\r\nX-Bear-CSRF: {self.app.config['csrf']}\r\n"
                   "Content-Type: application/json\r\nContent-Length: 100\r\n\r\n{")
        with patch.object(WebHandler, "guard", guard):
            client.sendall(request.encode("ascii"))
            self.assertTrue(accepted.wait(timeout=3))
            started = time.monotonic()
            server.server_close()
            self.assertLess(time.monotonic() - started, 8)
        self.assertFalse(worker.is_alive())
        self.assertFalse(server._requests)
        self.assertFalse(self.app.conversations)
        server.handle_error.assert_not_called()

    def test_close_before_serving_and_repeated_close_are_safe(self):
        server = WebServer(("127.0.0.1", 0), self.app)
        self.addCleanup(server.server_close)
        server.server_close()
        server.server_close()
        worker = threading.Thread(target=server.serve_forever, daemon=True)
        worker.start()
        worker.join(timeout=2)
        self.assertFalse(worker.is_alive())
        self.app.close()
        self.app.close()

    def test_loopback_service_starts_and_serves_requests_without_reverse_dns(self):
        with patch("socket.getfqdn", side_effect=AssertionError("Reverse DNS must not be required")) as lookup:
            server, worker = self.start_server()
            self.assertEqual(server.server_name, "127.0.0.1")
            self.assertGreater(server.server_port, 0)
            response = self.connect(server, "/api/config")
            self.assertTrue(response.read())
            response.close()
            server.server_close()
            self.assertFalse(worker.is_alive())
            lookup.assert_not_called()

    def test_real_process_exits_cleanly_with_open_streams_and_cp1252_stderr(self):
        script = textwrap.dedent('''
            import faulthandler
            # Startup/import costs are separate from the shutdown deadline.
            faulthandler.dump_traceback_later(45, exit=True)
            import http.client
            from pathlib import Path
            import tempfile
            import threading
            from unittest.mock import patch
            from agents import session
            from agents.web import WebApp, WebServer
            print('imports ready', flush=True)

            with tempfile.TemporaryDirectory() as directory, patch.object(session, 'SESSION_DIR', Path(directory)):
                app = WebApp(options={'api_key': 'offline', 'custom_system_prompt': 'test'})
                server = WebServer(('127.0.0.1', 0), app)
                worker = threading.Thread(target=server.serve_forever, kwargs={'poll_interval': .01}, daemon=True)
                worker.start()
                clients, responses = [], []
                try:
                    for _ in range(2):
                        state = app.call(app.dispatch('POST', '/api/sessions', {}))
                        client = http.client.HTTPConnection('127.0.0.1', server.server_port, timeout=3)
                        clients.append(client)
                        client.request('GET', '/api/events?session=' + state['id'], headers={'Cookie': 'bear_web=' + app.auth})
                        response = client.getresponse()
                        responses.append(response)
                        assert response.readline() == b'event: state\\n'
                        while True:
                            line = response.readline()
                            assert line, 'stream ended before the initial event was complete'
                            if line == b'\\n':
                                break
                    handlers = list(server._threads)
                    print('streams ready', flush=True)
                    faulthandler.cancel_dump_traceback_later()
                    faulthandler.dump_traceback_later(10, exit=True)
                    app.close()
                    assert not worker.is_alive()
                    assert all(not handler.is_alive() for handler in handlers)
                    assert all(view.stream_clients == 0 for view in app.conversations.values())
                    print('clean shutdown', flush=True)
                finally:
                    for response in responses:
                        response.close()
                    for client in clients:
                        client.close()
                    server.server_close()
                    app.close()
                    faulthandler.cancel_dump_traceback_later()
        ''')
        environment = {**os.environ, "PYTHONUTF8": "0", "PYTHONIOENCODING": "cp1252"}
        for attempt in range(3):
            with self.subTest(attempt=attempt):
                try:
                    result = subprocess.run([sys.executable, "-B", "-X", "utf8=0", "-c", script],
                                            capture_output=True, timeout=60, env=environment)
                except subprocess.TimeoutExpired as exc:
                    output = (exc.stdout or b"").decode("cp1252", errors="replace")
                    errors = (exc.stderr or b"").decode("cp1252", errors="replace")
                    self.fail(f"Web subprocess timed out. Startup/shutdown progress:\n{output}\n{errors}")
                errors = result.stderr.decode("cp1252", errors="replace")
                progress = result.stdout.decode("cp1252", errors="replace")
                self.assertEqual(result.returncode, 0, f"Startup/shutdown progress:\n{progress}\n{errors}")
                self.assertIn(b"streams ready", result.stdout)
                self.assertIn(b"clean shutdown", result.stdout)
                self.assertNotIn("Traceback", errors)
                self.assertNotIn("Fatal Python error", errors)
