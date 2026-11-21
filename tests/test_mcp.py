"""MCP stdio regression: initialize, tools/list and tools/call."""

import json
import pathlib
import subprocess
import sys
import tempfile
import unittest

FIXTURES = pathlib.Path(__file__).parent / "fixtures"
CLI = "import sound_rs, sys; sys.exit(sound_rs.cli())"


def run_mcp(requests):
    payload = "".join(json.dumps(request) + "\n" for request in requests)
    return subprocess.run(
        [sys.executable, "-c", CLI, "mcp"],
        input=payload,
        capture_output=True,
        text=True,
        timeout=60,
    )


class McpTests(unittest.TestCase):
    def test_initialize_and_list_tools(self):
        result = run_mcp(
            [
                {
                    "jsonrpc": "2.0",
                    "id": 1,
                    "method": "initialize",
                    "params": {"protocolVersion": "2025-06-18"},
                },
                {"jsonrpc": "2.0", "method": "notifications/initialized"},
                {"jsonrpc": "2.0", "id": 2, "method": "tools/list"},
            ]
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        responses = [json.loads(line) for line in result.stdout.splitlines()]
        self.assertEqual(responses[0]["result"]["serverInfo"]["name"], "sound_rs")
        self.assertEqual(
            [tool["name"] for tool in responses[1]["result"]["tools"]],
            ["audio_info", "convert"],
        )

    def test_tools_call_info_and_convert(self):
        with tempfile.TemporaryDirectory() as directory:
            output = pathlib.Path(directory) / "out.wav"
            result = run_mcp(
                [
                    {
                        "jsonrpc": "2.0",
                        "id": 1,
                        "method": "tools/call",
                        "params": {
                            "name": "audio_info",
                            "arguments": {"path": str(FIXTURES / "tone.wav")},
                        },
                    },
                    {
                        "jsonrpc": "2.0",
                        "id": 2,
                        "method": "tools/call",
                        "params": {
                            "name": "convert",
                            "arguments": {
                                "input": str(FIXTURES / "tone.mp3"),
                                "output": str(output),
                                "subtype": "pcm24",
                            },
                        },
                    },
                ]
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            responses = [json.loads(line) for line in result.stdout.splitlines()]
            info = json.loads(responses[0]["result"]["content"][0]["text"])
            self.assertEqual((info["sample_rate"], info["channels"]), (44100, 2))
            converted = json.loads(responses[1]["result"]["content"][0]["text"])
            self.assertTrue(output.is_file())
            self.assertGreater(converted["bytes"], 0)


if __name__ == "__main__":
    unittest.main()
