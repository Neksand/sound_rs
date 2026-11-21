//! Minimal Model Context Protocol server over stdio.
//!
//! Exposes metadata and WAV conversion as MCP tools. No audio device is opened,
//! so it is safe in headless containers. One JSON-RPC message per line.

use serde_json::{json, Value};
use std::io::{BufRead, Write};

const PROTOCOL_VERSION: &str = "2024-11-05";
const SERVER_NAME: &str = "sound_rs";

pub(crate) fn run() -> i32 {
    let stdin = std::io::stdin();
    let mut stdout = std::io::stdout().lock();
    for line in stdin.lock().lines() {
        let line = match line {
            Ok(line) => line,
            Err(error) => {
                eprintln!("soundrs mcp: {error}");
                return 1;
            }
        };
        if line.trim().is_empty() {
            continue;
        }
        if let Some(response) = handle_line(&line) {
            if serde_json::to_writer(&mut stdout, &response).is_err() {
                return 1;
            }
            if stdout.write_all(b"\n").is_err() || stdout.flush().is_err() {
                return 1;
            }
        }
    }
    0
}

fn handle_line(line: &str) -> Option<Value> {
    let request: Value = match serde_json::from_str(line) {
        Ok(request) => request,
        Err(error) => {
            return Some(error_response(
                Value::Null,
                -32700,
                &format!("Parse error: {error}"),
            ))
        }
    };
    let id = request.get("id").cloned();
    let method = request.get("method").and_then(Value::as_str).unwrap_or("");
    match method {
        "initialize" => {
            let requested = request
                .pointer("/params/protocolVersion")
                .and_then(Value::as_str)
                .unwrap_or(PROTOCOL_VERSION);
            Some(result_response(
                id,
                json!({
                    "protocolVersion": requested,
                    "capabilities": {"tools": {}},
                    "serverInfo": {"name": SERVER_NAME, "version": env!("CARGO_PKG_VERSION")},
                    "instructions": "Use audio_info for metadata and convert to write WAV files. Paths are local; no audio device is opened."
                }),
            ))
        }
        "notifications/initialized" | "notifications/cancelled" | "exit" => None,
        "ping" => Some(result_response(id, json!({}))),
        "shutdown" => Some(result_response(id, Value::Null)),
        "tools/list" => Some(result_response(id, json!({"tools": tools()}))),
        "tools/call" => Some(call_tool(id, &request)),
        _ => {
            if id.is_some() {
                Some(error_response(
                    id.unwrap_or(Value::Null),
                    -32601,
                    &format!("Method not found: {method}"),
                ))
            } else {
                None
            }
        }
    }
}

fn tools() -> Value {
    json!([
        {
            "name": "audio_info",
            "description": "Read sample rate, channel count and duration from an audio file without decoding the whole file unless the header lacks a duration.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Local audio file path"}
                },
                "required": ["path"],
                "additionalProperties": false
            }
        },
        {
            "name": "convert",
            "description": "Decode any supported audio file and write a WAV file.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "input": {"type": "string", "description": "Local input path"},
                    "output": {"type": "string", "description": "Destination .wav path"},
                    "subtype": {
                        "type": "string",
                        "enum": ["pcm16", "pcm24", "float32"],
                        "default": "pcm16"
                    }
                },
                "required": ["input", "output"],
                "additionalProperties": false
            }
        }
    ])
}

fn call_tool(id: Option<Value>, request: &Value) -> Value {
    let name = request
        .pointer("/params/name")
        .and_then(Value::as_str)
        .unwrap_or("");
    let args = request
        .pointer("/params/arguments")
        .cloned()
        .unwrap_or_else(|| json!({}));
    match name {
        "audio_info" => {
            let Some(path) = args.get("path").and_then(Value::as_str) else {
                return tool_error(id, "audio_info requires a string 'path'");
            };
            match crate::audio::info_or_decode(path) {
                Ok((rate, channels, duration)) => tool_text(
                    id,
                    &json!({
                        "file": path,
                        "sample_rate": rate,
                        "channels": channels,
                        "duration_seconds": duration
                    })
                    .to_string(),
                ),
                Err(message) => tool_error(id, &message),
            }
        }
        "convert" => {
            let (Some(input), Some(output)) = (
                args.get("input").and_then(Value::as_str),
                args.get("output").and_then(Value::as_str),
            ) else {
                return tool_error(id, "convert requires string 'input' and 'output'");
            };
            let subtype = args
                .get("subtype")
                .and_then(Value::as_str)
                .unwrap_or("pcm16");
            let format = match crate::writer::Subtype::parse(Some(subtype)) {
                Ok(format) => format,
                Err(message) => return tool_error(id, &message),
            };
            let audio = match crate::audio::decode(input) {
                Ok(audio) => audio,
                Err(message) => return tool_error(id, &message),
            };
            match crate::writer::write(
                output,
                &audio.samples,
                audio.channels,
                audio.sample_rate,
                format,
            ) {
                Ok(()) => {
                    let bytes = std::fs::metadata(output)
                        .map(|meta| meta.len())
                        .unwrap_or(0);
                    tool_text(
                        id,
                        &json!({
                            "input": input,
                            "output": output,
                            "subtype": subtype,
                            "sample_rate": audio.sample_rate,
                            "channels": audio.channels,
                            "frames": audio.samples.len() / audio.channels.max(1) as usize,
                            "bytes": bytes
                        })
                        .to_string(),
                    )
                }
                Err(error) => tool_error(id, &error.to_string()),
            }
        }
        other => tool_error(id, &format!("Unknown tool '{other}'")),
    }
}

fn result_response(id: Option<Value>, result: Value) -> Value {
    json!({"jsonrpc": "2.0", "id": id.unwrap_or(Value::Null), "result": result})
}

fn error_response(id: Value, code: i64, message: &str) -> Value {
    json!({
        "jsonrpc": "2.0",
        "id": id,
        "error": {"code": code, "message": message}
    })
}

fn tool_text(id: Option<Value>, text: &str) -> Value {
    result_response(
        id,
        json!({"content": [{"type": "text", "text": text}], "isError": false}),
    )
}

fn tool_error(id: Option<Value>, message: &str) -> Value {
    result_response(
        id,
        json!({"content": [{"type": "text", "text": message}], "isError": true}),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn initialize_reports_tools() {
        let response = handle_line(r#"{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":"2025-06-18"}}"#)
            .unwrap();
        assert_eq!(response["result"]["protocolVersion"], "2025-06-18");
        assert_eq!(response["result"]["capabilities"]["tools"], json!({}));
        assert_eq!(response["result"]["serverInfo"]["name"], "sound_rs");
    }

    #[test]
    fn tools_list_contains_info_and_convert() {
        let response = handle_line(r#"{"jsonrpc":"2.0","id":2,"method":"tools/list"}"#).unwrap();
        let names: Vec<_> = response["result"]["tools"]
            .as_array()
            .unwrap()
            .iter()
            .map(|tool| tool["name"].as_str().unwrap())
            .collect();
        assert_eq!(names, ["audio_info", "convert"]);
    }

    #[test]
    fn audio_info_reads_the_fixture() {
        let path = format!("{}/tests/fixtures/tone.wav", env!("CARGO_MANIFEST_DIR"));
        let request = json!({
            "jsonrpc": "2.0", "id": 3, "method": "tools/call",
            "params": {"name": "audio_info", "arguments": {"path": path}}
        });
        let response = handle_line(&request.to_string()).unwrap();
        assert_eq!(response["result"]["isError"], false);
        let text = response["result"]["content"][0]["text"].as_str().unwrap();
        let info: Value = serde_json::from_str(text).unwrap();
        assert_eq!(info["sample_rate"], 44100);
        assert_eq!(info["channels"], 2);
    }

    #[test]
    fn convert_writes_a_wav() {
        let directory = std::env::temp_dir().join(format!("sound_rs_mcp_{}", std::process::id()));
        std::fs::create_dir_all(&directory).unwrap();
        let input = format!("{}/tests/fixtures/tone.mp3", env!("CARGO_MANIFEST_DIR"));
        let output = directory.join("out.wav");
        let request = json!({
            "jsonrpc": "2.0", "id": 4, "method": "tools/call",
            "params": {"name": "convert", "arguments": {"input": input, "output": output, "subtype": "pcm24"}}
        });
        let response = handle_line(&request.to_string()).unwrap();
        assert_eq!(response["result"]["isError"], false);
        assert!(output.is_file());
        let info = crate::audio::info_or_decode(output.to_str().unwrap()).unwrap();
        assert_eq!((info.0, info.1), (44100, 2));
        std::fs::remove_dir_all(&directory).ok();
    }

    #[test]
    fn unknown_tool_is_a_tool_error() {
        let response = handle_line(
            r#"{"jsonrpc":"2.0","id":5,"method":"tools/call","params":{"name":"nope","arguments":{}}}"#,
        )
        .unwrap();
        assert_eq!(response["result"]["isError"], true);
    }

    #[test]
    fn notifications_have_no_response() {
        assert!(handle_line(r#"{"jsonrpc":"2.0","method":"notifications/initialized"}"#).is_none());
    }
}
