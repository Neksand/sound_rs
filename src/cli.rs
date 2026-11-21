//! The `soundrs` console command.
//!
//! Deliberately tiny: a hand-written argument parser with no dependency. The
//! point is that someone who just ran `pip install sound_rs` can type
//! `soundrs --help` and find the documentation without reading the source.

use pyo3::prelude::*;

use crate::audio;
use crate::player::Player;
use crate::writer;

pub(crate) const DOCS: &str = "https://neksand.github.io/sound_rs/";
const SOURCE: &str = "https://github.com/Neksand/sound_rs";
const VERSION: &str = env!("CARGO_PKG_VERSION");

const HELP: &str = "\
sound_rs — audio for Python

Usage
  soundrs <command> [options]
  soundrs play <file> [--device <name>] [--volume <0-4>]
  soundrs info <file> [--json]
  soundrs convert <input> <output.wav> [--subtype pcm16|pcm24|float32]
  soundrs devices
  soundrs docs
  soundrs mcp
  soundrs --version
  soundrs --help

Commands
  play     Play a file and wait until it ends.
           --device <name>   output device, see `soundrs devices`
           --volume <number> 0 to 4, default 1
  info     Sample rate, channels and duration.
           Reads the header when complete; decodes only when duration/format info is missing.
           --json            one line of JSON instead of the human output
  convert  Decode any supported file and write it as WAV.
           --subtype pcm16|pcm24|float32   sample format, default pcm16
  devices  Output devices, one per line.
  docs     Print the documentation link.
  mcp      Run the Model Context Protocol stdio server (audio_info, convert).

Links
  Documentation  {docs}
  Source         {source}
";

#[derive(Debug, PartialEq)]
pub(crate) enum Command {
    Help,
    Version,
    Docs,
    Devices,
    Mcp,
    Info {
        path: String,
        json: bool,
    },
    Convert {
        input: String,
        output: String,
        subtype: writer::Subtype,
    },
    Play {
        path: String,
        device: Option<String>,
        volume: f32,
    },
    Fail(String),
}

pub(crate) fn parse(args: &[String]) -> Command {
    let Some(first) = args.first().map(String::as_str) else {
        return Command::Help;
    };

    match first {
        "-h" | "--help" | "help" => Command::Help,
        "-V" | "--version" | "version" => Command::Version,
        "docs" => Command::Docs,
        "devices" => Command::Devices,
        "mcp" => Command::Mcp,
        "info" => {
            let Some(path) = args.get(1).filter(|p| !p.starts_with("--")) else {
                return Command::Fail("info needs a file path".to_string());
            };
            let mut json = false;
            for arg in &args[2..] {
                match arg.as_str() {
                    "--json" => json = true,
                    other => return Command::Fail(format!("unknown option '{other}'")),
                }
            }
            Command::Info {
                path: path.clone(),
                json,
            }
        }
        "convert" => {
            let Some(input) = args.get(1).filter(|p| !p.starts_with("--")) else {
                return Command::Fail("convert needs an input and an output path".to_string());
            };
            let Some(output) = args.get(2).filter(|p| !p.starts_with("--")) else {
                return Command::Fail("convert needs an input and an output path".to_string());
            };
            let mut subtype = writer::Subtype::Pcm16;
            let mut index = 3;
            while index < args.len() {
                match args[index].as_str() {
                    "--subtype" => match args.get(index + 1) {
                        Some(name) => match writer::Subtype::parse(Some(name)) {
                            Ok(parsed) => {
                                subtype = parsed;
                                index += 2;
                            }
                            Err(message) => return Command::Fail(message),
                        },
                        None => return Command::Fail("--subtype needs a value".to_string()),
                    },
                    other => return Command::Fail(format!("unknown option '{other}'")),
                }
            }
            Command::Convert {
                input: input.clone(),
                output: output.clone(),
                subtype,
            }
        }
        "play" => {
            let Some(path) = args.get(1).filter(|p| !p.starts_with("--")) else {
                return Command::Fail("play needs a file path".to_string());
            };
            let mut device = None;
            let mut volume = 1.0f32;
            let mut index = 2;
            while index < args.len() {
                match args[index].as_str() {
                    "--device" => match args.get(index + 1) {
                        Some(name) => {
                            device = Some(name.clone());
                            index += 2;
                        }
                        None => return Command::Fail("--device needs a name".to_string()),
                    },
                    "--volume" => match args.get(index + 1).map(|v| v.parse::<f32>()) {
                        Some(Ok(value)) if value.is_finite() => {
                            volume = value.clamp(0.0, 4.0);
                            index += 2;
                        }
                        _ => return Command::Fail("--volume needs a finite number".to_string()),
                    },
                    other => return Command::Fail(format!("unknown option '{other}'")),
                }
            }
            Command::Play {
                path: path.clone(),
                device,
                volume,
            }
        }
        other => Command::Fail(format!("unknown command '{other}'")),
    }
}

/// Entry point for the `soundrs` console script. The wrapper calls this with no
/// arguments, so the command line has to be read from `sys.argv`: under a
/// shebang the process arguments still carry the interpreter and script paths.
#[pyfunction]
pub(crate) fn cli(py: Python<'_>) -> PyResult<i32> {
    let argv: Vec<String> = py.import("sys")?.getattr("argv")?.extract()?;
    let args: Vec<String> = argv.into_iter().skip(1).collect();
    Ok(run(py, parse(&args)))
}

fn run(py: Python<'_>, command: Command) -> i32 {
    match command {
        Command::Help => {
            println!(
                "{}",
                HELP.replace("{docs}", DOCS).replace("{source}", SOURCE)
            );
            0
        }
        Command::Version => {
            println!("soundrs {VERSION}");
            0
        }
        Command::Docs => {
            println!("{DOCS}");
            0
        }
        Command::Mcp => crate::mcp::run(),
        Command::Devices => match crate::devices() {
            Ok(list) if list.is_empty() => {
                eprintln!("No output devices found.");
                1
            }
            Ok(list) => {
                for (index, name, is_default) in list {
                    println!("{index}\t{}{name}", if is_default { "* " } else { "  " });
                }
                0
            }
            Err(message) => fail(message),
        },
        Command::Info { path, json } => match py.detach(|| audio::info_or_decode(&path)) {
            Ok((rate, channels, seconds)) => {
                if json {
                    println!("{}", info_json(&path, rate, channels, seconds));
                } else {
                    println!("file         {path}");
                    println!("sample rate  {rate} Hz");
                    println!("channels     {channels}");
                    println!("duration     {} ({seconds:.3} s)", clock(seconds));
                }
                0
            }
            Err(message) => fail(message),
        },
        Command::Convert {
            input,
            output,
            subtype,
        } => match py.detach(|| audio::decode(&input)) {
            Ok(samples) => {
                let write = py.detach(|| {
                    writer::write(
                        &output,
                        &samples.samples,
                        samples.channels,
                        samples.sample_rate,
                        subtype,
                    )
                });
                match write {
                    Ok(()) => 0,
                    Err(err) => fail(err.to_string()),
                }
            }
            Err(message) => fail(message),
        },
        Command::Play {
            path,
            device,
            volume,
        } => {
            let mut player = match Player::new(device) {
                Ok(player) => player,
                Err(err) => return fail(err.to_string()),
            };
            if let Err(err) = player.set_volume(volume) {
                return fail(err.to_string());
            }
            if let Err(err) = player.play_file(py, path.into()) {
                return fail(err.to_string());
            }
            match player.wait(py) {
                Ok(()) => 0,
                Err(err) => fail(err.to_string()),
            }
        }
        Command::Fail(message) => {
            eprintln!("soundrs: {message}");
            eprintln!("Try `soundrs --help`.");
            2
        }
    }
}

fn fail(message: String) -> i32 {
    eprintln!("soundrs: {message}");
    1
}

/// One JSON line for `info --json`: hand-rolled so the CLI needs no serde.
fn info_json(path: &str, rate: u32, channels: u16, seconds: f32) -> String {
    // A non-finite duration would make the line invalid JSON.
    let seconds = if seconds.is_finite() { seconds } else { 0.0 };
    format!(
        "{{\"file\":{},\"sample_rate\":{rate},\"channels\":{channels},\"duration_seconds\":{seconds}}}",
        json_string(path)
    )
}

fn json_string(value: &str) -> String {
    let mut out = String::with_capacity(value.len() + 2);
    out.push('"');
    for c in value.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if c.is_control() => out.push_str(&format!("\\u{:04x}", c as u32)),
            c => out.push(c),
        }
    }
    out.push('"');
    out
}

/// 213.4 seconds -> "3:33.4".
fn clock(seconds: f32) -> String {
    // Round to tenths first so 59.96 becomes "1:00.0", not "0:60.0".
    let tenths = (seconds.max(0.0) * 10.0).round() as u64;
    let minutes = tenths / 600;
    let rest = (tenths % 600) as f32 / 10.0;
    format!("{minutes}:{rest:04.1}")
}

#[cfg(test)]
mod tests {
    use super::*;

    fn args(list: &[&str]) -> Vec<String> {
        list.iter().map(|s| s.to_string()).collect()
    }

    #[test]
    fn no_arguments_prints_help() {
        assert_eq!(parse(&args(&[])), Command::Help);
        assert_eq!(parse(&args(&["--help"])), Command::Help);
        assert_eq!(parse(&args(&["-h"])), Command::Help);
    }

    #[test]
    fn version_flags() {
        assert_eq!(parse(&args(&["--version"])), Command::Version);
        assert_eq!(parse(&args(&["-V"])), Command::Version);
    }

    #[test]
    fn mcp_command_parses() {
        assert_eq!(parse(&args(&["mcp"])), Command::Mcp);
    }

    #[test]
    fn play_parses_device_and_volume() {
        assert_eq!(
            parse(&args(&[
                "play", "a.flac", "--device", "Speakers", "--volume", "0.5"
            ])),
            Command::Play {
                path: "a.flac".to_string(),
                device: Some("Speakers".to_string()),
                volume: 0.5
            }
        );
        assert_eq!(
            parse(&args(&["play", "a.flac"])),
            Command::Play {
                path: "a.flac".to_string(),
                device: None,
                volume: 1.0
            }
        );
    }

    #[test]
    fn play_clamps_volume() {
        match parse(&args(&["play", "a.flac", "--volume", "9"])) {
            Command::Play { volume, .. } => assert_eq!(volume, 4.0),
            other => panic!("{other:?}"),
        }
    }

    #[test]
    fn bad_input_fails_with_a_message() {
        for bad in [
            args(&["play"]),
            args(&["play", "a.flac", "--volume"]),
            args(&["play", "a.flac", "--volume", "NaN"]),
            args(&["play", "a.flac", "--volume", "inf"]),
            args(&["play", "a.flac", "--volume", "-inf"]),
            args(&["play", "a.flac", "--nope"]),
            args(&["sing"]),
            args(&["info"]),
        ] {
            match parse(&bad) {
                Command::Fail(message) => assert!(!message.is_empty()),
                other => panic!("{bad:?} parsed as {other:?}"),
            }
        }
    }

    #[test]
    fn info_parses_optional_json() {
        assert_eq!(
            parse(&args(&["info", "a.flac"])),
            Command::Info {
                path: "a.flac".to_string(),
                json: false
            }
        );
        assert_eq!(
            parse(&args(&["info", "a.flac", "--json"])),
            Command::Info {
                path: "a.flac".to_string(),
                json: true
            }
        );
    }

    #[test]
    fn convert_parses_subtype() {
        assert_eq!(
            parse(&args(&["convert", "a.flac", "b.wav"])),
            Command::Convert {
                input: "a.flac".to_string(),
                output: "b.wav".to_string(),
                subtype: writer::Subtype::Pcm16
            }
        );
        assert_eq!(
            parse(&args(&[
                "convert",
                "a.flac",
                "b.wav",
                "--subtype",
                "float32"
            ])),
            Command::Convert {
                input: "a.flac".to_string(),
                output: "b.wav".to_string(),
                subtype: writer::Subtype::Float32
            }
        );
    }

    #[test]
    fn info_and_convert_reject_bad_arguments() {
        for bad in [
            args(&["info"]),
            args(&["info", "a.flac", "--json", "--nope"]),
            args(&["convert"]),
            args(&["convert", "a.flac"]),
            args(&["convert", "a.flac", "b.wav", "--subtype"]),
            args(&["convert", "a.flac", "b.wav", "--subtype", "mp3"]),
            args(&["convert", "a.flac", "b.wav", "--nope"]),
        ] {
            match parse(&bad) {
                Command::Fail(message) => assert!(!message.is_empty()),
                other => panic!("{bad:?} parsed as {other:?}"),
            }
        }
    }

    #[test]
    fn json_escapes_quotes_backslashes_and_controls() {
        assert_eq!(json_string("a\"b\\c"), r#""a\"b\\c""#);
        assert_eq!(json_string("a\nb\tc"), r#""a\nb\tc""#);
        assert_eq!(json_string("a\u{1}b"), r#""a\u0001b""#);
        assert!(!json_string("a\nb").contains('\n'));
    }

    #[test]
    fn info_json_is_one_line_with_numeric_values() {
        assert_eq!(
            info_json("a.flac", 48_000, 2, 1.5),
            r#"{"file":"a.flac","sample_rate":48000,"channels":2,"duration_seconds":1.5}"#
        );
        assert_eq!(
            info_json("x\".wav", 44_100, 1, f32::NAN),
            r#"{"file":"x\".wav","sample_rate":44100,"channels":1,"duration_seconds":0}"#
        );
    }

    #[test]
    fn clock_formats_minutes() {
        assert_eq!(clock(0.0), "0:00.0");
        assert_eq!(clock(213.4), "3:33.4");
        assert_eq!(clock(59.96), "1:00.0");
        assert_eq!(clock(-5.0), "0:00.0");
    }

    #[test]
    fn help_mentions_the_docs_url() {
        assert!(HELP.contains("{docs}"));
    }
}
