//! The `soundrs` console command.
//!
//! Deliberately tiny: a hand-written argument parser with no dependency. The
//! point is that someone who just ran `pip install sound_rs` can type
//! `soundrs --help` and find the documentation without reading the source.

use pyo3::prelude::*;

use crate::player::Player;
use crate::{audio, error};

pub(crate) const DOCS: &str = "https://neksand.github.io/sound_rs/";
const SOURCE: &str = "https://github.com/Neksand/sound_rs";
const VERSION: &str = env!("CARGO_PKG_VERSION");

const HELP: &str = "\
sound_rs — audio for Python

Usage
  soundrs <command> [options]
  soundrs play <file> [--device <name>] [--volume <0-4>]
  soundrs info <file>
  soundrs devices
  soundrs docs
  soundrs --version
  soundrs --help

Commands
  play     Play a file and wait until it ends.
           --device <name>   output device, see `soundrs devices`
           --volume <number> 0 to 4, default 1
  info     Sample rate, channels and duration, read from the header only.
  devices  Output devices, one per line.
  docs     Print the documentation link.

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
    Info(String),
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
        "info" => match args.get(1) {
            Some(path) => Command::Info(path.clone()),
            None => Command::Fail("info needs a file path".to_string()),
        },
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
                        Some(Ok(value)) => {
                            volume = value.clamp(0.0, 4.0);
                            index += 2;
                        }
                        _ => return Command::Fail("--volume needs a number".to_string()),
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
        Command::Info(path) => match audio::info_or_decode(&path) {
            Ok((rate, channels, seconds)) => {
                println!("file         {path}");
                println!("sample rate  {rate} Hz");
                println!("channels     {channels}");
                println!("duration     {} ({seconds:.3} s)", clock(seconds));
                0
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
            if let Err(err) = player.play_file(py, &path) {
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

/// 213.4 seconds -> "3:33.4".
fn clock(seconds: f32) -> String {
    // Round to tenths first so 59.96 becomes "1:00.0", not "0:60.0".
    let tenths = (seconds.max(0.0) * 10.0).round() as u64;
    let minutes = tenths / 600;
    let rest = (tenths % 600) as f32 / 10.0;
    format!("{minutes}:{rest:04.1}")
}

/// Keeps the `error` import used when the module is built without the CLI.
#[allow(dead_code)]
fn unused() -> PyErr {
    error(String::new())
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
