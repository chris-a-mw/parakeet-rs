/*
Push-to-talk microphone transcription with continuous listening

Types transcriptions directly into the active window.

Usage:
    cargo run --release --example mic_transcribe --features cuda -- ../parakeet-tdt-0.6b-v3-onnx

Config file (~/.config/mic_transcribe.conf):
    hotkey = LAlt+Grave
    mode = hold    # hold = press-to-talk, toggle = press to start/stop
    # or: hotkey = F12
    # or: hotkey = RControl+Space

Controls:
    Configured hotkey - Toggle listening on/off (works globally, any window)
    Ctrl+C in terminal - Exit

While listening:
    - Speak naturally, pause between sentences
    - Each pause (~500ms) triggers transcription of that segment
    - Transcription is typed into the active window
    - Keeps listening for more speech until you press hotkey again

Supported keys: F1-F20, Grave, Home, End, Insert, Delete, PageUp, PageDown,
                Space, Tab, Escape, Enter, Backspace, A-Z, 0-9
Supported modifiers: LAlt, RAlt, LControl, RControl, LShift, RShift, LMeta, RMeta

Requirements:
    - TDT v3 ONNX model directory
    - Working microphone
    - Linux: X11 + libxdo (for keyboard simulation)
*/

use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use device_query::{DeviceQuery, DeviceState, Keycode};
use enigo::{Enigo, Keyboard, Settings};
use parakeet_rs::{ParakeetTDT, Transcriber};
use std::env;
use std::fs;
use std::io::Write;
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{mpsc, Arc, Mutex};
use std::thread;
use std::time::{Duration, Instant};

const SAMPLE_RATE: u32 = 16000;
const CHANNELS: u16 = 1;

// VAD configuration
const RMS_THRESHOLD: f32 = 0.01;
const SILENCE_DURATION_MS: u64 = 500;
const MIN_SPEECH_SAMPLES: usize = 8000; // ~0.5s minimum

struct RmsVad {
    threshold: f32,
    silence_start: Option<Instant>,
    silence_duration: Duration,
    speech_detected: bool,
}

impl RmsVad {
    fn new(threshold: f32, silence_duration_ms: u64) -> Self {
        Self {
            threshold,
            silence_start: None,
            silence_duration: Duration::from_millis(silence_duration_ms),
            speech_detected: false,
        }
    }

    /// Returns true if end-of-speech detected (silence after speech)
    fn process(&mut self, samples: &[f32]) -> bool {
        let rms = calculate_rms(samples);

        if rms >= self.threshold {
            self.speech_detected = true;
            self.silence_start = None;
            false
        } else if self.speech_detected {
            match self.silence_start {
                None => {
                    self.silence_start = Some(Instant::now());
                    false
                }
                Some(start) => {
                    if start.elapsed() >= self.silence_duration {
                        // End of speech - reset for next utterance
                        self.speech_detected = false;
                        self.silence_start = None;
                        true
                    } else {
                        false
                    }
                }
            }
        } else {
            false
        }
    }

    fn reset(&mut self) {
        self.silence_start = None;
        self.speech_detected = false;
    }
}

fn calculate_rms(samples: &[f32]) -> f32 {
    if samples.is_empty() {
        return 0.0;
    }
    let sum_squares: f32 = samples.iter().map(|s| s * s).sum();
    (sum_squares / samples.len() as f32).sqrt()
}

fn resample(samples: &[f32], from_rate: u32, to_rate: u32) -> Vec<f32> {
    if samples.is_empty() {
        return Vec::new();
    }
    if from_rate == to_rate {
        return samples.to_vec();
    }

    let ratio = from_rate as f64 / to_rate as f64;
    let output_len = (samples.len() as f64 / ratio).ceil() as usize;
    let mut output = Vec::with_capacity(output_len);

    for i in 0..output_len {
        let src_idx = i as f64 * ratio;
        let idx0 = src_idx.floor() as usize;
        let idx1 = (idx0 + 1).min(samples.len() - 1);
        let frac = src_idx - idx0 as f64;

        let sample = samples[idx0] * (1.0 - frac as f32) + samples[idx1] * frac as f32;
        output.push(sample);
    }

    output
}

fn to_mono(samples: &[f32], channels: u16) -> Vec<f32> {
    if channels == 1 {
        return samples.to_vec();
    }

    samples
        .chunks(channels as usize)
        .map(|chunk| chunk.iter().sum::<f32>() / channels as f32)
        .collect()
}

/// Push-to-talk mode
#[derive(Clone, Copy, PartialEq)]
enum PttMode {
    Toggle, // Press to start, press again to stop
    Hold,   // Hold to record, release to stop
}

/// Configuration loaded from file
#[derive(Clone)]
struct Config {
    hotkey: HotkeyConfig,
    mode: PttMode,
}

/// Hotkey configuration with optional modifiers
#[derive(Clone)]
struct HotkeyConfig {
    key: Keycode,
    modifiers: Vec<Keycode>,
    display_name: String,
}

impl HotkeyConfig {
    fn is_pressed(&self, keys: &[Keycode]) -> bool {
        // Check main key is pressed
        if !keys.contains(&self.key) {
            return false;
        }
        // Check all modifiers are pressed
        for modifier in &self.modifiers {
            if !keys.contains(modifier) {
                return false;
            }
        }
        true
    }
}

fn parse_keycode(key_name: &str) -> Option<Keycode> {
    match key_name.to_lowercase().as_str() {
        // Function keys
        "f1" => Some(Keycode::F1),
        "f2" => Some(Keycode::F2),
        "f3" => Some(Keycode::F3),
        "f4" => Some(Keycode::F4),
        "f5" => Some(Keycode::F5),
        "f6" => Some(Keycode::F6),
        "f7" => Some(Keycode::F7),
        "f8" => Some(Keycode::F8),
        "f9" => Some(Keycode::F9),
        "f10" => Some(Keycode::F10),
        "f11" => Some(Keycode::F11),
        "f12" => Some(Keycode::F12),
        "f13" => Some(Keycode::F13),
        "f14" => Some(Keycode::F14),
        "f15" => Some(Keycode::F15),
        "f16" => Some(Keycode::F16),
        "f17" => Some(Keycode::F17),
        "f18" => Some(Keycode::F18),
        "f19" => Some(Keycode::F19),
        "f20" => Some(Keycode::F20),
        // Navigation
        "insert" => Some(Keycode::Insert),
        "delete" => Some(Keycode::Delete),
        "home" => Some(Keycode::Home),
        "end" => Some(Keycode::End),
        "pageup" => Some(Keycode::PageUp),
        "pagedown" => Some(Keycode::PageDown),
        // Special
        "grave" | "`" | "backtick" => Some(Keycode::Grave),
        "space" => Some(Keycode::Space),
        "tab" => Some(Keycode::Tab),
        "escape" | "esc" => Some(Keycode::Escape),
        "enter" | "return" => Some(Keycode::Enter),
        "backspace" => Some(Keycode::Backspace),
        "capslock" => Some(Keycode::CapsLock),
        // Modifiers (can also be used as main key)
        "lalt" | "leftalt" => Some(Keycode::LAlt),
        "ralt" | "rightalt" => Some(Keycode::RAlt),
        "lcontrol" | "lctrl" | "leftcontrol" => Some(Keycode::LControl),
        "rcontrol" | "rctrl" | "rightcontrol" => Some(Keycode::RControl),
        "lshift" | "leftshift" => Some(Keycode::LShift),
        "rshift" | "rightshift" => Some(Keycode::RShift),
        "lmeta" | "lsuper" | "leftmeta" | "leftsuper" => Some(Keycode::LMeta),
        "rmeta" | "rsuper" | "rightmeta" | "rightsuper" => Some(Keycode::RMeta),
        // Letters
        "a" => Some(Keycode::A),
        "b" => Some(Keycode::B),
        "c" => Some(Keycode::C),
        "d" => Some(Keycode::D),
        "e" => Some(Keycode::E),
        "f" => Some(Keycode::F),
        "g" => Some(Keycode::G),
        "h" => Some(Keycode::H),
        "i" => Some(Keycode::I),
        "j" => Some(Keycode::J),
        "k" => Some(Keycode::K),
        "l" => Some(Keycode::L),
        "m" => Some(Keycode::M),
        "n" => Some(Keycode::N),
        "o" => Some(Keycode::O),
        "p" => Some(Keycode::P),
        "q" => Some(Keycode::Q),
        "r" => Some(Keycode::R),
        "s" => Some(Keycode::S),
        "t" => Some(Keycode::T),
        "u" => Some(Keycode::U),
        "v" => Some(Keycode::V),
        "w" => Some(Keycode::W),
        "x" => Some(Keycode::X),
        "y" => Some(Keycode::Y),
        "z" => Some(Keycode::Z),
        // Numbers
        "0" | "key0" => Some(Keycode::Key0),
        "1" | "key1" => Some(Keycode::Key1),
        "2" | "key2" => Some(Keycode::Key2),
        "3" | "key3" => Some(Keycode::Key3),
        "4" | "key4" => Some(Keycode::Key4),
        "5" | "key5" => Some(Keycode::Key5),
        "6" | "key6" => Some(Keycode::Key6),
        "7" | "key7" => Some(Keycode::Key7),
        "8" | "key8" => Some(Keycode::Key8),
        "9" | "key9" => Some(Keycode::Key9),
        _ => None,
    }
}

fn is_modifier(keycode: &Keycode) -> bool {
    matches!(
        keycode,
        Keycode::LAlt
            | Keycode::RAlt
            | Keycode::LControl
            | Keycode::RControl
            | Keycode::LShift
            | Keycode::RShift
            | Keycode::LMeta
            | Keycode::RMeta
    )
}

fn parse_hotkey_combo(combo: &str) -> Option<HotkeyConfig> {
    let parts: Vec<&str> = combo.split('+').map(|s| s.trim()).collect();
    if parts.is_empty() {
        return None;
    }

    let mut modifiers = Vec::new();
    let mut main_key = None;

    for part in &parts {
        if let Some(keycode) = parse_keycode(part) {
            if is_modifier(&keycode) && parts.len() > 1 {
                modifiers.push(keycode);
            } else {
                main_key = Some(keycode);
            }
        } else {
            eprintln!("Warning: Unknown key '{}' in hotkey combo", part);
            return None;
        }
    }

    main_key.map(|key| HotkeyConfig {
        key,
        modifiers,
        display_name: combo.to_string(),
    })
}

fn get_config_path() -> PathBuf {
    if let Ok(xdg_config) = env::var("XDG_CONFIG_HOME") {
        PathBuf::from(xdg_config).join("mic_transcribe.conf")
    } else if let Ok(home) = env::var("HOME") {
        PathBuf::from(home).join(".config/mic_transcribe.conf")
    } else {
        PathBuf::from("mic_transcribe.conf")
    }
}

fn load_config() -> Option<Config> {
    let config_path = get_config_path();

    if !config_path.exists() {
        return None;
    }

    let content = fs::read_to_string(&config_path).ok()?;

    let mut hotkey: Option<HotkeyConfig> = None;
    let mut mode = PttMode::Toggle; // Default to toggle

    for line in content.lines() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }

        if let Some((key, value)) = line.split_once('=') {
            let key = key.trim().to_lowercase();
            let value = value.trim();

            match key.as_str() {
                "hotkey" => {
                    hotkey = parse_hotkey_combo(value);
                }
                "mode" => {
                    mode = match value.to_lowercase().as_str() {
                        "hold" | "ptt" | "push" | "press" => PttMode::Hold,
                        "toggle" | "switch" => PttMode::Toggle,
                        _ => {
                            eprintln!("Warning: Unknown mode '{}', using 'toggle'", value);
                            PttMode::Toggle
                        }
                    };
                }
                _ => {}
            }
        }
    }

    hotkey.map(|hk| Config { hotkey: hk, mode })
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = env::args().collect();

    // Model path from args
    let model_path = args.get(1).map(|s| s.as_str()).unwrap_or("./tdt");

    // Load config from file, with fallback
    let config_path = get_config_path();
    let config = match load_config() {
        Some(cfg) => {
            println!("Loaded config from: {}", config_path.display());
            cfg
        }
        None => {
            if config_path.exists() {
                eprintln!("Warning: Could not parse config file, using defaults");
            } else {
                println!("No config file found at: {}", config_path.display());
                println!("Create it with:");
                println!("  echo 'hotkey = LAlt+R' > {}", config_path.display());
                println!("  echo 'mode = hold' >> {}", config_path.display());
            }
            // Default config
            Config {
                hotkey: HotkeyConfig {
                    key: Keycode::Grave,
                    modifiers: vec![],
                    display_name: "Grave (`)".to_string(),
                },
                mode: PttMode::Toggle,
            }
        }
    };

    let hotkey_config = &config.hotkey;
    let ptt_mode = config.mode;
    let mode_str = match ptt_mode {
        PttMode::Hold => "hold-to-talk",
        PttMode::Toggle => "toggle",
    };
    println!("Mode: {}", mode_str);

    println!("Loading TDT model from: {}", model_path);
    let model = Arc::new(Mutex::new(ParakeetTDT::from_pretrained(model_path, None)?));
    println!("Model loaded.");

    // Channel for transcription results
    let (tx, rx) = mpsc::channel::<String>();

    // Setup audio capture
    let host = cpal::default_host();
    let device = host
        .default_input_device()
        .ok_or("No input device available")?;

    println!("Using input device: {}", device.name()?);

    let stream_config = device.default_input_config()?;
    let device_sample_rate = stream_config.sample_rate().0;
    let device_channels = stream_config.channels();

    println!(
        "Device config: {}Hz, {} channel(s)",
        device_sample_rate, device_channels
    );

    // Shared state
    let running = Arc::new(AtomicBool::new(true));
    let listening = Arc::new(AtomicBool::new(false));
    let transcribing = Arc::new(AtomicBool::new(false)); // Prevent transcription pile-up
    let audio_buffer: Arc<Mutex<Vec<f32>>> = Arc::new(Mutex::new(Vec::new()));

    // Ctrl+C handler
    {
        let running = running.clone();
        ctrlc::set_handler(move || {
            println!("\nExiting...");
            running.store(false, Ordering::SeqCst);
        })?;
    }

    // Audio callback - accumulates samples when listening
    // Buffer limit: ~30 seconds at 16kHz = 480,000 samples
    const MAX_BUFFER_SAMPLES: usize = 480_000;
    let listening_clone = listening.clone();
    let audio_buffer_clone = audio_buffer.clone();
    let stream = device.build_input_stream(
        &stream_config.into(),
        move |data: &[f32], _: &cpal::InputCallbackInfo| {
            if listening_clone.load(Ordering::SeqCst) {
                let mono = to_mono(data, device_channels);
                let resampled = resample(&mono, device_sample_rate, SAMPLE_RATE);

                let mut buffer = audio_buffer_clone.lock().unwrap();
                // Cap buffer to prevent unbounded growth
                if buffer.len() + resampled.len() <= MAX_BUFFER_SAMPLES {
                    buffer.extend(resampled);
                }
            }
        },
        |err| eprintln!("Audio stream error: {}", err),
        None,
    )?;

    stream.play()?;

    // Keyboard state
    let device_state = DeviceState::new();
    let mut was_key_pressed = false;
    let mut is_listening = false;
    let mut vad = RmsVad::new(RMS_THRESHOLD, SILENCE_DURATION_MS);

    // Keyboard simulator for typing
    let mut enigo = Enigo::new(&Settings::default())?;

    println!("\n┌────────────────────────────────────────┐");
    println!("│  Continuous Transcription Ready        │");
    println!("│                                        │");
    match ptt_mode {
        PttMode::Hold => {
            println!("│  {:12} - Hold to record           │", hotkey_config.display_name);
        }
        PttMode::Toggle => {
            println!("│  {:12} - Toggle listening         │", hotkey_config.display_name);
        }
    }
    println!("│  Ctrl+C       - Exit                   │");
    println!("│                                        │");
    println!("│  Transcriptions typed to active window │");
    println!("└────────────────────────────────────────┘\n");
    match ptt_mode {
        PttMode::Hold => println!("[IDLE] Hold {} to record...", hotkey_config.display_name),
        PttMode::Toggle => println!("[IDLE] Press {} to start listening...", hotkey_config.display_name),
    }

    while running.load(Ordering::SeqCst) {
        // Check for transcription results from background threads
        while let Ok(text) = rx.try_recv() {
            if let Err(e) = enigo.text(&text) {
                eprintln!("\r\x1b[K[ERROR] Failed to type: {}", e);
            } else {
                print!("\r\x1b[K[TYPED] \"{}\" ", truncate_for_display(&text, 40));
                if is_listening {
                    print!("(listening...)");
                }
                println!();
                std::io::stdout().flush()?;
            }
        }

        // Check hotkey
        let keys = device_state.get_keys();
        let key_pressed = hotkey_config.is_pressed(&keys);
        let key_just_pressed = key_pressed && !was_key_pressed;
        let key_just_released = !key_pressed && was_key_pressed;
        was_key_pressed = key_pressed;

        // Handle key events based on mode
        match ptt_mode {
            PttMode::Toggle => {
                if key_just_pressed {
                    is_listening = !is_listening;
                    listening.store(is_listening, Ordering::SeqCst);

                    if is_listening {
                        // Start listening
                        {
                            let mut buffer = audio_buffer.lock().unwrap();
                            buffer.clear();
                        }
                        vad.reset();
                        println!("\r\x1b[K[LISTENING] Speak now... (press {} to stop)", hotkey_config.display_name);
                    } else {
                        // Stop listening - transcribe any remaining audio
                        let buffer = audio_buffer.lock().unwrap();
                        if buffer.len() >= MIN_SPEECH_SAMPLES {
                            let samples = buffer.clone();
                            drop(buffer);

                            if transcribing.compare_exchange(false, true, Ordering::SeqCst, Ordering::SeqCst).is_ok() {
                                let model = model.clone();
                                let tx = tx.clone();
                                let transcribing = transcribing.clone();
                                thread::spawn(move || {
                                    if let Ok(mut model) = model.try_lock() {
                                        match model.transcribe_samples(samples, SAMPLE_RATE, CHANNELS, None) {
                                            Ok(result) => {
                                                let text = result.text.trim().to_string();
                                                if !text.is_empty() {
                                                    let _ = tx.send(text);
                                                }
                                            }
                                            Err(e) => eprintln!("\r\x1b[K[ERROR] Transcription failed: {}", e),
                                        }
                                    }
                                    transcribing.store(false, Ordering::SeqCst);
                                });
                            }
                        } else {
                            drop(buffer);
                        }
                        println!("\r\x1b[K[IDLE] Press {} to start listening...", hotkey_config.display_name);
                    }
                }
            }
            PttMode::Hold => {
                if key_just_pressed && !is_listening {
                    // Start listening (hold mode)
                    is_listening = true;
                    listening.store(true, Ordering::SeqCst);
                    {
                        let mut buffer = audio_buffer.lock().unwrap();
                        buffer.clear();
                    }
                    vad.reset();
                    println!("\r\x1b[K[LISTENING] Recording... (release {} to transcribe)", hotkey_config.display_name);
                }

                if key_just_released && is_listening {
                    // Stop listening and transcribe (hold mode)
                    is_listening = false;
                    listening.store(false, Ordering::SeqCst);

                    let buffer = audio_buffer.lock().unwrap();
                    if buffer.len() >= MIN_SPEECH_SAMPLES {
                        let samples = buffer.clone();
                        drop(buffer);

                        if transcribing.compare_exchange(false, true, Ordering::SeqCst, Ordering::SeqCst).is_ok() {
                            print!("\r\x1b[K[TRANSCRIBING] Processing...");
                            std::io::stdout().flush()?;

                            let model = model.clone();
                            let tx = tx.clone();
                            let transcribing = transcribing.clone();
                            let hotkey_display = hotkey_config.display_name.clone();
                            thread::spawn(move || {
                                if let Ok(mut model) = model.try_lock() {
                                    match model.transcribe_samples(samples, SAMPLE_RATE, CHANNELS, None) {
                                        Ok(result) => {
                                            let text = result.text.trim().to_string();
                                            if !text.is_empty() {
                                                let _ = tx.send(text);
                                            }
                                        }
                                        Err(e) => eprintln!("\r\x1b[K[ERROR] Transcription failed: {}", e),
                                    }
                                }
                                transcribing.store(false, Ordering::SeqCst);
                                println!("\r\x1b[K[IDLE] Hold {} to record...", hotkey_display);
                            });
                        } else {
                            println!("\r\x1b[K[BUSY] Previous transcription still running, skipping...");
                        }
                    } else {
                        drop(buffer);
                        println!("\r\x1b[K[IDLE] (too short) Hold {} to record...", hotkey_config.display_name);
                    }
                }
            }
        }

        // While listening in toggle mode, check for end-of-speech to trigger transcription
        // (In hold mode, user controls when to stop via key release)
        if is_listening && ptt_mode == PttMode::Toggle {
            let mut buffer = audio_buffer.lock().unwrap();

            if buffer.len() > 1600 {
                // Check last ~100ms for VAD
                let check_start = buffer.len().saturating_sub(1600);
                let recent = &buffer[check_start..];

                if vad.process(recent) && buffer.len() >= MIN_SPEECH_SAMPLES {
                    // End of speech detected - transcribe this segment
                    let samples = buffer.clone();
                    buffer.clear(); // Clear for next utterance
                    drop(buffer);

                    // Skip if already transcribing to prevent pile-up
                    if transcribing.compare_exchange(false, true, Ordering::SeqCst, Ordering::SeqCst).is_ok() {
                        print!("\r\x1b[K[TRANSCRIBING] Processing segment...");
                        std::io::stdout().flush()?;

                        let model = model.clone();
                        let tx = tx.clone();
                        let transcribing = transcribing.clone();
                        thread::spawn(move || {
                            if let Ok(mut model) = model.try_lock() {
                                match model.transcribe_samples(samples, SAMPLE_RATE, CHANNELS, None) {
                                    Ok(result) => {
                                        let text = result.text.trim().to_string();
                                        if !text.is_empty() {
                                            let _ = tx.send(text);
                                        }
                                    }
                                    Err(e) => eprintln!("\r\x1b[K[ERROR] Transcription failed: {}", e),
                                }
                            }
                            transcribing.store(false, Ordering::SeqCst);
                        });
                    }
                }
            }
        }

        std::thread::sleep(Duration::from_millis(10));
    }

    println!("\nGoodbye!");
    Ok(())
}

fn truncate_for_display(s: &str, max_chars: usize) -> String {
    let char_count = s.chars().count();
    if char_count <= max_chars {
        s.to_string()
    } else {
        let truncated: String = s.chars().take(max_chars.saturating_sub(3)).collect();
        format!("{}...", truncated)
    }
}
