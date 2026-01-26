/*
Push-to-talk microphone transcription with continuous listening

Types transcriptions directly into the active window.

Usage:
    cargo run --release --example mic_transcribe --features cuda -- ../parakeet-tdt-0.6b-v3-onnx

Config file (~/.config/mic_transcribe.conf):
    hotkey = Ctrl+Alt+Shift   # Generic modifiers match either side of keyboard
    mode = hold               # hold = press-to-talk, toggle = press to start/stop
    # or: hotkey = LAlt+Grave           # Specific left-alt + grave key
    # or: hotkey = F12                  # Single key
    # or: hotkey = LControl+LShift+R    # Specific side modifiers

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
Generic modifiers (match either side): Ctrl, Alt, Shift, Meta/Super

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
const MIN_SPEECH_RMS: f32 = 0.005; // Minimum RMS to consider audio as containing speech

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

/// Check if audio buffer contains actual speech content (not just silence/noise)
fn has_speech_content(samples: &[f32]) -> bool {
    if samples.len() < MIN_SPEECH_SAMPLES {
        return false;
    }
    calculate_rms(samples) >= MIN_SPEECH_RMS
}

/// Inverse Text Normalization (ITN) - convert spoken numbers to digits
fn normalize_numbers(text: &str) -> String {
    use std::collections::HashMap;

    // Basic number words to digits
    let units: HashMap<&str, &str> = [
        ("zero", "0"), ("one", "1"), ("two", "2"), ("three", "3"), ("four", "4"),
        ("five", "5"), ("six", "6"), ("seven", "7"), ("eight", "8"), ("nine", "9"),
        ("ten", "10"), ("eleven", "11"), ("twelve", "12"), ("thirteen", "13"),
        ("fourteen", "14"), ("fifteen", "15"), ("sixteen", "16"), ("seventeen", "17"),
        ("eighteen", "18"), ("nineteen", "19"),
    ].into_iter().collect();

    let tens: HashMap<&str, u32> = [
        ("twenty", 20), ("thirty", 30), ("forty", 40), ("fifty", 50),
        ("sixty", 60), ("seventy", 70), ("eighty", 80), ("ninety", 90),
    ].into_iter().collect();

    let mut result = text.to_lowercase();

    // Handle compound numbers like "twenty-three" or "twenty three"
    for (ten_word, ten_val) in &tens {
        for (unit_word, unit_str) in &units {
            let unit_val: u32 = unit_str.parse().unwrap_or(0);
            if unit_val >= 1 && unit_val <= 9 {
                let compound = ten_val + unit_val;
                // Match "twenty-three" or "twenty three"
                let hyphenated = format!("{}-{}", ten_word, unit_word);
                let spaced = format!("{} {}", ten_word, unit_word);
                result = result.replace(&hyphenated, &compound.to_string());
                result = result.replace(&spaced, &compound.to_string());
            }
        }
    }

    // Handle standalone tens (twenty, thirty, etc.)
    for (ten_word, ten_val) in &tens {
        // Use word boundaries to avoid partial replacements
        let pattern = format!(r"\b{}\b", ten_word);
        if let Ok(re) = regex::Regex::new(&pattern) {
            result = re.replace_all(&result, ten_val.to_string().as_str()).to_string();
        }
    }

    // Handle standalone units (one, two, etc.) - be careful with common words
    // Only replace when they appear as standalone numbers, not in phrases
    for (unit_word, unit_str) in &units {
        // Skip words that are commonly used in non-numeric contexts
        if *unit_word == "one" || *unit_word == "two" || *unit_word == "four" {
            continue; // These are too ambiguous ("one of", "two of", "four" as adjective)
        }
        let pattern = format!(r"\b{}\b", unit_word);
        if let Ok(re) = regex::Regex::new(&pattern) {
            result = re.replace_all(&result, *unit_str).to_string();
        }
    }

    // Handle "hundred" and "thousand" patterns
    // e.g., "three hundred" -> "300", "five thousand" -> "5000"
    if let Ok(re) = regex::Regex::new(r"\b(\d+)\s+hundred\b") {
        result = re.replace_all(&result, |caps: &regex::Captures| {
            let num: u32 = caps[1].parse().unwrap_or(0);
            (num * 100).to_string()
        }).to_string();
    }

    if let Ok(re) = regex::Regex::new(r"\b(\d+)\s+thousand\b") {
        result = re.replace_all(&result, |caps: &regex::Captures| {
            let num: u32 = caps[1].parse().unwrap_or(0);
            (num * 1000).to_string()
        }).to_string();
    }

    // Handle "X hundred Y" pattern (e.g., "3 hundred 50" -> "350")
    if let Ok(re) = regex::Regex::new(r"\b(\d+)00\s+(\d{1,2})\b") {
        result = re.replace_all(&result, |caps: &regex::Captures| {
            let hundreds: u32 = caps[1].parse().unwrap_or(0);
            let rest: u32 = caps[2].parse().unwrap_or(0);
            (hundreds * 100 + rest).to_string()
        }).to_string();
    }

    result
}

// i3status-rust integration
const STATUS_FILE: &str = "/tmp/parakeet-status";
const CONTROL_FILE: &str = "/tmp/parakeet-control";

fn write_status(state: &str) {
    let text = match state {
        "listening" => "󰏃 Listening",
        "transcribing" => "⏳ Transcribing",
        _ => "󰍭 Idle",
    };
    let _ = fs::write(STATUS_FILE, text);
}

fn clear_status() {
    let _ = fs::remove_file(STATUS_FILE);
    let _ = fs::remove_file(CONTROL_FILE);
}

/// Check if toggle was requested via control file (e.g., i3status-rust click)
fn check_toggle_request() -> bool {
    if let Ok(content) = fs::read_to_string(CONTROL_FILE) {
        let _ = fs::remove_file(CONTROL_FILE);
        content.trim() == "toggle"
    } else {
        false
    }
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

/// Represents a modifier that can match either left or right variants
#[derive(Clone)]
enum Modifier {
    Specific(Keycode),           // Exact key (e.g., LControl)
    Either(Keycode, Keycode),    // Either variant (e.g., LControl OR RControl)
}

impl Modifier {
    fn is_pressed(&self, keys: &[Keycode]) -> bool {
        match self {
            Modifier::Specific(k) => keys.contains(k),
            Modifier::Either(l, r) => keys.contains(l) || keys.contains(r),
        }
    }
}

/// Hotkey configuration with optional modifiers
#[derive(Clone)]
struct HotkeyConfig {
    key: Option<Keycode>,        // Main key (optional for modifier-only combos)
    modifiers: Vec<Modifier>,
    display_name: String,
}

impl HotkeyConfig {
    fn is_pressed(&self, keys: &[Keycode]) -> bool {
        // Check main key is pressed (if specified)
        if let Some(ref key) = self.key {
            if !keys.contains(key) {
                return false;
            }
        }
        // Check all modifiers are pressed
        for modifier in &self.modifiers {
            if !modifier.is_pressed(keys) {
                return false;
            }
        }
        // Must have at least one key requirement
        self.key.is_some() || !self.modifiers.is_empty()
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

/// Parse generic modifier names that match either left or right variants
fn parse_generic_modifier(key_name: &str) -> Option<Modifier> {
    match key_name.to_lowercase().as_str() {
        "control" | "ctrl" => Some(Modifier::Either(Keycode::LControl, Keycode::RControl)),
        "alt" => Some(Modifier::Either(Keycode::LAlt, Keycode::RAlt)),
        "shift" => Some(Modifier::Either(Keycode::LShift, Keycode::RShift)),
        "meta" | "super" => Some(Modifier::Either(Keycode::LMeta, Keycode::RMeta)),
        _ => None,
    }
}

fn parse_hotkey_combo(combo: &str) -> Option<HotkeyConfig> {
    let parts: Vec<&str> = combo.split('+').map(|s| s.trim()).collect();
    if parts.is_empty() {
        return None;
    }

    let mut modifiers: Vec<Modifier> = Vec::new();
    let mut main_key: Option<Keycode> = None;

    for part in &parts {
        // First, try parsing as a generic modifier (Control, Alt, Shift, Meta)
        if let Some(generic_mod) = parse_generic_modifier(part) {
            modifiers.push(generic_mod);
        } else if let Some(keycode) = parse_keycode(part) {
            // Specific keycode - could be a specific modifier or a regular key
            if is_modifier(&keycode) && parts.len() > 1 {
                modifiers.push(Modifier::Specific(keycode));
            } else {
                main_key = Some(keycode);
            }
        } else {
            eprintln!("Warning: Unknown key '{}' in hotkey combo", part);
            return None;
        }
    }

    // Allow modifier-only combos (e.g., Ctrl+Alt+Shift)
    if main_key.is_none() && modifiers.is_empty() {
        return None;
    }

    Some(HotkeyConfig {
        key: main_key,
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
            // Default config: Ctrl+Alt+Shift (either side)
            Config {
                hotkey: HotkeyConfig {
                    key: None,
                    modifiers: vec![
                        Modifier::Either(Keycode::LControl, Keycode::RControl),
                        Modifier::Either(Keycode::LAlt, Keycode::RAlt),
                        Modifier::Either(Keycode::LShift, Keycode::RShift),
                    ],
                    display_name: "Ctrl+Alt+Shift".to_string(),
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
            clear_status();
            running.store(false, Ordering::SeqCst);
        })?;
    }

    // Audio callback - accumulates samples when listening
    // Buffer limit: ~12 seconds at 16kHz = 192,000 samples
    const MAX_BUFFER_SAMPLES: usize = 192_000;
    // Handle buffer when it reaches this threshold (~8 seconds)
    const BUFFER_FULL_THRESHOLD: usize = 128_000;
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
                // Note: if buffer is full, new audio is dropped - main loop handles this
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
    write_status("idle");

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

        // Check for toggle request from i3status-rust click
        let click_toggle = check_toggle_request();

        // Handle key events based on mode
        match ptt_mode {
            PttMode::Toggle => {
                if key_just_pressed || click_toggle {
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
                        write_status("listening");
                    } else {
                        // Stop listening - transcribe any remaining audio
                        let buffer = audio_buffer.lock().unwrap();
                        let samples = buffer.clone();
                        drop(buffer);

                        if has_speech_content(&samples) {
                            if transcribing.compare_exchange(false, true, Ordering::SeqCst, Ordering::SeqCst).is_ok() {
                                let model = model.clone();
                                let tx = tx.clone();
                                let transcribing = transcribing.clone();
                                thread::spawn(move || {
                                    if let Ok(mut model) = model.try_lock() {
                                        match model.transcribe_samples(samples, SAMPLE_RATE, CHANNELS, None) {
                                            Ok(result) => {
                                                let text = normalize_numbers(result.text.trim());
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
                        println!("\r\x1b[K[IDLE] Press {} to start listening...", hotkey_config.display_name);
                        write_status("idle");
                    }
                }
            }
            PttMode::Hold => {
                // Click toggle acts like toggle mode in hold mode
                if click_toggle && !is_listening {
                    // Start listening via click
                    is_listening = true;
                    listening.store(true, Ordering::SeqCst);
                    {
                        let mut buffer = audio_buffer.lock().unwrap();
                        buffer.clear();
                    }
                    vad.reset();
                    println!("\r\x1b[K[LISTENING] Recording... (click or release {} to transcribe)", hotkey_config.display_name);
                    write_status("listening");
                } else if key_just_pressed && !is_listening {
                    // Start listening (hold mode - key press)
                    is_listening = true;
                    listening.store(true, Ordering::SeqCst);
                    {
                        let mut buffer = audio_buffer.lock().unwrap();
                        buffer.clear();
                    }
                    vad.reset();
                    println!("\r\x1b[K[LISTENING] Recording... (release {} to transcribe)", hotkey_config.display_name);
                    write_status("listening");
                }

                // Transcribe on key release OR click toggle while listening
                let stop_requested = key_just_released || (click_toggle && is_listening);
                if stop_requested && is_listening {
                    // Stop listening and transcribe (hold mode)
                    is_listening = false;
                    listening.store(false, Ordering::SeqCst);

                    let buffer = audio_buffer.lock().unwrap();
                    let samples = buffer.clone();
                    drop(buffer);

                    if has_speech_content(&samples) {
                        if transcribing.compare_exchange(false, true, Ordering::SeqCst, Ordering::SeqCst).is_ok() {
                            print!("\r\x1b[K[TRANSCRIBING] Processing...");
                            std::io::stdout().flush()?;
                            write_status("transcribing");

                            let model = model.clone();
                            let tx = tx.clone();
                            let transcribing = transcribing.clone();
                            let hotkey_display = hotkey_config.display_name.clone();
                            thread::spawn(move || {
                                if let Ok(mut model) = model.try_lock() {
                                    match model.transcribe_samples(samples, SAMPLE_RATE, CHANNELS, None) {
                                        Ok(result) => {
                                            let text = normalize_numbers(result.text.trim());
                                            if !text.is_empty() {
                                                let _ = tx.send(text);
                                            }
                                        }
                                        Err(e) => eprintln!("\r\x1b[K[ERROR] Transcription failed: {}", e),
                                    }
                                }
                                transcribing.store(false, Ordering::SeqCst);
                                println!("\r\x1b[K[IDLE] Hold {} to record...", hotkey_display);
                                write_status("idle");
                            });
                        } else {
                            println!("\r\x1b[K[BUSY] Previous transcription still running, skipping...");
                        }
                    } else {
                        println!("\r\x1b[K[IDLE] (no speech detected) Hold {} to record...", hotkey_config.display_name);
                        write_status("idle");
                    }
                }
            }
        }

        // While listening in toggle mode, check for end-of-speech to trigger transcription
        // (In hold mode, user controls when to stop via key release)
        if is_listening && ptt_mode == PttMode::Toggle {
            let mut buffer = audio_buffer.lock().unwrap();
            let buffer_len = buffer.len();

            if buffer_len > 1600 {
                // Check last ~100ms for VAD
                let check_start = buffer_len.saturating_sub(1600);
                let recent = &buffer[check_start..];

                // Trigger transcription if:
                // 1. VAD detects end of speech, OR
                // 2. Buffer is getting full (transcribe if speech, discard if silence)
                let vad_triggered = vad.process(recent);
                let buffer_full = buffer_len >= BUFFER_FULL_THRESHOLD;

                if vad_triggered || buffer_full {
                    // Check if buffer has speech content
                    let has_speech = has_speech_content(&buffer);

                    // If buffer is full but no speech, just discard and continue
                    if buffer_full && !has_speech {
                        buffer.clear();
                        drop(buffer);
                        vad.reset();
                        continue;
                    }

                    // Only proceed with transcription if we have speech
                    if has_speech {
                        // Only proceed if we can actually transcribe (not already busy)
                        if transcribing.compare_exchange(false, true, Ordering::SeqCst, Ordering::SeqCst).is_ok() {
                            // Now safe to take the samples since we'll actually transcribe them
                            let samples = buffer.clone();
                            buffer.clear();
                            drop(buffer);

                            print!("\r\x1b[K[TRANSCRIBING] Processing segment ({:.1}s)...", samples.len() as f32 / SAMPLE_RATE as f32);
                            std::io::stdout().flush()?;
                            write_status("transcribing");

                            let model = model.clone();
                            let tx = tx.clone();
                            let transcribing = transcribing.clone();
                            thread::spawn(move || {
                                if let Ok(mut model) = model.try_lock() {
                                    match model.transcribe_samples(samples, SAMPLE_RATE, CHANNELS, None) {
                                        Ok(result) => {
                                            let text = normalize_numbers(result.text.trim());
                                            if !text.is_empty() {
                                                let _ = tx.send(text);
                                            }
                                        }
                                        Err(e) => eprintln!("\r\x1b[K[ERROR] Transcription failed: {}", e),
                                    }
                                }
                                transcribing.store(false, Ordering::SeqCst);
                                // Still listening in toggle mode, so go back to listening state
                                write_status("listening");
                            });
                        } else if buffer_full {
                            // Buffer is full but we're busy transcribing
                            // Drop oldest audio to make room for new
                            let drop_samples = buffer_len / 2; // Drop oldest 50%
                            buffer.drain(0..drop_samples);
                            drop(buffer);
                        }
                        // If VAD triggered but transcribing is busy, keep accumulating
                        // (don't clear buffer, audio will be transcribed when current one finishes)
                    }
                }
            }
        }

        std::thread::sleep(Duration::from_millis(10));
    }

    clear_status();
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_generic_modifier_ctrl() {
        let modifier = parse_generic_modifier("Ctrl").unwrap();
        match modifier {
            Modifier::Either(l, r) => {
                assert_eq!(l, Keycode::LControl);
                assert_eq!(r, Keycode::RControl);
            }
            _ => panic!("Expected Either variant"),
        }
    }

    #[test]
    fn test_parse_generic_modifier_alt() {
        let modifier = parse_generic_modifier("Alt").unwrap();
        match modifier {
            Modifier::Either(l, r) => {
                assert_eq!(l, Keycode::LAlt);
                assert_eq!(r, Keycode::RAlt);
            }
            _ => panic!("Expected Either variant"),
        }
    }

    #[test]
    fn test_parse_generic_modifier_shift() {
        let modifier = parse_generic_modifier("Shift").unwrap();
        match modifier {
            Modifier::Either(l, r) => {
                assert_eq!(l, Keycode::LShift);
                assert_eq!(r, Keycode::RShift);
            }
            _ => panic!("Expected Either variant"),
        }
    }

    #[test]
    fn test_parse_generic_modifier_meta() {
        let modifier = parse_generic_modifier("Meta").unwrap();
        match modifier {
            Modifier::Either(l, r) => {
                assert_eq!(l, Keycode::LMeta);
                assert_eq!(r, Keycode::RMeta);
            }
            _ => panic!("Expected Either variant"),
        }
    }

    #[test]
    fn test_parse_hotkey_combo_single_key() {
        let config = parse_hotkey_combo("F12").unwrap();
        assert_eq!(config.key, Some(Keycode::F12));
        assert!(config.modifiers.is_empty());
    }

    #[test]
    fn test_parse_hotkey_combo_specific_modifier_and_key() {
        let config = parse_hotkey_combo("LControl+Space").unwrap();
        assert_eq!(config.key, Some(Keycode::Space));
        assert_eq!(config.modifiers.len(), 1);
        match &config.modifiers[0] {
            Modifier::Specific(k) => assert_eq!(*k, Keycode::LControl),
            _ => panic!("Expected Specific variant"),
        }
    }

    #[test]
    fn test_parse_hotkey_combo_generic_modifiers_only() {
        let config = parse_hotkey_combo("Ctrl+Alt+Shift").unwrap();
        assert_eq!(config.key, None);
        assert_eq!(config.modifiers.len(), 3);

        // Check all are Either variants
        for modifier in &config.modifiers {
            match modifier {
                Modifier::Either(_, _) => {}
                _ => panic!("Expected Either variant"),
            }
        }
    }

    #[test]
    fn test_parse_hotkey_combo_mixed_modifiers_and_key() {
        let config = parse_hotkey_combo("Ctrl+Shift+R").unwrap();
        assert_eq!(config.key, Some(Keycode::R));
        assert_eq!(config.modifiers.len(), 2);
    }

    #[test]
    fn test_hotkey_is_pressed_single_key() {
        let config = parse_hotkey_combo("F12").unwrap();
        assert!(config.is_pressed(&[Keycode::F12]));
        assert!(!config.is_pressed(&[Keycode::F11]));
        assert!(!config.is_pressed(&[]));
    }

    #[test]
    fn test_hotkey_is_pressed_specific_modifier() {
        let config = parse_hotkey_combo("LControl+Space").unwrap();
        assert!(config.is_pressed(&[Keycode::LControl, Keycode::Space]));
        assert!(!config.is_pressed(&[Keycode::RControl, Keycode::Space])); // Wrong side
        assert!(!config.is_pressed(&[Keycode::Space])); // Missing modifier
        assert!(!config.is_pressed(&[Keycode::LControl])); // Missing key
    }

    #[test]
    fn test_hotkey_is_pressed_generic_modifier_left_side() {
        let config = parse_hotkey_combo("Ctrl+Alt+Shift").unwrap();
        // Left side should work
        assert!(config.is_pressed(&[Keycode::LControl, Keycode::LAlt, Keycode::LShift]));
    }

    #[test]
    fn test_hotkey_is_pressed_generic_modifier_right_side() {
        let config = parse_hotkey_combo("Ctrl+Alt+Shift").unwrap();
        // Right side should work
        assert!(config.is_pressed(&[Keycode::RControl, Keycode::RAlt, Keycode::RShift]));
    }

    #[test]
    fn test_hotkey_is_pressed_generic_modifier_mixed_sides() {
        let config = parse_hotkey_combo("Ctrl+Alt+Shift").unwrap();
        // Mixed sides should also work (LCtrl + RAlt + LShift)
        assert!(config.is_pressed(&[Keycode::LControl, Keycode::RAlt, Keycode::LShift]));
    }

    #[test]
    fn test_hotkey_is_pressed_generic_modifier_missing_one() {
        let config = parse_hotkey_combo("Ctrl+Alt+Shift").unwrap();
        // Missing one modifier should fail
        assert!(!config.is_pressed(&[Keycode::LControl, Keycode::LAlt]));
        assert!(!config.is_pressed(&[Keycode::LControl, Keycode::LShift]));
        assert!(!config.is_pressed(&[Keycode::LAlt, Keycode::LShift]));
    }

    #[test]
    fn test_hotkey_is_pressed_with_extra_keys() {
        let config = parse_hotkey_combo("Ctrl+Alt+Shift").unwrap();
        // Extra keys should be ignored (hotkey still works)
        assert!(config.is_pressed(&[
            Keycode::LControl,
            Keycode::LAlt,
            Keycode::LShift,
            Keycode::A
        ]));
    }

    #[test]
    fn test_parse_hotkey_combo_invalid_key() {
        assert!(parse_hotkey_combo("InvalidKey").is_none());
        assert!(parse_hotkey_combo("Ctrl+InvalidKey").is_none());
    }

    #[test]
    fn test_parse_hotkey_combo_empty() {
        assert!(parse_hotkey_combo("").is_none());
    }

    #[test]
    fn test_modifier_either_is_pressed() {
        let modifier = Modifier::Either(Keycode::LControl, Keycode::RControl);
        assert!(modifier.is_pressed(&[Keycode::LControl]));
        assert!(modifier.is_pressed(&[Keycode::RControl]));
        assert!(modifier.is_pressed(&[Keycode::LControl, Keycode::RControl]));
        assert!(!modifier.is_pressed(&[Keycode::LAlt]));
        assert!(!modifier.is_pressed(&[]));
    }

    #[test]
    fn test_modifier_specific_is_pressed() {
        let modifier = Modifier::Specific(Keycode::LControl);
        assert!(modifier.is_pressed(&[Keycode::LControl]));
        assert!(!modifier.is_pressed(&[Keycode::RControl]));
        assert!(!modifier.is_pressed(&[]));
    }

    // ==================== Edge Cases ====================

    #[test]
    fn test_case_insensitivity() {
        // All case variations should parse the same
        assert!(parse_hotkey_combo("ctrl").is_some());
        assert!(parse_hotkey_combo("CTRL").is_some());
        assert!(parse_hotkey_combo("Ctrl").is_some());
        assert!(parse_hotkey_combo("cTrL").is_some());

        // They should all produce equivalent behavior
        let lower = parse_hotkey_combo("ctrl+alt+shift").unwrap();
        let upper = parse_hotkey_combo("CTRL+ALT+SHIFT").unwrap();
        let mixed = parse_hotkey_combo("Ctrl+Alt+Shift").unwrap();

        let keys = vec![Keycode::LControl, Keycode::LAlt, Keycode::LShift];
        assert!(lower.is_pressed(&keys));
        assert!(upper.is_pressed(&keys));
        assert!(mixed.is_pressed(&keys));
    }

    #[test]
    fn test_whitespace_around_plus() {
        // Spaces around + should be trimmed
        let config = parse_hotkey_combo("Ctrl + Alt + Shift").unwrap();
        assert_eq!(config.modifiers.len(), 3);
        assert!(config.is_pressed(&[Keycode::LControl, Keycode::LAlt, Keycode::LShift]));

        // Tabs and multiple spaces
        let config2 = parse_hotkey_combo("Ctrl  +  Alt").unwrap();
        assert!(config2.is_pressed(&[Keycode::LControl, Keycode::LAlt]));
    }

    #[test]
    fn test_single_modifier_as_hotkey() {
        // A single specific modifier should work as a hotkey
        let config = parse_hotkey_combo("LControl").unwrap();
        assert_eq!(config.key, Some(Keycode::LControl));
        assert!(config.modifiers.is_empty());
        assert!(config.is_pressed(&[Keycode::LControl]));
        assert!(!config.is_pressed(&[Keycode::RControl]));

        // A single generic modifier should also work
        let config2 = parse_hotkey_combo("Ctrl").unwrap();
        // When it's the only key, it becomes the main key (not a modifier)
        // Wait, let me check this...actually for a single generic modifier,
        // it should be treated as a modifier-only hotkey
        assert!(config2.is_pressed(&[Keycode::LControl]));
        assert!(config2.is_pressed(&[Keycode::RControl]));
    }

    #[test]
    fn test_both_sides_pressed_satisfies_generic() {
        let config = parse_hotkey_combo("Ctrl").unwrap();
        // Both LControl and RControl pressed should still satisfy Ctrl
        assert!(config.is_pressed(&[Keycode::LControl, Keycode::RControl]));
    }

    #[test]
    fn test_order_independence() {
        // Different orderings should produce functionally equivalent hotkeys
        let order1 = parse_hotkey_combo("Ctrl+Alt+Shift").unwrap();
        let order2 = parse_hotkey_combo("Shift+Ctrl+Alt").unwrap();
        let order3 = parse_hotkey_combo("Alt+Shift+Ctrl").unwrap();

        let keys = vec![Keycode::LControl, Keycode::LAlt, Keycode::LShift];
        assert!(order1.is_pressed(&keys));
        assert!(order2.is_pressed(&keys));
        assert!(order3.is_pressed(&keys));
    }

    #[test]
    fn test_empty_parts_rejected() {
        // Double plus should fail (creates empty part)
        assert!(parse_hotkey_combo("Ctrl++Alt").is_none());
        // Trailing plus
        assert!(parse_hotkey_combo("Ctrl+").is_none());
        // Leading plus
        assert!(parse_hotkey_combo("+Ctrl").is_none());
    }

    #[test]
    fn test_control_ctrl_aliases() {
        // Both "Control" and "Ctrl" should work
        let ctrl = parse_hotkey_combo("Ctrl").unwrap();
        let control = parse_hotkey_combo("Control").unwrap();

        assert!(ctrl.is_pressed(&[Keycode::LControl]));
        assert!(control.is_pressed(&[Keycode::LControl]));
        assert!(ctrl.is_pressed(&[Keycode::RControl]));
        assert!(control.is_pressed(&[Keycode::RControl]));
    }

    #[test]
    fn test_super_meta_aliases() {
        // Both "Super" and "Meta" should work
        let super_mod = parse_hotkey_combo("Super").unwrap();
        let meta = parse_hotkey_combo("Meta").unwrap();

        assert!(super_mod.is_pressed(&[Keycode::LMeta]));
        assert!(meta.is_pressed(&[Keycode::LMeta]));
        assert!(super_mod.is_pressed(&[Keycode::RMeta]));
        assert!(meta.is_pressed(&[Keycode::RMeta]));
    }

    #[test]
    fn test_complex_combo_with_key() {
        // Multiple modifiers plus a regular key
        let config = parse_hotkey_combo("Ctrl+Alt+Shift+R").unwrap();
        assert_eq!(config.key, Some(Keycode::R));
        assert_eq!(config.modifiers.len(), 3);

        // Should work with all modifiers and key pressed
        assert!(config.is_pressed(&[
            Keycode::LControl,
            Keycode::LAlt,
            Keycode::LShift,
            Keycode::R
        ]));

        // Missing the key should fail
        assert!(!config.is_pressed(&[Keycode::LControl, Keycode::LAlt, Keycode::LShift]));

        // Missing one modifier should fail
        assert!(!config.is_pressed(&[Keycode::LControl, Keycode::LAlt, Keycode::R]));
    }

    #[test]
    fn test_specific_and_generic_modifiers_mixed() {
        // Mix specific (LControl) with generic (Alt)
        let config = parse_hotkey_combo("LControl+Alt+R").unwrap();
        assert_eq!(config.key, Some(Keycode::R));

        // LControl + LAlt should work
        assert!(config.is_pressed(&[Keycode::LControl, Keycode::LAlt, Keycode::R]));
        // LControl + RAlt should also work (Alt is generic)
        assert!(config.is_pressed(&[Keycode::LControl, Keycode::RAlt, Keycode::R]));
        // RControl should NOT work (LControl is specific)
        assert!(!config.is_pressed(&[Keycode::RControl, Keycode::LAlt, Keycode::R]));
    }

    #[test]
    fn test_all_function_keys() {
        // Spot check several function keys
        for (name, expected) in [
            ("F1", Keycode::F1),
            ("F12", Keycode::F12),
            ("F20", Keycode::F20),
        ] {
            let config = parse_hotkey_combo(name).unwrap();
            assert_eq!(config.key, Some(expected));
        }
    }

    #[test]
    fn test_special_keys() {
        // Test various special key names
        for (name, expected) in [
            ("Grave", Keycode::Grave),
            ("`", Keycode::Grave),
            ("backtick", Keycode::Grave),
            ("Space", Keycode::Space),
            ("Tab", Keycode::Tab),
            ("Escape", Keycode::Escape),
            ("Esc", Keycode::Escape),
            ("Enter", Keycode::Enter),
            ("Return", Keycode::Enter),
        ] {
            let config = parse_hotkey_combo(name).unwrap();
            assert_eq!(config.key, Some(expected), "Failed for key: {}", name);
        }
    }

    #[test]
    fn test_is_pressed_empty_config_returns_false() {
        // A config with no key and no modifiers should never match
        // (This shouldn't be possible to create via parse, but test the struct directly)
        let config = HotkeyConfig {
            key: None,
            modifiers: vec![],
            display_name: "empty".to_string(),
        };
        assert!(!config.is_pressed(&[]));
        assert!(!config.is_pressed(&[Keycode::A]));
    }

    #[test]
    fn test_very_long_key_list() {
        // Hotkey should match even with many extra keys pressed
        let config = parse_hotkey_combo("Ctrl+R").unwrap();
        let many_keys = vec![
            Keycode::A,
            Keycode::B,
            Keycode::C,
            Keycode::LControl,
            Keycode::R,
            Keycode::D,
            Keycode::E,
        ];
        assert!(config.is_pressed(&many_keys));
    }

    #[test]
    fn test_number_keys() {
        for i in 0..=9 {
            let name = format!("{}", i);
            let config = parse_hotkey_combo(&name).unwrap();
            assert!(config.key.is_some(), "Failed for number key: {}", i);
        }
    }

    #[test]
    fn test_letter_keys() {
        for c in 'a'..='z' {
            let config = parse_hotkey_combo(&c.to_string()).unwrap();
            assert!(config.key.is_some(), "Failed for letter key: {}", c);
        }
    }

    #[test]
    fn test_modifier_only_two_keys() {
        // Two generic modifiers without a main key
        let config = parse_hotkey_combo("Ctrl+Shift").unwrap();
        assert_eq!(config.key, None);
        assert_eq!(config.modifiers.len(), 2);
        assert!(config.is_pressed(&[Keycode::LControl, Keycode::LShift]));
        assert!(config.is_pressed(&[Keycode::RControl, Keycode::RShift]));
    }

    #[test]
    fn test_specific_modifier_only() {
        // Two specific modifiers (same side)
        let config = parse_hotkey_combo("LControl+LShift").unwrap();
        // When there are only modifiers, the last one becomes the "main key"
        // Actually, let me verify this by checking the implementation...
        // From the parse logic: specific modifiers are added to modifiers vec when parts.len() > 1
        // So with two specific modifiers, the last one becomes main_key
        // Let's verify this works correctly
        assert!(config.is_pressed(&[Keycode::LControl, Keycode::LShift]));
        assert!(!config.is_pressed(&[Keycode::RControl, Keycode::RShift]));
    }

    // ==================== Number Normalization Tests ====================

    #[test]
    fn test_normalize_compound_numbers_hyphenated() {
        assert_eq!(normalize_numbers("twenty-one"), "21");
        assert_eq!(normalize_numbers("thirty-two"), "32");
        assert_eq!(normalize_numbers("forty-five"), "45");
        assert_eq!(normalize_numbers("ninety-nine"), "99");
    }

    #[test]
    fn test_normalize_compound_numbers_spaced() {
        assert_eq!(normalize_numbers("twenty one"), "21");
        assert_eq!(normalize_numbers("thirty two"), "32");
        assert_eq!(normalize_numbers("fifty seven"), "57");
    }

    #[test]
    fn test_normalize_standalone_tens() {
        assert_eq!(normalize_numbers("twenty"), "20");
        assert_eq!(normalize_numbers("thirty"), "30");
        assert_eq!(normalize_numbers("forty"), "40");
        assert_eq!(normalize_numbers("ninety"), "90");
    }

    #[test]
    fn test_normalize_teens() {
        assert_eq!(normalize_numbers("thirteen"), "13");
        assert_eq!(normalize_numbers("fifteen"), "15");
        assert_eq!(normalize_numbers("nineteen"), "19");
    }

    #[test]
    fn test_normalize_in_sentence() {
        assert_eq!(
            normalize_numbers("I have thirty-two apples"),
            "i have 32 apples"
        );
        assert_eq!(
            normalize_numbers("The answer is forty-two"),
            "the answer is 42"
        );
    }

    #[test]
    fn test_normalize_hundreds() {
        assert_eq!(normalize_numbers("5 hundred"), "500");
        assert_eq!(normalize_numbers("3 hundred"), "300");
    }

    #[test]
    fn test_normalize_preserves_existing_digits() {
        assert_eq!(normalize_numbers("I have 5 items"), "i have 5 items");
        assert_eq!(normalize_numbers("room 42"), "room 42");
    }

    #[test]
    fn test_normalize_multiple_numbers() {
        assert_eq!(
            normalize_numbers("twenty-one and thirty-two"),
            "21 and 32"
        );
    }

    #[test]
    fn test_normalize_case_insensitive() {
        assert_eq!(normalize_numbers("THIRTY-TWO"), "32");
        assert_eq!(normalize_numbers("Forty-Five"), "45");
    }
}
