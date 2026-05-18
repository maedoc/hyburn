use std::fs::File;
use std::io::{BufWriter, Write};
use std::sync::Mutex;

pub struct Logger {
    writer: Mutex<BufWriter<File>>,
}

impl Logger {
    pub fn new(path: &str) -> Result<Self, std::io::Error> {
        let file = File::create(path)?;
        Ok(Logger {
            writer: Mutex::new(BufWriter::new(file)),
        })
    }

    pub fn log(&self, event: &str) {
        if let Ok(mut w) = self.writer.lock() {
            let timestamp = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_secs_f64())
                .unwrap_or(0.0);
            let _ = writeln!(w, "[{:.3}] {}", timestamp, event);
            let _ = w.flush();
        }
    }

    pub fn log_start(&self, file_path: &str) {
        self.log(&format!("START file={}", file_path));
    }

    pub fn log_key(&self, code: &str) {
        self.log(&format!("KEY code={}", code));
    }

    pub fn log_action(&self, action: &str) {
        self.log(&format!("ACTION {}", action));
    }

    pub fn log_result(&self, path: &str, ok: bool) {
        self.log(&format!("RESULT path={} {}", path, if ok { "ok" } else { "error" }));
    }

    pub fn log_quit(&self, modified: bool) {
        self.log(&format!("QUIT modified={}", modified));
    }
}
