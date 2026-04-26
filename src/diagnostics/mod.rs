use ariadne::{Color, Label, Report, ReportKind, Source};
use std::io;
use std::sync::atomic::{AtomicUsize, Ordering};

use crate::ast::Span;

/// Severity level of a diagnostic.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Severity {
    Error,
    Warning,
    Info,
}

/// A labeled region in source code.
#[derive(Debug, Clone)]
pub struct DiagLabel {
    pub span: Span,
    pub message: String,
    pub color: Option<Color>,
}

/// A single diagnostic message with labeled source regions.
#[derive(Debug, Clone)]
pub struct Diagnostic {
    pub severity: Severity,
    pub message: String,
    pub labels: Vec<DiagLabel>,
    pub notes: Vec<String>,
}

impl Diagnostic {
    /// Create an error diagnostic.
    pub fn error(message: impl Into<String>) -> Self {
        Self {
            severity: Severity::Error,
            message: message.into(),
            labels: Vec::new(),
            notes: Vec::new(),
        }
    }

    /// Create a warning diagnostic.
    pub fn warning(message: impl Into<String>) -> Self {
        Self {
            severity: Severity::Warning,
            message: message.into(),
            labels: Vec::new(),
            notes: Vec::new(),
        }
    }

    /// Create an info diagnostic.
    pub fn info(message: impl Into<String>) -> Self {
        Self {
            severity: Severity::Info,
            message: message.into(),
            labels: Vec::new(),
            notes: Vec::new(),
        }
    }

    /// Add a labeled span.
    pub fn with_label(mut self, span: Span, message: impl Into<String>) -> Self {
        self.labels.push(DiagLabel {
            span,
            message: message.into(),
            color: None,
        });
        self
    }

    /// Add a labeled span with a specific color.
    pub fn with_colored_label(
        mut self,
        span: Span,
        message: impl Into<String>,
        color: Color,
    ) -> Self {
        self.labels.push(DiagLabel {
            span,
            message: message.into(),
            color: Some(color),
        });
        self
    }

    /// Add a note to the diagnostic.
    pub fn with_note(mut self, note: impl Into<String>) -> Self {
        self.notes.push(note.into());
        self
    }

    /// Render this diagnostic to a writer using ariadne.
    pub fn render(&self, source: &str, writer: &mut dyn io::Write) -> io::Result<()> {
        let kind = match self.severity {
            Severity::Error => ReportKind::Error,
            Severity::Warning => ReportKind::Warning,
            Severity::Info => ReportKind::Advice,
        };

        let span = self.labels.first().map(|l| l.span.clone()).unwrap_or(0..0);

        let mut builder = Report::build(kind, span).with_message(&self.message);

        for label in &self.labels {
            let color = label.color.unwrap_or(match self.severity {
                Severity::Error => Color::Red,
                Severity::Warning => Color::Yellow,
                Severity::Info => Color::Blue,
            });
            builder = builder.with_label(
                Label::new(label.span.clone())
                    .with_message(&label.message)
                    .with_color(color),
            );
        }

        for note in &self.notes {
            builder = builder.with_note(note);
        }

        let report = builder.finish();
        report.write(Source::from(source), writer)
    }

    /// Render to a String (useful for testing and error_fn callbacks).
    pub fn render_to_string(&self, source: &str) -> String {
        let mut buf = Vec::new();
        let _ = self.render(source, &mut buf);
        String::from_utf8(buf).unwrap_or_default()
    }

    /// Render to stderr.
    pub fn eprint(&self, source: &str) {
        let mut buf = Vec::new();
        let _ = self.render(source, &mut buf);
        let _ = io::Write::write_all(&mut io::stderr(), &buf);
    }

    /// Render to stderr for system-level diagnostics that don't tie
    /// back to a Wren source span — FFI load failures, ABI version
    /// mismatches, missing native libs, etc. Same colored Error /
    /// Warning header + notes as the labelled variant, but without
    /// a code-snippet rendering pass that would have nothing to
    /// show. Use this anywhere the codebase used to call
    /// `eprintln!("wrenlift: ...")` so every error and warning that
    /// reaches the user travels through the same formatter.
    pub fn eprint_no_source(&self) {
        self.eprint("")
    }
}

/// Accumulates diagnostics during compilation.
pub struct DiagnosticEmitter {
    diagnostics: Vec<Diagnostic>,
    error_count: AtomicUsize,
}

impl DiagnosticEmitter {
    pub fn new() -> Self {
        Self {
            diagnostics: Vec::new(),
            error_count: AtomicUsize::new(0),
        }
    }

    /// Record a diagnostic.
    pub fn emit(&mut self, diag: Diagnostic) {
        if diag.severity == Severity::Error {
            self.error_count.fetch_add(1, Ordering::Relaxed);
        }
        self.diagnostics.push(diag);
    }

    /// Were any errors recorded?
    pub fn has_errors(&self) -> bool {
        self.error_count.load(Ordering::Relaxed) > 0
    }

    /// How many errors?
    pub fn error_count(&self) -> usize {
        self.error_count.load(Ordering::Relaxed)
    }

    /// All accumulated diagnostics.
    pub fn diagnostics(&self) -> &[Diagnostic] {
        &self.diagnostics
    }

    /// Render all diagnostics to a writer.
    pub fn render_all(&self, source: &str, writer: &mut dyn io::Write) -> io::Result<()> {
        for diag in &self.diagnostics {
            diag.render(source, writer)?;
        }
        Ok(())
    }

    /// Render all diagnostics to stderr.
    pub fn eprint_all(&self, source: &str) {
        for diag in &self.diagnostics {
            diag.eprint(source);
        }
    }
}

impl Default for DiagnosticEmitter {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_diagnostic_creation() {
        let err = Diagnostic::error("undefined variable");
        assert_eq!(err.severity, Severity::Error);
        assert_eq!(err.message, "undefined variable");

        let warn = Diagnostic::warning("unused variable");
        assert_eq!(warn.severity, Severity::Warning);

        let info = Diagnostic::info("type inferred as Num");
        assert_eq!(info.severity, Severity::Info);
    }

    #[test]
    fn test_diagnostic_labels() {
        let diag = Diagnostic::error("type mismatch")
            .with_label(0..3, "expected Num")
            .with_label(6..9, "found String");
        assert_eq!(diag.labels.len(), 2);
        assert_eq!(diag.labels[0].span, 0..3);
        assert_eq!(diag.labels[1].message, "found String");
    }

    #[test]
    fn test_diagnostic_render() {
        let source = "var x = true + 1";
        let diag = Diagnostic::error("type mismatch")
            .with_label(8..12, "this is Bool")
            .with_label(15..16, "expected Bool, found Num");

        let mut buf = Vec::new();
        diag.render(source, &mut buf).unwrap();
        let output = String::from_utf8(buf).unwrap();

        // Verify the output contains key elements
        assert!(output.contains("type mismatch"), "missing message");
        assert!(output.contains("this is Bool"), "missing label");
    }

    #[test]
    fn test_error_accumulation() {
        let mut emitter = DiagnosticEmitter::new();
        assert!(!emitter.has_errors());
        assert_eq!(emitter.error_count(), 0);

        emitter.emit(Diagnostic::warning("unused"));
        assert!(!emitter.has_errors()); // warnings don't count

        emitter.emit(Diagnostic::error("bad"));
        assert!(emitter.has_errors());
        assert_eq!(emitter.error_count(), 1);

        emitter.emit(Diagnostic::error("also bad"));
        assert_eq!(emitter.error_count(), 2);
        assert_eq!(emitter.diagnostics().len(), 3); // 1 warning + 2 errors
    }

    #[test]
    fn test_diagnostic_with_note() {
        let diag = Diagnostic::error("cannot call null")
            .with_label(0..4, "this is null")
            .with_note("only functions and methods can be called");
        assert_eq!(diag.notes.len(), 1);

        let mut buf = Vec::new();
        diag.render("null()", &mut buf).unwrap();
        let output = String::from_utf8(buf).unwrap();
        assert!(output.contains("only functions and methods can be called"));
    }
}
