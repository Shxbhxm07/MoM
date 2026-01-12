import { useState, useEffect, useRef } from "react";  // ← Add useRef here
import axios from "axios";
import {
  FiMic,
  FiUploadCloud,
  FiChevronLeft,
  FiChevronRight,
  FiMoon,
  FiSun,
  FiDownload
} from "react-icons/fi";
import "./App.css";
import logo from "./assets/logo.svg";
import logoWhite from "./assets/logo-white.svg";

const BACKEND_URL = "http://localhost:8002";

const formatDuration = (sec) => {
  const s = Math.floor(sec || 0);
  const h = Math.floor(s / 3600);
  const m = Math.floor((s % 3600) / 60);
  const r = s % 60;
  if (h > 0) return `${String(h).padStart(2, "0")}:${String(m).padStart(2, "0")}:${String(r).padStart(2, "0")}`;
  return `${String(m).padStart(2, "0")}:${String(r).padStart(2, "0")}`;
};

const formatTimestamp = (seconds) => {
  const mins = Math.floor(seconds / 60);
  const secs = Math.floor(seconds % 60);
  return `${String(mins).padStart(2, "0")}:${String(secs).padStart(2, "0")}`;
};

function App() {
  const [activeTab, setActiveTab] = useState("file");
  const [sidebarCollapsed, setSidebarCollapsed] = useState(false);
  const [darkMode, setDarkMode] = useState(() => {
    return localStorage.getItem("theme") === "dark";
  });

  useEffect(() => {
    if (darkMode) {
      document.body.classList.add("dark-mode");
      localStorage.setItem("theme", "dark");
    } else {
      document.body.classList.remove("dark-mode");
      localStorage.setItem("theme", "light");
    }
  }, [darkMode]);

  const [liveStatus, setLiveStatus] = useState({
    is_recording: false,
    duration: 0,
    word_count: 0,
    total_segments: 0,
    latest_text: "",
    transcript: "",
    audio_level: 0,
  });

  useEffect(() => {
    if (activeTab !== "live") return;
    const id = setInterval(async () => {
      try {
        const res = await axios.get(`${BACKEND_URL}/live/status`);
        setLiveStatus(res.data);
      } catch (err) {
        console.error(err);
      }
    }, 1000);
    return () => clearInterval(id);
  }, [activeTab]);

  const handleLiveStart = async () => {
    try {
      await axios.post(`${BACKEND_URL}/live/start`);
    } catch (e) {
      console.error(e);
      alert(e.response?.data?.detail || "Failed to start recording");
    }
  };

  const handleLiveStop = async () => {
    try {
      await axios.post(`${BACKEND_URL}/live/stop`);
    } catch (e) {
      console.error(e);
    }
  };

  const handleLiveReset = async () => {
    try {
      await axios.post(`${BACKEND_URL}/live/reset`);
      setLiveStatus((s) => ({
        ...s,
        duration: 0,
        word_count: 0,
        total_segments: 0,
        latest_text: "",
        transcript: "",
        audio_level: 0,
      }));
    } catch (e) {
      console.error(e);
    }
  };

  const handleLiveDiarize = async () => {
    try {
      const res = await axios.post(`${BACKEND_URL}/live/diarize`);
      setLiveStatus((s) => ({
        ...s,
        transcript: res.data.formatted_transcript,
        word_count: count_words(res.data.formatted_transcript),
      }));
    } catch (e) {
      console.error(e);
      alert(e.response?.data?.detail || "Diarization failed");
    }
  };

  return (
    <div className={`app-shell ${darkMode ? "dark" : ""}`}>
      <aside className={`sidebar ${sidebarCollapsed ? "collapsed" : ""}`}>
        <div className="sidebar-header">
          <div className="logo-container">
            <img src={darkMode ? logoWhite : logo} alt="AngelBot Logo" style={{ height: '32px', width: 'auto' }} />
          </div>
          {!sidebarCollapsed && (
            <div className="logo-text">
              <div className="logo-title">ANGELBOT.AI</div>
              <div className="logo-subtitle">Transcription Studio</div>
            </div>
          )}
          <button
            className="sidebar-toggle"
            onClick={() => setSidebarCollapsed((v) => !v)}
          >
            {sidebarCollapsed ? <FiChevronRight /> : <FiChevronLeft />}
          </button>
        </div>

        <nav className="sidebar-nav">
          <button
            className={`nav-item ${activeTab === "live" ? "active" : ""}`}
            onClick={() => setActiveTab("live")}
          >
            <FiMic className="nav-icon" />
            {!sidebarCollapsed && <span>Live Session</span>}
          </button>
          <button
            className={`nav-item ${activeTab === "file" ? "active" : ""}`}
            onClick={() => setActiveTab("file")}
          >
            <FiUploadCloud className="nav-icon" />
            {!sidebarCollapsed && <span>File Transcription</span>}
          </button>
        </nav>

        <div style={{ marginTop: 'auto', padding: '16px' }}>
          <button
            className="nav-item"
            onClick={() => setDarkMode(!darkMode)}
            style={{ justifyContent: sidebarCollapsed ? 'center' : 'flex-start' }}
          >
            {darkMode ? <FiSun className="nav-icon" /> : <FiMoon className="nav-icon" />}
            {!sidebarCollapsed && <span>{darkMode ? "Light Mode" : "Dark Mode"}</span>}
          </button>
        </div>
      </aside>

      <main className="main">
        <div style={{ display: activeTab === "live" ? "block" : "none" }}>
          <LiveSession
            status={liveStatus}
            onStart={handleLiveStart}
            onStop={handleLiveStop}
            onReset={handleLiveReset}
            onDiarize={handleLiveDiarize}
          />
        </div>

        <div style={{ display: activeTab === "file" ? "block" : "none" }}>
          <FileTranscription />
        </div>

        <footer className="footer">
          © 2026 AngelBot.AI · Whisper · NeMo · Llama · React
        </footer>
      </main>
    </div>
  );
}

// Helper function for word counting
const count_words = (text) => {
  return text ? text.split(/\s+/).filter(Boolean).length : 0;
};

function LiveSession({ status, onStart, onStop, onReset, onDiarize }) {
  const { is_recording, duration, word_count, total_segments, transcript, audio_level, latest_text } = status;
  const transcriptEndRef = useRef(null);

  // Auto-scroll to bottom when new text arrives
  useEffect(() => {
    if (transcriptEndRef.current) {
      transcriptEndRef.current.scrollIntoView({ behavior: 'smooth' });
    }
  }, [transcript]);

  return (
    <div className="page">
      <header className="page-header">
        <h1>Live Session</h1>
        <p>Real-time speech transcription with Hindi, English & Hinglish support (~2s latency)</p>
      </header>

      <section className="card controls-card">
        <div className="controls-row">
          <button className="btn btn-primary" onClick={onStart} disabled={is_recording}>
            🎤 Start Recording
          </button>
          <button className="btn btn-secondary" onClick={onStop} disabled={!is_recording}>
            ⏹️ Stop Recording
          </button>
          <button className="btn btn-secondary" onClick={onDiarize} disabled={is_recording}>
            👥 Diarize
          </button>
          <button className="btn btn-ghost" onClick={onReset} disabled={is_recording}>
            🔄 Reset
          </button>
          <span className={`recording-pill ${is_recording ? "on" : "off"}`}>
            {is_recording ? "🔴 Recording..." : "⚪ Idle"}
          </span>
        </div>

        <div className="audio-row">
          <span className="audio-label">Audio Level</span>
          <div className="audio-bar">
            <div className="audio-bar-fill" style={{
              width: `${(audio_level * 100).toFixed(2)}%`,
              background: audio_level > 0.8 ? 'linear-gradient(90deg, #e74c3c, #c0392b)' : 
                         audio_level > 0.5 ? 'linear-gradient(90deg, #f39c12, #e67e22)' : 
                         'linear-gradient(90deg, #27ae60, #2ecc71)'
            }} />
            <span className="audio-bar-text">{(audio_level * 100).toFixed(2)}%</span>
          </div>
        </div>

        {/* Show latest text in real-time */}
        {latest_text && is_recording && (
          <div className="latest-text-bubble" style={{
            marginTop: '16px',
            padding: '12px 16px',
            background: 'var(--primary-light)',
            borderRadius: '8px',
            borderLeft: '4px solid var(--primary)',
            fontSize: '14px',
            color: 'var(--text-main)',
            animation: 'fadeIn 0.3s ease-in'
          }}>
            <strong>Latest:</strong> {latest_text}
          </div>
        )}
      </section>

      <section className="card overview-card">
        <h2 className="section-title">Live Session Overview</h2>
        <div className="overview-grid">
          <div className="overview-item">
            <span className="overview-label">Duration</span>
            <span className="overview-value">{formatDuration(duration)}</span>
          </div>
          <div className="overview-item">
            <span className="overview-label">Words</span>
            <span className="overview-value">{word_count}</span>
          </div>
          <div className="overview-item">
            <span className="overview-label">Segments</span>
            <span className="overview-value">{total_segments}</span>
          </div>
        </div>
      </section>

      <section className="card transcript-card">
        <div className="transcript-header">
          <h2 className="section-title">Live Transcript</h2>
          {is_recording && (
            <span style={{ fontSize: '12px', color: 'var(--text-muted)' }}>
              ⚡ ~2s latency | 🌐 Multi-language
            </span>
          )}
        </div>
        <div className="transcript-box" style={{ maxHeight: '500px', overflowY: 'auto' }}>
          {transcript ? (
            <>
              <pre style={{ margin: 0, whiteSpace: 'pre-wrap', fontFamily: 'Noto Sans Devanagari, Arial Unicode MS, sans-serif' }}>{transcript}</pre>
              <div ref={transcriptEndRef} />
            </>
          ) : (
            <span className="transcript-placeholder">
              Click <strong>🎤 Start Recording</strong> to begin real-time transcription.
              <br />
              <small style={{ color: 'var(--text-muted)', marginTop: '8px', display: 'block' }}>
                Supports: English, Hindi, and Hinglish (mixed)
              </small>
            </span>
          )}
        </div>

        {transcript && !is_recording && (
          <div className="export-row" style={{ marginTop: '16px', display: 'flex', gap: '8px' }}>
            <button className="export-btn" onClick={() => {
              const blob = new Blob([transcript], { type: "text/plain; charset=utf-8" });
              const url = URL.createObjectURL(blob);
              const a = document.createElement("a");
              a.href = url;
              a.download = `live_session_${Date.now()}.txt`;
              document.body.appendChild(a);
              a.click();
              a.remove();
              URL.revokeObjectURL(url);
            }}>💾 Save as TXT</button>
            <button className="export-btn" onClick={() => {
              const md = `# Live Session Transcript\n\n${transcript}`;
              const blob = new Blob([md], { type: "text/markdown; charset=utf-8" });
              const url = URL.createObjectURL(blob);
              const a = document.createElement("a");
              a.href = url;
              a.download = `live_session_${Date.now()}.md`;
              document.body.appendChild(a);
              a.click();
              a.remove();
              URL.revokeObjectURL(url);
            }}>📝 Save as Markdown</button>
          </div>
        )}
      </section>
    </div>
  );
}

function FileTranscription() {
  const [file, setFile] = useState(null);
  const [loading, setLoading] = useState(false);
  const [progress, setProgress] = useState(0);
  const [logs, setLogs] = useState([]);
  const [result, setResult] = useState(null);
  const [error, setError] = useState("");
  const [abortController, setAbortController] = useState(null);

  useEffect(() => {
    if (!loading) return;
    setProgress(0);
    let currentProgress = 0;

    const updateProgress = () => {
      if (!loading) return;
      const remaining = 99.9 - currentProgress;
      const factor = currentProgress < 80 ? 0.08 : 0.02;
      const increment = remaining * (Math.random() * factor + 0.01);
      currentProgress += increment;
      setProgress(Math.min(99.9, currentProgress));
      timeoutRef.current = setTimeout(updateProgress, 400 + Math.random() * 600);
    };

    const timeoutRef = { current: setTimeout(updateProgress, 500) };
    return () => {
      if (timeoutRef.current) clearTimeout(timeoutRef.current);
    };
  }, [loading]);

  const appendLog = (msg) => {
    const stamp = new Date().toLocaleTimeString();
    setLogs((prev) => [...prev, `[${stamp}] ${msg}`].slice(-3));
  };

  const handleFileChange = (e) => {
    const f = e.target.files[0] || null;
    setFile(f);
    setError("");
    setResult(null);
    setLogs([]);
    if (f) appendLog(`Selected file: ${f.name}`);
  };

  const handleFullPipeline = async () => {
    if (!file) {
      setError("Please select an audio file first.");
      return;
    }

    setLoading(true);
    setError("");
    setResult(null);
    setLogs([]);
    appendLog("Uploading file to server…");

    const controller = new AbortController();
    setAbortController(controller);

    try {
      const formData = new FormData();
      formData.append("file", file);

      appendLog("Running full pipeline (transcribe + diarize + summarize)…");
      
      const res = await axios.post(
        `${BACKEND_URL}/full-pipeline?summarize_output=true`,
        formData,
        {
          headers: { "Content-Type": "multipart/form-data" },
          signal: controller.signal,
        }
      );

      appendLog("Pipeline completed successfully!");
      setResult(res.data);
      setProgress(100);
    } catch (e) {
      if (axios.isCancel(e)) {
        appendLog("Processing stopped by user.");
      } else {
        console.error(e);
        setError(e.response?.data?.detail || "Pipeline failed");
        appendLog("Error during processing.");
      }
    } finally {
      setLoading(false);
      setAbortController(null);
      setTimeout(() => setProgress(0), 800);
    }
  };

  const handleStop = () => {
    if (abortController) {
      abortController.abort();
      setAbortController(null);
      setLoading(false);
    }
  };

  const handleReset = () => {
    handleStop();
    setFile(null);
    setLoading(false);
    setProgress(0);
    setLogs([]);
    setResult(null);
    setError("");
    const fileInput = document.querySelector('input[type="file"]');
    if (fileInput) fileInput.value = "";
  };

  // Generate formatted diarized transcript from backend formatted_transcript
  const generateDiarizedTranscript = () => {
    if (!result) return "";
    
    // Use the formatted_transcript from backend if available
    if (result.formatted_transcript) {
      return result.formatted_transcript;
    }
    
    // Fallback: try to format from diarization data
    if (result.diarization && result.diarization.segments) {
      return result.diarization.segments.map(seg => {
        const timestamp = formatTimestamp(seg.start || 0);
        const speaker = (seg.speaker || "SPEAKER_00").toUpperCase();
        const text = seg.text || "";
        return `[${timestamp}] ${speaker}: ${text}`;
      }).join("\n");
    }
    
    // Final fallback: return raw transcript
    return result.transcription?.text || "No transcript available";
  };

  // Generate complete document for download
  const generateCompleteDocument = () => {
    if (!result) return "";
    
    const diarized = generateDiarizedTranscript();
    const summary = result.summary?.analysis || result.summary?.summary || "No summary available";
    const rawTranscript = result.transcription?.text || "No transcript available";
    const numSpeakers = result.diarization?.num_speakers || 
                       result.diarization?.speakers?.length || 
                       0;
    
    // Calculate duration from transcription segments if available
    let duration = "N/A";
    if (result.transcription?.segments && result.transcription.segments.length > 0) {
      const lastSegment = result.transcription.segments[result.transcription.segments.length - 1];
      duration = formatDuration(lastSegment.end);
    }
    
    return `================================================================================
ANGELBOT.AI - COMPLETE TRANSCRIPTION REPORT
================================================================================
Generated: ${new Date().toLocaleString()}
File: ${file?.name || "Unknown"}
Speakers Detected: ${numSpeakers}
Total Duration: ${duration}

================================================================================
SECTION 1: DIARIZED TRANSCRIPT (Speaker-Labeled with Timestamps)
================================================================================

${diarized}

================================================================================
SECTION 2: AI SUMMARY & ANALYSIS
================================================================================

${summary}

================================================================================
SECTION 3: RAW TRANSCRIPT (No Speaker Labels)
================================================================================

${rawTranscript}

================================================================================
Generated by AngelBot.AI
© 2026 AngelBot.AI · Whisper · NeMo · Llama · React
================================================================================
`;
  };

  const downloadFile = (content, fileName, mimeType) => {
    const blob = new Blob([content], { type: mimeType + "; charset=utf-8" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = fileName;
    document.body.appendChild(a);
    a.click();
    a.remove();
    URL.revokeObjectURL(url);
  };

  const handleDownloadTxt = () => {
    if (!result) return;
    const content = generateCompleteDocument();
    downloadFile(content, `transcript_complete_${Date.now()}.txt`, "text/plain");
  };

  const handleDownloadMd = () => {
    if (!result) return;
    
    const diarized = generateDiarizedTranscript();
    const summary = result.summary?.analysis || result.summary?.summary || "No summary available";
    const rawTranscript = result.transcription?.text || "No transcript available";
    const numSpeakers = result.diarization?.num_speakers || 
                       result.diarization?.speakers?.length || 
                       0;
    
    let duration = "N/A";
    if (result.transcription?.segments && result.transcription.segments.length > 0) {
      const lastSegment = result.transcription.segments[result.transcription.segments.length - 1];
      duration = formatDuration(lastSegment.end);
    }
    
    const markdown = `# AngelBot.AI - Complete Transcription Report

**Generated:** ${new Date().toLocaleString()}  
**File:** ${file?.name || "Unknown"}  
**Speakers Detected:** ${numSpeakers}  
**Total Duration:** ${duration}

---

## Section 1: Diarized Transcript (Speaker-Labeled with Timestamps)

\`\`\`
${diarized}
\`\`\`

---

## Section 2: AI Summary & Analysis

${summary}

---

## Section 3: Raw Transcript (No Speaker Labels)

\`\`\`
${rawTranscript}
\`\`\`

---

*Generated by AngelBot.AI*  
© 2026 AngelBot.AI · Whisper · NeMo · Llama · React
`;
    
    downloadFile(markdown, `transcript_complete_${Date.now()}.md`, "text/markdown");
  };

  const handleDownloadPdf = async () => {
    if (!result) return;
    
    const diarized = generateDiarizedTranscript();
    const summary = result.summary?.analysis || result.summary?.summary || "No summary available";
    const rawTranscript = result.transcription?.text || "No transcript available";
    const numSpeakers = result.diarization?.num_speakers || 
                       result.diarization?.speakers?.length || 
                       0;
    
    try {
      const payload = {
        formatted_transcript: diarized,
        summary: summary,
        raw_transcript: rawTranscript,
        filename: file?.name || "transcript",
        speaker_count: numSpeakers
      };
      
      const res = await axios.post(
        `${BACKEND_URL}/export/complete-pdf`,
        payload,
        { 
          responseType: "blob",
          headers: { 'Content-Type': 'application/json' }
        }
      );
      
      const url = window.URL.createObjectURL(res.data);
      const a = document.createElement("a");
      a.href = url;
      a.download = `transcript_complete_${Date.now()}.pdf`;
      document.body.appendChild(a);
      a.click();
      a.remove();
      window.URL.revokeObjectURL(url);
    } catch (e) {
      console.error(e);
      setError("PDF export failed: " + (e.response?.data?.detail || e.message));
    }
  };

  return (
    <div className="page file-page">
      <div className="file-inner">
        <header className="page-header">
          <h1>File Transcription</h1>
          <p>Upload an audio file for full AI pipeline: transcription, diarization, and summarization.</p>
        </header>

        <section className="card">
          <label className="file-drop">
            <input type="file" accept="audio/*" onChange={handleFileChange} />
            <div className="file-drop-title">{file ? "Change audio file" : "Choose audio file"}</div>
            <div className="file-drop-sub">Supported formats: MP3, WAV, M4A, FLAC, OGG</div>
            {file && <div className="file-selected-name">{file.name}</div>}
          </label>
        </section>

        <section className="card">
          <h2 className="section-title">Processing</h2>
          <div className="file-progress-container">
            <div className="progress-bar">
              <div className="progress-bar-fill" style={{
                width: `${Math.min(100, progress)}%`,
                background: progress >= 100 ? 'linear-gradient(90deg, #27ae60, #2ecc71)' : progress > 50 ? 'linear-gradient(90deg, #3498db, #2980b9)' : 'linear-gradient(90deg, #9b59b6, #8e44ad)'
              }} />
              <span className="progress-bar-text">{progress.toFixed(2)}%</span>
            </div>
            <div className="file-logs">
              {logs.map((line, idx) => <div className="file-log-line" key={idx}>{line}</div>)}
              {!logs.length && <div className="file-log-line file-log-muted">Waiting for processing job…</div>}
            </div>
          </div>
        </section>

        <section className="card">
          <div className="file-buttons-row">
            {!loading ? (
              <button className="btn btn-primary" onClick={handleFullPipeline} disabled={!file}>Run Full Pipeline</button>
            ) : (
              <button className="btn btn-secondary" onClick={handleStop} style={{ borderColor: "#e74c3c", color: "#e74c3c" }}>Stop Processing</button>
            )}
            <button className="btn btn-ghost" onClick={handleReset} disabled={loading || (!file && !result)}>Reset</button>
          </div>
          {error && <div className="error-text">{error}</div>}
        </section>

        {result && (
          <>
            <section className="card transcript-card file-transcript-card">
              <div className="transcript-header">
                <h2 className="section-title">Diarized Transcript</h2>
                <span className="transcript-status">
                  {result.diarization?.num_speakers || result.diarization?.speakers?.length || 0} speakers detected
                </span>
              </div>
              <div className="transcript-box">
                <pre style={{ margin: 0, whiteSpace: 'pre-wrap', wordWrap: 'break-word', fontFamily: 'Noto Sans Devanagari, SF Mono, Monaco, monospace' }}>
                  {generateDiarizedTranscript() || "No diarized transcript available"}
                </pre>
              </div>
            </section>

            {result.summary && (
              <section className="card">
                <h2 className="section-title">AI Summary & Analysis</h2>
                <div className="transcript-box">
                  <pre style={{ margin: 0, whiteSpace: 'pre-wrap', wordWrap: 'break-word' }}>
                    {result.summary.analysis || result.summary.summary || "No summary available"}
                  </pre>
                </div>
              </section>
            )}

            <section className="card">
              <div className="export-row">
                <button className="export-btn" onClick={handleDownloadTxt}>
                  <FiDownload /> Complete Report (TXT)
                </button>
                <button className="export-btn" onClick={handleDownloadMd}>
                  <FiDownload /> Complete Report (Markdown)
                </button>
                <button className="export-btn" onClick={handleDownloadPdf}>
                  <FiDownload /> Complete Report (PDF)
                </button>
              </div>
              <p style={{ marginTop: '12px', fontSize: '13px', color: 'var(--text-muted)', lineHeight: '1.5' }}>
                📦 Downloads include: Diarized transcript with speaker labels & timestamps + AI Summary + Raw transcript
              </p>
            </section>
          </>
        )}
      </div>
    </div>
  );
}

export default App;