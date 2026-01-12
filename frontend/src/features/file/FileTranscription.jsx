import { useState, useEffect } from "react";
import axios from "axios";
import { motion, AnimatePresence } from "framer-motion";
import toast from "react-hot-toast";
import { FiUploadCloud, FiFileText, FiDownload, FiCpu, FiCheckCircle, FiAlertCircle, FiPlus } from "react-icons/fi";

const BACKEND_URL = "http://localhost:8000";

export default function FileTranscription() {
    const [file, setFile] = useState(null);
    const [loading, setLoading] = useState(false);
    const [progress, setProgress] = useState(0);
    const [logs, setLogs] = useState([]);
    const [result, setResult] = useState({ formatted: "", full: "" });
    const [error, setError] = useState("");
    const [abortController, setAbortController] = useState(null);

    // Enhanced progress animation: slows down as it approaches 100%, never stops
    useEffect(() => {
        if (!loading) {
            // Don't reset to 0 immediately if we just finished (so users see the 100%)
            return;
        }

        setProgress(0);
        let currentProgress = 0;

        // We update every 300-600ms for a more "organic" feel
        const updateProgress = () => {
            if (!loading) return;

            // Asymptotic logic: always add a fraction of the remaining space to 100%
            // This mathematically ensures it never hits 100% (or even 99.9%)
            const remaining = 99.9 - currentProgress;
            const factor = currentProgress < 80 ? 0.08 : 0.02; // Slower near the end
            const increment = remaining * (Math.random() * factor + 0.01);

            currentProgress += increment;
            setProgress(Math.min(99.9, currentProgress)); // Safety clamp

            // Schedule next update with varied delay
            const nextDelay = 400 + Math.random() * 600;
            timeoutRef.current = setTimeout(updateProgress, nextDelay);
        };

        const timeoutRef = { current: setTimeout(updateProgress, 500) };

        return () => {
            if (timeoutRef.current) clearTimeout(timeoutRef.current);
        };
    }, [loading]);

    const addLog = (msg) => {
        const time = new Date().toLocaleTimeString();
        setLogs(prev => [...prev, { time, msg }].slice(-5));
    };

    const handleFileChange = (e) => {
        const selected = e.target.files[0];
        if (selected) {
            setFile(selected);
            setError("");
            setResult({ formatted: "", full: "" });
            setLogs([]);
            addLog(`Selected: ${selected.name}`);
        }
    };

    const handleTranscribe = async (withDiarization) => {
        if (!file) return toast.error("Please select a file first");

        setLoading(true);
        setError("");
        setLogs([]);
        addLog("Uploading file...");
        const toastId = toast.loading("Starting transcription...");
        const controller = new AbortController();
        setAbortController(controller);

        try {
            const formData = new FormData();
            formData.append("file", file);
            const endpoint = withDiarization ? "/transcribe-file-diarize" : "/transcribe-file";

            addLog("Processing on server...");
            const res = await axios.post(`${BACKEND_URL}${endpoint}`, formData, {
                headers: { "Content-Type": "multipart/form-data" },
                signal: controller.signal,
            });

            addLog("Completed!");
            setResult({
                formatted: res.data.formatted_text,
                full: res.data.full_text
            });
            setProgress(100);
            toast.success("Transcription complete!", { id: toastId });
        } catch (e) {
            if (axios.isCancel(e)) {
                addLog("Stopped.");
                toast.dismiss(toastId);
            } else {
                console.error(e);
                const errMsg = e.response?.data?.detail || "Transcription failed";
                setError(errMsg);
                addLog("Error occurred");
                toast.error(errMsg, { id: toastId });
            }
        } finally {
            setLoading(false);
            setAbortController(null);
        }
    };

    const handleStop = () => {
        if (abortController) {
            abortController.abort();
            setAbortController(null);
            setLoading(false);
            addLog("Transcription stopped.");
        }
    };

    const handleReset = () => {
        handleStop();
        setFile(null);
        setLoading(false);
        setProgress(0);
        setLogs([]);
        setResult({ formatted: "", full: "" });
        setError("");
        addLog("Session reset.");
    };

    const download = (content, ext, type) => {
        if (!content) return;
        const blob = new Blob([content], { type });
        const url = URL.createObjectURL(blob);
        const a = document.createElement("a");
        a.href = url;
        a.download = `transcript_${Date.now()}.${ext}`;
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        URL.revokeObjectURL(url);
    };

    return (
        <div className="page focus-page" style={{ height: '100%', display: 'flex', flexDirection: 'column', maxWidth: '800px', width: '100%', margin: '0 auto' }}>

            {/* Minimal Header */}
            <header style={{
                padding: '32px 0',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'space-between',
                borderBottom: '1px solid rgba(0,0,0,0.05)'
            }}>
                <div>
                    <h1 style={{ margin: 0, fontSize: '24px', fontWeight: '400', color: '#2c3e50', fontFamily: 'serif' }}>
                        {file ? file.name : "No File Selected"}
                    </h1>
                    <div style={{ fontSize: '13px', color: '#95a5a6', marginTop: '4px' }}>
                        {file ? "Ready to transcribe" : "Import an audio file to begin"}
                    </div>
                </div>

                <div style={{ display: 'flex', gap: '12px' }}>
                    <input type="file" id="focus-file-upload" accept="audio/*" onChange={handleFileChange} hidden />
                    <label
                        htmlFor="focus-file-upload"
                        style={{
                            padding: '8px 16px', borderRadius: '4px', border: '1px solid #2c3e50',
                            background: 'transparent', color: '#2c3e50', cursor: 'pointer',
                            fontSize: '13px', fontWeight: '500', fontFamily: 'sans-serif',
                            display: 'flex', alignItems: 'center'
                        }}
                    >
                        Import Audio
                    </label>

                    {file && (
                        <div style={{ display: 'flex', gap: '12px' }}>
                            {!loading ? (
                                <button
                                    onClick={() => handleTranscribe(false)}
                                    style={{
                                        padding: '8px 16px', borderRadius: '4px', border: 'none',
                                        background: '#2c3e50', color: '#fff', cursor: 'pointer',
                                        fontSize: '13px', fontWeight: '500', fontFamily: 'sans-serif'
                                    }}
                                >
                                    Transcribe
                                </button>
                            ) : (
                                <button
                                    onClick={handleStop}
                                    style={{
                                        padding: '8px 16px', borderRadius: '4px', border: '1px solid #e74c3c',
                                        background: 'transparent', color: '#e74c3c', cursor: 'pointer',
                                        fontSize: '13px', fontWeight: '500', fontFamily: 'sans-serif'
                                    }}
                                >
                                    Stop
                                </button>
                            )}

                            <button
                                onClick={handleReset}
                                disabled={loading}
                                style={{
                                    padding: '8px 16px', borderRadius: '4px', border: '1px solid #bdc3c7',
                                    background: 'transparent', color: '#7f8c8d', cursor: loading ? 'default' : 'pointer',
                                    fontSize: '13px', fontWeight: '500', fontFamily: 'sans-serif',
                                    opacity: loading ? 0.5 : 1
                                }}
                            >
                                Reset
                            </button>
                        </div>
                    )}
                </div>
            </header>

            {/* Document Area */}
            <div className="document-area" style={{
                flex: 1,
                overflowY: 'auto',
                padding: '40px 0',
                fontFamily: 'Georgia, serif',
                fontSize: '18px',
                lineHeight: '1.8',
                color: '#2c3e50'
            }}>

                {loading && (
                    <div style={{ color: '#95a5a6', fontStyle: 'italic', display: 'flex', flexDirection: 'column', gap: '12px' }}>
                        <div>Processing audio file...</div>
                        <div style={{ height: '2px', width: '100%', background: '#f0f2f5' }}>
                            <motion.div
                                animate={{ width: `${Math.min(100, progress)}%` }}
                                style={{ height: '100%', background: '#2c3e50' }}
                            />
                        </div>
                        <div style={{ fontSize: '12px' }}>{logs.length > 0 && logs[logs.length - 1].msg}</div>
                    </div>
                )}

                {!loading && result.formatted ? (
                    <div style={{ whiteSpace: 'pre-wrap' }}>
                        {result.formatted}
                    </div>
                ) : (
                    !loading && (
                        <div style={{ color: '#bdc3c7', fontStyle: 'italic' }}>
                            Transcript will appear here...
                        </div>
                    )
                )}
            </div>

        </div>
    );
}

