// import { useState, useEffect, useRef } from "react";
// import axios from "axios";
// import { motion } from "framer-motion";
// import toast from "react-hot-toast";
// import { FiMic, FiSquare, FiRefreshCw, FiLayers, FiPlus } from "react-icons/fi";

// const BACKEND_URL = "http://localhost:8000";

// const formatDuration = (sec) => {
//     const s = Math.floor(sec || 0);
//     const h = Math.floor(s / 3600);
//     const m = Math.floor((s % 3600) / 60);
//     const r = s % 60;
//     if (h > 0) return `${String(h).padStart(2, "0")}:${String(m).padStart(2, "0")}:${String(r).padStart(2, "0")}`;
//     return `${String(m).padStart(2, "0")}:${String(r).padStart(2, "0")}`;
// };

// export default function LiveSession() {
//     const [status, setStatus] = useState({
//         is_recording: false,
//         duration: 0,
//         word_count: 0,
//         total_segments: 0,
//         latest_text: "",
//         transcript: "",
//         audio_level: 0,
//     });

//     // Poll status
//     useEffect(() => {
//         const id = setInterval(async () => {
//             try {
//                 const res = await axios.get(`${BACKEND_URL}/live/status`);
//                 setStatus(res.data);
//             } catch (err) {
//                 console.error(err);
//             }
//         }, 1000);
//         return () => clearInterval(id);
//     }, []);

//     const handleStart = async () => {
//         try {
//             await axios.post(`${BACKEND_URL}/live/start`);
//             toast.success("Recording started");
//         } catch (e) {
//             toast.error("Failed to start recording");
//         }
//     };

//     const handleStop = async () => {
//         try {
//             await axios.post(`${BACKEND_URL}/live/stop`);
//             toast.success("Recording stopped");
//         } catch (e) {
//             toast.error("Failed to stop recording");
//         }
//     };

//     const handleReset = async () => {
//         try {
//             await axios.post(`${BACKEND_URL}/live/reset`);
//             setStatus((s) => ({
//                 ...s,
//                 duration: 0,
//                 word_count: 0,
//                 total_segments: 0,
//                 latest_text: "",
//                 transcript: "",
//                 audio_level: 0,
//             }));
//             toast.success("Session reset");
//         } catch (e) {
//             toast.error("Failed to reset session");
//         }
//     };

//     const handleDiarize = async () => {
//         const toastId = toast.loading("Diarizing...");
//         try {
//             const res = await axios.post(`${BACKEND_URL}/live/diarize`);
//             setStatus((s) => ({
//                 ...s,
//                 transcript: res.data.formatted_text,
//                 word_count: res.data.full_text.split(/\s+/).filter(Boolean).length,
//             }));
//             toast.success("Diarization complete", { id: toastId });
//         } catch (e) {
//             toast.error(e.response?.data?.detail || "Diarization failed", { id: toastId });
//         }
//     };

//     const levelPct = Math.min(100, Math.round(status.audio_level * 100));

//     return (
//         <div className="page focus-page" style={{ height: '100%', display: 'flex', flexDirection: 'column', maxWidth: '800px', width: '100%', margin: '0 auto' }}>

//             {/* Minimal Header */}
//             <header style={{
//                 padding: '32px 0',
//                 display: 'flex',
//                 alignItems: 'center',
//                 justifyContent: 'space-between',
//                 borderBottom: '1px solid rgba(0,0,0,0.05)'
//             }}>
//                 <div>
//                     <h1 style={{ margin: 0, fontSize: '24px', fontWeight: '400', color: '#2c3e50', fontFamily: 'serif' }}>Untitled Session</h1>
//                     <div style={{ fontSize: '13px', color: '#95a5a6', marginTop: '4px' }}>
//                         {status.is_recording ? "Recording..." : "Last edited just now"} &bull; {status.word_count} words
//                     </div>
//                 </div>

//                 <div style={{ display: 'flex', gap: '12px' }}>
//                     {!status.is_recording ? (
//                         <button
//                             onClick={handleStart}
//                             style={{
//                                 padding: '8px 16px', borderRadius: '4px', border: '1px solid #2c3e50',
//                                 background: 'transparent', color: '#2c3e50', cursor: 'pointer',
//                                 fontSize: '13px', fontWeight: '500', fontFamily: 'sans-serif'
//                             }}
//                         >
//                             Record
//                         </button>
//                     ) : (
//                         <button
//                             onClick={handleStop}
//                             style={{
//                                 padding: '8px 16px', borderRadius: '4px', border: 'none',
//                                 background: '#c0392b', color: '#fff', cursor: 'pointer',
//                                 fontSize: '13px', fontWeight: '500', fontFamily: 'sans-serif'
//                             }}
//                         >
//                             Stop
//                         </button>
//                     )}
//                 </div>
//             </header>

//             {/* Audio Level Bar */}
//             {status.is_recording && (
//                 <div style={{
//                     padding: '20px 0',
//                     borderBottom: '1px solid rgba(0,0,0,0.05)'
//                 }}>
//                     <div style={{
//                         display: 'flex',
//                         alignItems: 'center',
//                         justifyContent: 'space-between',
//                         marginBottom: '8px'
//                     }}>
//                         <span style={{
//                             fontSize: '12px',
//                             fontWeight: '600',
//                             color: '#95a5a6',
//                             textTransform: 'uppercase',
//                             letterSpacing: '0.05em'
//                         }}>
//                             Audio Level
//                         </span>
//                         <span style={{
//                             fontSize: '13px',
//                             fontWeight: '700',
//                             color: '#2c3e50',
//                             fontFamily: 'monospace'
//                         }}>
//                             {(status.audio_level * 100).toFixed(2)}%
//                         </span>
//                     </div>
//                     <div style={{
//                         width: '100%',
//                         height: '8px',
//                         background: 'rgba(0,0,0,0.05)',
//                         borderRadius: '4px',
//                         overflow: 'hidden',
//                         position: 'relative'
//                     }}>
//                         <motion.div
//                             initial={{ width: 0 }}
//                             animate={{ width: `${Math.min(100, status.audio_level * 100)}%` }}
//                             transition={{ duration: 0.3, ease: 'easeOut' }}
//                             style={{
//                                 height: '100%',
//                                 background: status.audio_level > 0.8
//                                     ? 'linear-gradient(90deg, #e74c3c, #c0392b)'
//                                     : status.audio_level > 0.5
//                                         ? 'linear-gradient(90deg, #f39c12, #e67e22)'
//                                         : 'linear-gradient(90deg, #27ae60, #2ecc71)',
//                                 borderRadius: '4px',
//                                 boxShadow: status.audio_level > 0.1 ? '0 0 8px rgba(46, 204, 113, 0.4)' : 'none'
//                             }}
//                         />
//                     </div>
//                 </div>
//             )}

//             {/* Document Area */}
//             <div className="document-area" style={{
//                 flex: 1,
//                 overflowY: 'auto',
//                 padding: '40px 0',
//                 fontFamily: 'Georgia, serif',
//                 fontSize: '18px',
//                 lineHeight: '1.8',
//                 color: '#2c3e50'
//             }}>
//                 {status.transcript ? (
//                     <div style={{ whiteSpace: 'pre-wrap' }}>
//                         {status.transcript}
//                         {status.is_recording && (
//                             <span className="cursor-blink" style={{ display: 'inline-block', width: '2px', height: '1.2em', background: '#2c3e50', marginLeft: '2px', verticalAlign: 'text-bottom' }} />
//                         )}
//                     </div>
//                 ) : (
//                     <div style={{ color: '#bdc3c7', fontStyle: 'italic' }}>
//                         Start recording to begin writing...
//                     </div>
//                 )}
//             </div>

//         </div>
//     );
// }


// function StatRow({ label, value }) {
//     return (
//         <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', fontSize: '14px' }}>
//             <span style={{ color: '#5d6d7e' }}>{label}</span>
//             <span style={{ fontWeight: '600', color: '#2c3e50' }}>{value}</span>
//         </div>
//     );
// }

// function OverviewItem({ label, value }) {
//     return (
//         <div className="overview-item">
//             <span className="overview-label">{label}</span>
//             <span className="overview-value">{value}</span>
//         </div>
//     );
// }


import { useState, useEffect, useRef } from "react";
import axios from "axios";
import { motion } from "framer-motion";
import toast from "react-hot-toast";
import { FiMic, FiSquare, FiRefreshCw, FiLayers, FiPlus, FiDownload, FiFileText, FiFile } from "react-icons/fi";

const BACKEND_URL = "http://localhost:8000";

const formatDuration = (sec) => {
    const s = Math.floor(sec || 0);
    const h = Math.floor(s / 3600);
    const m = Math.floor((s % 3600) / 60);
    const r = s % 60;
    if (h > 0) return `${String(h).padStart(2, "0")}:${String(m).padStart(2, "0")}:${String(r).padStart(2, "0")}`;
    return `${String(m).padStart(2, "0")}:${String(r).padStart(2, "0")}`;
};

export default function LiveSession() {
    const [status, setStatus] = useState({
        is_recording: false,
        duration: 0,
        word_count: 0,
        total_segments: 0,
        latest_text: "",
        transcript: "",
        audio_level: 0,
    });

    // Poll status
    useEffect(() => {
        const id = setInterval(async () => {
            try {
                const res = await axios.get(`${BACKEND_URL}/live/status`);
                setStatus(res.data);
            } catch (err) {
                console.error(err);
            }
        }, 1000);
        return () => clearInterval(id);
    }, []);

    const handleStart = async () => {
        try {
            await axios.post(`${BACKEND_URL}/live/start`);
            toast.success("Recording started");
        } catch (e) {
            toast.error("Failed to start recording");
        }
    };

    const handleStop = async () => {
        try {
            await axios.post(`${BACKEND_URL}/live/stop`);
            toast.success("Recording stopped");
        } catch (e) {
            toast.error("Failed to stop recording");
        }
    };

    const handleReset = async () => {
        try {
            await axios.post(`${BACKEND_URL}/live/reset`);
            setStatus((s) => ({
                ...s,
                duration: 0,
                word_count: 0,
                total_segments: 0,
                latest_text: "",
                transcript: "",
                audio_level: 0,
            }));
            toast.success("Session reset");
        } catch (e) {
            toast.error("Failed to reset session");
        }
    };

    const handleDiarize = async () => {
        const toastId = toast.loading("Diarizing...");
        try {
            const res = await axios.post(`${BACKEND_URL}/live/diarize`);
            setStatus((s) => ({
                ...s,
                transcript: res.data.formatted_text,
                word_count: res.data.full_text.split(/\s+/).filter(Boolean).length,
            }));
            toast.success("Diarization complete", { id: toastId });
        } catch (e) {
            toast.error(e.response?.data?.detail || "Diarization failed", { id: toastId });
        }
    };

    const download = (content, ext, type) => {
        if (!content) return;
        const blob = new Blob([content], { type });
        const url = URL.createObjectURL(blob);
        const a = document.createElement("a");
        a.href = url;
        a.download = `live_session_${Date.now()}.${ext}`;
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        URL.revokeObjectURL(url);
    };

    const handleExportTxt = () => {
        if (!status.transcript) return toast.error("Nothing to export");
        download(status.transcript, "txt", "text/plain");
        toast.success("Exported to TXT");
    };

    const handleExportMd = () => {
        if (!status.transcript) return toast.error("Nothing to export");
        const mdContent = `# Live Session Transcript\n\n${status.transcript}`;
        download(mdContent, "md", "text/markdown");
        toast.success("Exported to Markdown");
    };

    const handleExportPdf = async () => {
        if (!status.transcript) return toast.error("Nothing to export");
        const toastId = toast.loading("Generating PDF...");
        try {
            const res = await axios.post(`${BACKEND_URL}/export/pdf`, {
                content: status.transcript,
                title: "Live Session Transcript"
            }, {
                responseType: 'blob'
            });

            const url = window.URL.createObjectURL(new Blob([res.data]));
            const link = document.createElement('a');
            link.href = url;
            link.setAttribute('download', `live_session_${Date.now()}.pdf`);
            document.body.appendChild(link);
            link.click();
            link.remove();
            toast.success("Exported to PDF", { id: toastId });
        } catch (e) {
            console.error(e);
            toast.error("PDF generation failed", { id: toastId });
        }
    };

    const levelPct = Math.min(100, Math.round(status.audio_level * 100));

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
                    <h1 style={{ margin: 0, fontSize: '24px', fontWeight: '400', color: '#2c3e50', fontFamily: 'serif' }}>Untitled Session</h1>
                    <div style={{ fontSize: '13px', color: '#95a5a6', marginTop: '4px' }}>
                        {status.is_recording ? "Recording..." : "Last edited just now"} &bull; {status.word_count} words
                    </div>
                </div>

                <div style={{ display: 'flex', gap: '12px' }}>
                    {!status.is_recording ? (
                        <button
                            onClick={handleStart}
                            style={{
                                padding: '8px 16px', borderRadius: '4px', border: '1px solid #2c3e50',
                                background: 'transparent', color: '#2c3e50', cursor: 'pointer',
                                fontSize: '13px', fontWeight: '500', fontFamily: 'sans-serif'
                            }}
                        >
                            Record
                        </button>
                    ) : (
                        <button
                            onClick={handleStop}
                            style={{
                                padding: '8px 16px', borderRadius: '4px', border: 'none',
                                background: '#c0392b', color: '#fff', cursor: 'pointer',
                                fontSize: '13px', fontWeight: '500', fontFamily: 'sans-serif'
                            }}
                        >
                            Stop
                        </button>
                    )}

                    {/* Diarize button - only show when not recording and transcript exists */}
                    {!status.is_recording && status.transcript && (
                        <button
                            onClick={handleDiarize}
                            style={{
                                padding: '8px 16px', borderRadius: '4px', border: '1px solid #27ae60',
                                background: 'transparent', color: '#27ae60', cursor: 'pointer',
                                fontSize: '13px', fontWeight: '500', fontFamily: 'sans-serif',
                                display: 'flex', alignItems: 'center', gap: '6px'
                            }}
                        >
                            <FiLayers size={14} />
                            Diarize
                        </button>
                    )}



                    {/* Export Actions - only show when not recording and transcript exists */}
                    {!status.is_recording && status.transcript && (
                        <>
                            <button
                                onClick={handleExportTxt}
                                style={{
                                    padding: '8px 16px', borderRadius: '4px', border: '1px solid #95a5a6',
                                    background: 'transparent', color: '#7f8c8d', cursor: 'pointer',
                                    fontSize: '13px', fontWeight: '500', fontFamily: 'sans-serif',
                                    display: 'flex', alignItems: 'center', gap: '6px'
                                }}
                            >
                                <FiFileText size={14} />
                                Save TXT
                            </button>
                            <button
                                onClick={handleExportMd}
                                style={{
                                    padding: '8px 16px', borderRadius: '4px', border: '1px solid #95a5a6',
                                    background: 'transparent', color: '#7f8c8d', cursor: 'pointer',
                                    fontSize: '13px', fontWeight: '500', fontFamily: 'sans-serif',
                                    display: 'flex', alignItems: 'center', gap: '6px'
                                }}
                            >
                                <FiFile size={14} />
                                Save MD
                            </button>
                            <button
                                onClick={handleExportPdf}
                                style={{
                                    padding: '8px 16px', borderRadius: '4px', border: '1px solid #95a5a6',
                                    background: 'transparent', color: '#7f8c8d', cursor: 'pointer',
                                    fontSize: '13px', fontWeight: '500', fontFamily: 'sans-serif',
                                    display: 'flex', alignItems: 'center', gap: '6px'
                                }}
                            >
                                <FiDownload size={14} />
                                Save PDF
                            </button>
                        </>
                    )}

                    {/* Reset button - only show when not recording */}
                    {!status.is_recording && (
                        <button
                            onClick={handleReset}
                            style={{
                                padding: '8px 16px', borderRadius: '4px', border: '1px solid #95a5a6',
                                background: 'transparent', color: '#95a5a6', cursor: 'pointer',
                                fontSize: '13px', fontWeight: '500', fontFamily: 'sans-serif',
                                display: 'flex', alignItems: 'center', gap: '6px'
                            }}
                        >
                            <FiRefreshCw size={14} />
                            Reset
                        </button>
                    )}
                </div>
            </header>

            {/* Audio Level Bar */}
            {status.is_recording && (
                <div style={{
                    padding: '20px 0',
                    borderBottom: '1px solid rgba(0,0,0,0.05)'
                }}>
                    <div style={{
                        display: 'flex',
                        alignItems: 'center',
                        justifyContent: 'space-between',
                        marginBottom: '8px'
                    }}>
                        <span style={{
                            fontSize: '12px',
                            fontWeight: '600',
                            color: '#95a5a6',
                            textTransform: 'uppercase',
                            letterSpacing: '0.05em'
                        }}>
                            Audio Level
                        </span>
                        <span style={{
                            fontSize: '13px',
                            fontWeight: '700',
                            color: '#2c3e50',
                            fontFamily: 'monospace'
                        }}>
                            {(status.audio_level * 100).toFixed(2)}%
                        </span>
                    </div>
                    <div style={{
                        width: '100%',
                        height: '8px',
                        background: 'rgba(0,0,0,0.05)',
                        borderRadius: '4px',
                        overflow: 'hidden',
                        position: 'relative'
                    }}>
                        <motion.div
                            initial={{ width: 0 }}
                            animate={{ width: `${Math.min(100, status.audio_level * 100)}%` }}
                            transition={{ duration: 0.3, ease: 'easeOut' }}
                            style={{
                                height: '100%',
                                background: status.audio_level > 0.8
                                    ? 'linear-gradient(90deg, #e74c3c, #c0392b)'
                                    : status.audio_level > 0.5
                                        ? 'linear-gradient(90deg, #f39c12, #e67e22)'
                                        : 'linear-gradient(90deg, #27ae60, #2ecc71)',
                                borderRadius: '4px',
                                boxShadow: status.audio_level > 0.1 ? '0 0 8px rgba(46, 204, 113, 0.4)' : 'none'
                            }}
                        />
                    </div>
                </div>
            )}

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
                {status.transcript ? (
                    <div style={{ whiteSpace: 'pre-wrap' }}>
                        {status.transcript}
                        {status.is_recording && (
                            <span className="cursor-blink" style={{ display: 'inline-block', width: '2px', height: '1.2em', background: '#2c3e50', marginLeft: '2px', verticalAlign: 'text-bottom' }} />
                        )}
                    </div>
                ) : (
                    <div style={{ color: '#bdc3c7', fontStyle: 'italic' }}>
                        Start recording to begin writing...
                    </div>
                )}
            </div>

        </div>
    );
}


function StatRow({ label, value }) {
    return (
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', fontSize: '14px' }}>
            <span style={{ color: '#5d6d7e' }}>{label}</span>
            <span style={{ fontWeight: '600', color: '#2c3e50' }}>{value}</span>
        </div>
    );
}

function OverviewItem({ label, value }) {
    return (
        <div className="overview-item">
            <span className="overview-label">{label}</span>
            <span className="overview-value">{value}</span>
        </div>
    );
}