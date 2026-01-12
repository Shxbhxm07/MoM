import { NavLink } from "react-router-dom";
import { FiMessageSquare, FiUploadCloud, FiSettings, FiPlus, FiSearch } from "react-icons/fi";

export default function ChatSidebar() {
    return (
        <aside className="chat-sidebar" style={{
            width: '300px',
            background: '#f0f2f5',
            borderRight: '1px solid #e4e6eb',
            display: 'flex',
            flexDirection: 'column',
            height: '100vh'
        }}>
            {/* Header */}
            <div style={{ padding: '20px 16px', display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                <h1 style={{ margin: 0, fontSize: '20px', fontWeight: '700', color: '#050505' }}>Chats</h1>
                <div style={{ display: 'flex', gap: '8px' }}>
                    <button className="icon-btn" style={{ width: '36px', height: '36px', borderRadius: '50%', border: 'none', background: '#e4e6eb', cursor: 'pointer', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
                        <FiSettings size={18} />
                    </button>
                    <button className="icon-btn" style={{ width: '36px', height: '36px', borderRadius: '50%', border: 'none', background: '#e4e6eb', cursor: 'pointer', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
                        <FiPlus size={18} />
                    </button>
                </div>
            </div>

            {/* Search */}
            <div style={{ padding: '0 16px 16px' }}>
                <div style={{ background: '#e4e6eb', borderRadius: '99px', padding: '8px 12px', display: 'flex', alignItems: 'center', gap: '8px' }}>
                    <FiSearch color="#65676b" />
                    <input
                        placeholder="Search sessions"
                        style={{ border: 'none', background: 'transparent', outline: 'none', width: '100%', fontSize: '14px' }}
                    />
                </div>
            </div>

            {/* Navigation / Sessions List */}
            <nav className="chat-nav" style={{ flex: 1, overflowY: 'auto', display: 'flex', flexDirection: 'column' }}>
                <div style={{ padding: '8px 16px', fontSize: '13px', fontWeight: '600', color: '#65676b', textTransform: 'uppercase' }}>Active</div>

                <ChatItem to="/" icon={<FiMessageSquare />} label="Live Session" sub="Recording now..." active />
                <ChatItem to="/upload" icon={<FiUploadCloud />} label="File Transcription" sub="Ready to upload" />

                <div style={{ padding: '16px 16px 8px', fontSize: '13px', fontWeight: '600', color: '#65676b', textTransform: 'uppercase' }}>Recent History</div>
                {/* Mock History Items */}
                <HistoryItem label="Meeting with Team" date="Yesterday" preview="Okay, let's get started..." />
                <HistoryItem label="Project Brainstorm" date="Mon" preview="I think we should focus on..." />
                <HistoryItem label="Client Call" date="Sun" preview="Can you send the invoice?" />
            </nav>
        </aside>
    );
}

function ChatItem({ to, icon, label, sub }) {
    return (
        <NavLink
            to={to}
            className={({ isActive }) => isActive ? "chat-item active" : "chat-item"}
            style={({ isActive }) => ({
                display: 'flex',
                alignItems: 'center',
                gap: '12px',
                padding: '12px 16px',
                textDecoration: 'none',
                background: isActive ? '#e7f3ff' : 'transparent',
                position: 'relative'
            })}
        >
            <div style={{
                width: '48px', height: '48px', borderRadius: '50%',
                background: '#fff', display: 'flex', alignItems: 'center', justifyContent: 'center',
                fontSize: '20px', color: '#1877f2', boxShadow: '0 2px 4px rgba(0,0,0,0.05)'
            }}>
                {icon}
            </div>
            <div style={{ flex: 1, overflow: 'hidden' }}>
                <div style={{ fontSize: '15px', fontWeight: '600', color: '#050505' }}>{label}</div>
                <div style={{ fontSize: '13px', color: '#65676b', whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis' }}>{sub}</div>
            </div>
        </NavLink>
    );
}

function HistoryItem({ label, date, preview }) {
    return (
        <div style={{
            display: 'flex', alignItems: 'center', gap: '12px', padding: '12px 16px', cursor: 'pointer',
            transition: 'background 0.2s'
        }} className="history-item">
            <div style={{
                width: '48px', height: '48px', borderRadius: '50%',
                background: '#e4e6eb', display: 'flex', alignItems: 'center', justifyContent: 'center',
                fontSize: '16px', fontWeight: 'bold', color: '#65676b'
            }}>
                {label[0]}
            </div>
            <div style={{ flex: 1, overflow: 'hidden' }}>
                <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                    <span style={{ fontSize: '15px', fontWeight: '600', color: '#050505' }}>{label}</span>
                    <span style={{ fontSize: '12px', color: '#65676b' }}>{date}</span>
                </div>
                <div style={{ fontSize: '13px', color: '#65676b', whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis' }}>{preview}</div>
            </div>
        </div>
    );
}
