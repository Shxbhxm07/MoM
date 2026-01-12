import { NavLink } from "react-router-dom";
import { FiMic, FiUploadCloud, FiSettings, FiMenu } from "react-icons/fi";
import { motion } from "framer-motion";

export default function TopNav() {
    return (
        <motion.nav
            className="top-nav"
            initial={{ y: -20, opacity: 0 }}
            animate={{ y: 0, opacity: 1 }}
            style={{
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'space-between',
                padding: '16px 32px',
                background: 'rgba(255,255,255,0.8)',
                backdropFilter: 'blur(20px)',
                borderBottom: '1px solid rgba(255,255,255,0.5)',
                position: 'sticky',
                top: 0,
                zIndex: 100
            }}
        >
            <div className="nav-brand" style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
                <div className="logo-mark" style={{
                    width: '36px', height: '36px',
                    background: 'linear-gradient(135deg, #799351, #a1c181)',
                    borderRadius: '50%',
                    color: '#fff',
                    display: 'flex', alignItems: 'center', justifyContent: 'center', fontWeight: 'bold'
                }}>A</div>
                <div className="logo-text">
                    <div className="logo-title" style={{ fontSize: '18px', fontWeight: '700', color: '#2c3e50' }}>AngelBot</div>
                </div>
            </div>

            <div className="nav-links" style={{ display: 'flex', gap: '8px', background: '#f0f2f5', padding: '4px', borderRadius: '99px' }}>
                <NavItem to="/" icon={<FiMic />} label="Live Session" />
                <NavItem to="/upload" icon={<FiUploadCloud />} label="File Transcription" />
            </div>

            <div className="nav-actions">
                <button className="btn-icon" style={{
                    width: '40px', height: '40px', borderRadius: '50%', border: 'none', background: 'transparent', color: '#5d6d7e', cursor: 'pointer', display: 'flex', alignItems: 'center', justifyContent: 'center'
                }}>
                    <FiSettings size={20} />
                </button>
            </div>
        </motion.nav>
    );
}

function NavItem({ to, icon, label }) {
    return (
        <NavLink
            to={to}
            className={({ isActive }) => isActive ? "active-link" : "inactive-link"}
            style={({ isActive }) => ({
                display: 'flex',
                alignItems: 'center',
                gap: '8px',
                padding: '8px 20px',
                borderRadius: '99px',
                textDecoration: 'none',
                fontSize: '14px',
                fontWeight: '500',
                transition: 'all 0.2s ease',
                background: isActive ? '#fff' : 'transparent',
                color: isActive ? '#799351' : '#5d6d7e',
                boxShadow: isActive ? '0 2px 8px rgba(0,0,0,0.05)' : 'none'
            })}
        >
            {icon}
            <span>{label}</span>
        </NavLink>
    );
}
